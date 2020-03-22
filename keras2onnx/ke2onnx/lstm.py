###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numbers
import numpy as np
from collections.abc import Iterable
from ..common import cvtfunc, name_func
from ..common.onnx_ops import (
    apply_constant,
    apply_identity,
    apply_reshape,
    apply_slice,
    apply_split,
    apply_squeeze,
    apply_transpose,
    OnnxOperatorBuilder
)
from ..proto import onnx_proto, keras
from . import simplernn

LSTM = keras.layers.LSTM
TensorProto = onnx_proto.TensorProto


def check_sequence_lengths(operator, container):
    """Raises an exception if the shape is expected to be static, but the sequence lenghts
    are not provided. This only applies to opsets below 9.
    """
    op = operator.raw_operator

    _, seq_length, input_size = simplernn.extract_input_shape(op)
    is_static_shape = seq_length is not None
    if not is_static_shape and container.target_opset < 9:
        raise ValueError('None seq_length is not supported in opset ' + str(container.target_opset))


def convert_ifco_to_iofc(tensor_ifco):
    """Returns a tensor in input (i), output (o), forget (f), cell (c) ordering. The
    Keras ordering is ifco, while the ONNX ordering is iofc.
    """
    splits = np.split(tensor_ifco, 4)
    return np.concatenate((splits[0], splits[3], splits[1], splits[2]))


def extract_params(op, hidden_size, input_size):
    """Returns a tuple of the LSTM parameters, and converts them into the format for ONNX.
    """
    params = op.get_weights()

    # Keras: [W_x, W_h, b] each in I F C O
    # ONNX: W[iofc] I O F C
    W_x = convert_ifco_to_iofc(params[0].T).reshape(4, hidden_size, input_size)
    W_h = convert_ifco_to_iofc(params[1].T).reshape(4, hidden_size, hidden_size)

    b = None
    if op.use_bias:
        b = np.zeros((8, hidden_size), dtype=np.float32)
        b[:4] = convert_ifco_to_iofc(params[2]).reshape(4, hidden_size)

    return W_x, W_h, b

def build_parameters(scope, operator, container, bidirectional=False):
    """Returns the parameter initialization values after extracting them from the LSTM layer.
    """
    op = operator.raw_operator
    _, seq_length, input_size = simplernn.extract_input_shape(op)

    _name = name_func(scope, operator)

    tensor_w = _name('W')
    tensor_r = _name('R')
    tensor_b = ''

    if bidirectional:
        forward_layer = op.forward_layer
        backward_layer = op.backward_layer
        hidden_size = forward_layer.units

        W_x, W_h, b = extract_params(forward_layer, hidden_size, input_size)
        W_x_back, W_h_back, b_back = extract_params(backward_layer, hidden_size, input_size)

        W = np.concatenate([W_x, W_x_back]).flatten()
        W_shape = [2, 4 * hidden_size, input_size]

        R = np.concatenate([W_h, W_h_back]).flatten()
        R_shape = [2, 4 * hidden_size, hidden_size]

        if (b is None and b_back is not None) or (b is not None and b_back is None):
            raise ValueError('Bidirectional bias must be enabled (or disabled) for both forward '
                             'and backward layers.')

        if b is not None:
            B = np.concatenate([b, b_back]).flatten()
            B_shape = [2, 8 * hidden_size]

    else:
        hidden_size = op.units

        W_x, W_h, b = extract_params(op, hidden_size, input_size)

        W = W_x.flatten()
        W_shape = [1, 4 * hidden_size, input_size]

        R = W_h.flatten()
        R_shape = [1, 4 * hidden_size, hidden_size]

        if b is not None:
            B = b.flatten()
            B_shape = [1, 8 * hidden_size]

    # Create initializers
    container.add_initializer(tensor_w, TensorProto.FLOAT, W_shape, W)
    container.add_initializer(tensor_r, TensorProto.FLOAT, R_shape, R)

    if b is not None:
        tensor_b = _name('B')
        container.add_initializer(tensor_b, TensorProto.FLOAT, B_shape, B)


    return tensor_w, tensor_r, tensor_b

def build_initial_states(scope, operator, container, bidirectional=False):
    """
    """
    _name = name_func(scope, operator)

    initial_h = ''
    initial_c = ''

    if bidirectional:
        input_name = operator.inputs[0].full_name

        _, seq_length, input_size = simplernn.extract_input_shape(operator.raw_operator)
        is_static_shape = seq_length is not None
        hidden_size = operator.raw_operator.forward_layer.units

        oopb = OnnxOperatorBuilder(container, scope)

        if container.target_opset < 9:
            # need the zero initializer to correct some engine shape inference bug.
            # TODO: Fix the fixed batch size for this case
            state_shape = (2, 1, hidden_size)
            h_0 = np.zeros(shape=state_shape).flatten()
            c_0 = np.zeros(shape=state_shape).flatten()

            initial_h = _name('initial_h')
            initial_c = _name('initial_c')
            container.add_initializer(initial_h, TensorProto.FLOAT, state_shape, h_0)
            container.add_initializer(initial_c, TensorProto.FLOAT, state_shape, c_0)

    else:

        initial_h = simplernn.build_initial_states(scope, operator, container)

        if len(operator.inputs) > 1:
            # Add a reshape after initial_h, 2d -> 3d
            hidden_size = operator.raw_operator.units
            input_c = operator.inputs[2].full_name
            initial_c = _name('initial_c')
            apply_reshape(scope, operator.inputs[2].full_name, initial_c, container,
                          desired_shape=[1, -1, hidden_size])

    return initial_h, initial_c


def build_attributes(scope, operator, container, bidirectional=False):
    """Returns a dictionary of attributes for the LSTM layer.
    """
    op = operator.raw_operator

    attrs = {}

    if bidirectional:
        forward_layer = op.forward_layer
        backward_layer = op.backward_layer

        attrs['direction'] = 'bidirectional'
        attrs['hidden_size'] = forward_layer.units
        attrs.update(simplernn.extract_activations([
            forward_layer.recurrent_activation,
            forward_layer.activation,
            forward_layer.activation,
            backward_layer.recurrent_activation,
            backward_layer.activation,
            backward_layer.activation,
        ]))

    else:
        attrs['direction'] = 'reverse' if op.go_backwards else 'forward'
        attrs['hidden_size'] = op.units
        attrs.update(simplernn.extract_activations([
            op.recurrent_activation,
            op.activation,
            op.activation,
        ]))
    return attrs

def build_output(scope, operator, container, output_names, bidirectional=False):
    """
    """
    if bidirectional:
        return build_output_bidirectional(scope, operator, container, output_names)

    lstm_y, lstm_h, lstm_c = output_names

    op = operator.raw_operator
    hidden_size = op.units
    output_seq = op.return_sequences
    output_state = op.return_state
    _, seq_length, input_size = simplernn.extract_input_shape(op)
    is_static_shape = seq_length is not None

    _name = name_func(scope, operator)

    output_name = operator.outputs[0].full_name

    oopb = OnnxOperatorBuilder(container, scope)

    # Create output-adjusting operators
    if output_seq:
        transposed_y = _name('y_transposed')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, lstm_y, transposed_y, container, perm=perm)

        if is_static_shape:
            apply_reshape(scope, transposed_y, output_name, container,
                          desired_shape=[-1, seq_length, hidden_size])
        else:
            input_name = operator.inputs[0].full_name
            input_shape_tensor = oopb.add_node('Shape', [input_name],
                                               input_name + '_shape_tensor')

            seq_len_tensor = _name('seq_len_tensor')
            apply_slice(scope, input_shape_tensor, seq_len_tensor, container, [1], [2], axes=[0])

            shape_tensor = oopb.add_node('Concat',
                                         [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                          seq_len_tensor,
                                          ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                          ],
                                         input_name + '_output_seq_shape', axis=0)
            shape_tensor_output = oopb.add_node('Reshape',
                                                [transposed_y, shape_tensor],
                                                input_name + '_output_seq_shape_1')
            apply_identity(scope, shape_tensor_output, output_name, container)
    else:
        apply_reshape(scope, lstm_h, output_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        # Output hidden and cell states
        apply_reshape(scope, lstm_h, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
        apply_reshape(scope, lstm_c, operator.outputs[2].full_name, container, desired_shape=[-1, hidden_size])


def build_output_bidirectional(scope, operator, container, output_names):
    """
    """
    op = operator.raw_operator
    forward_layer = op.forward_layer
    backward_layer = op.backward_layer

    _, seq_length, input_size = simplernn.extract_input_shape(op)
    is_static_shape = seq_length is not None
    hidden_size = forward_layer.units
    output_seq = forward_layer.return_sequences
    output_state = forward_layer.return_state
    if output_state:
        raise ValueError('Keras Bidirectional cannot return hidden and cell states')
    if not isinstance(forward_layer, LSTM):
        raise TypeError('The bidirectional module only works with LSTM in Keras but we got %s' % type(forward_layer))


    _name = name_func(scope, operator)

    lstm_y, lstm_h, lstm_c = output_names
    input_name = operator.inputs[0].full_name

    oopb = OnnxOperatorBuilder(container, scope)

    # Define seq_len_tensor
    input_shape_tensor = oopb.add_node('Shape', [input_name], input_name + '_input_shape_tensor')

    batch_indices_tensor = input_name + '_batch_indices_tensor'
    apply_slice(scope, input_shape_tensor, batch_indices_tensor, container, [0], [1], axes=[0])

    if not is_static_shape:
        seq_len_tensor = input_name + '_seq_len_tensor'
        apply_slice(scope, input_shape_tensor, seq_len_tensor, container, [1], [2], axes=[0])


    if hasattr(op, 'merge_mode'):
        if op.merge_mode not in ['concat', None]:
            raise ValueError('Only support Bidirectional with merge_mode=\'concat\' but got %s' % op.merge_mode)
        merge_concat = False if op.merge_mode is None else True
    else:
        merge_concat = False

    if output_seq:
        # The output shape of runtime is 3-D while ONNX says 4-D, so we do a Reshape to fix it.
        if is_static_shape:
            lstm_y_fixed = _name('Y_fixed')
            apply_reshape(scope, lstm_y, lstm_y_fixed, container,
                          desired_shape=[seq_length, 2, -1, hidden_size])
        else:
            shape_tensor = oopb.add_node('Concat',
                                         [seq_len_tensor,
                                          ('_a', oopb.int64, np.array([2], dtype='int64')),
                                          ('_b', oopb.int64, np.array([-1], dtype='int64')),
                                          ('_c', oopb.int64, np.array([hidden_size], dtype='int64'))
                                          ],
                                         input_name + '_output_seq_shape', axis=0)
            lstm_y_fixed = oopb.add_node('Reshape',
                                              [lstm_y,
                                               shape_tensor
                                               ],
                                              input_name + '_output_seq_shape_1')

        if merge_concat:
            # In this case, only one Keras output with shape (N, T, 2 * C') should be produced

            # Transpose ONNX LSTM Y with shape (T, D, N, C') into (T, N, D, C')
            transposed_y = _name('Y_transposed')
            apply_transpose(scope, lstm_y_fixed, transposed_y, container, perm=[2, 0, 1, 3])

            # Change shape (T, N, D, C') to (N, T, D * C') to meet Keras spec
            if is_static_shape:
                apply_reshape(scope, transposed_y, operator.outputs[0].full_name, container,
                              desired_shape=[-1, seq_length, 2 * hidden_size])
            else:
                attrs = {'axis': 0}
                shape_tensor_2 = oopb.add_node('Concat',
                                               [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                                seq_len_tensor,
                                                ('_b', oopb.int64, np.array([2 * hidden_size], dtype='int64'))
                                                ],
                                               input_name + '_output_seq_shape_2', **attrs)
                shape_tensor_output = oopb.add_node('Reshape',
                                                    [transposed_y,
                                                     shape_tensor_2
                                                     ],
                                                    input_name + '_output_merge_concat')
                apply_identity(scope, shape_tensor_output, operator.outputs[0].full_name, container)
        else:
            # If merge_mode=None, two tensors should be generated. The first/second tensor is the output of
            # forward/backward pass.

            # Transpose ONNX LSTM Y with shape (T, D, N, C') into (T, N, D, C')
            transposed_y = _name('Y_transposed')
            apply_transpose(scope, lstm_y_fixed, transposed_y, container, perm=[2, 0, 1, 3])

            # Split the transposed Y with shape (T, N, D, C') into (T, N, 1, C') and (T, N, 1, C')
            forward_y = _name('Y_forward')
            backward_y = _name('Y_backward')
            axis_direction = 2
            apply_split(scope, transposed_y, [forward_y, backward_y], container, axis=axis_direction)

            # Change (T, N, 1, C') into (T, N, C') to meet Keras spec
            forward_y_1 = _name('Y_forward_1')
            backward_y_1 = _name('Y_backward_1')
            apply_squeeze(scope, forward_y, forward_y_1, container, axes=[axis_direction])
            apply_squeeze(scope, backward_y, backward_y_1, container, axes=[axis_direction])

            if is_static_shape:
                apply_reshape(scope, forward_y_1, operator.outputs[0].full_name, container,
                              desired_shape=[-1, seq_length, hidden_size])
                apply_reshape(scope, backward_y_1, operator.outputs[1].full_name, container,
                              desired_shape=[-1, seq_length, hidden_size])
            else:
                shape_tensor_3 = oopb.add_node('Concat',
                                               [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                                seq_len_tensor,
                                                ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                                ],
                                               input_name + '_output_seq_shape_3', **attrs)
                shape_tensor_output_0 = oopb.add_node('Reshape',
                                                      [forward_y_1,
                                                       shape_tensor_3
                                                       ],
                                                      input_name + '_shape_tensor_output_0')
                shape_tensor_output_1 = oopb.add_node('Reshape',
                                                      [backward_y_1,
                                                       shape_tensor_3
                                                       ],
                                                      input_name + '_shape_tensor_output_1')
                apply_identity(scope, shape_tensor_output_0, operator.outputs[0].full_name, container)
                apply_identity(scope, shape_tensor_output_1, operator.outputs[1].full_name, container)
    else:
        perm = [1, 0, 2]
        if merge_concat:
            # In this case, only one Keras output with shape (N, 2 * C') should be produced

            # Transpose ONNX LSTM Y_h with shape (D, N, C') into (N, D, C')
            transposed_h = _name('Y_h_transposed')
            apply_transpose(scope, lstm_h, transposed_h, container, perm=perm)

            # Flatten ONNX (N, D, C') into (N, D * C')
            oopb.apply_op_with_output("apply_flatten",
                                      transposed_h,
                                      operator.outputs[0].full_name,
                                      name=operator.full_name + '_flatten',
                                      axis=1)
        else:
            # If merge_mode=None, two tensors should be generated. The first/second tensor is the output of
            # forward/backward pass.

            # Transpose ONNX LSTM Y_h with shape (D, N, C') into (N, D, C')
            transposed_h = _name('Y_h_transposed')
            apply_transpose(scope, lstm_h, transposed_h, container, perm=perm)

            # Split the transposed Y with shape (T, N, D, C') into (T, N, 1, C') and (T, N, 1, C')
            forward_y = _name('Y_forward')
            backward_y = _name('Y_backward')
            axis_direction = 1
            apply_split(scope, transposed_h, [forward_y, backward_y], container, axis=axis_direction)

            # Change (T, N, 1, C') into (T, N, C') to meet Keras spec
            apply_squeeze(scope, forward_y, operator.outputs[0].full_name, container, axes=[axis_direction])
            apply_squeeze(scope, backward_y, operator.outputs[1].full_name, container, axes=[axis_direction])


def _calculate_keras_lstm_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None
                                              for i in op.output_shape[0])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None for i in op.output_shape)


@cvtfunc(shape_infer=_calculate_keras_lstm_output_shapes)
def convert_keras_lstm(scope, operator, container, bidirectional=False):
    op = operator.raw_operator
    _name = name_func(scope, operator)

    if bidirectional:
        output_seq = op.forward_layer.return_sequences
    else:
        output_seq = op.return_sequences

    check_sequence_lengths(operator, container)

    # Inputs
    lstm_x = _name('X')
    tensor_w, tensor_r, tensor_b = build_parameters(scope, operator, container, bidirectional)
    sequence_lengths = simplernn.build_sequence_lengths(scope, operator, container)
    initial_h, initial_c = build_initial_states(scope, operator, container, bidirectional)

    input_names = [
        lstm_x,
        tensor_w,
        tensor_r,
        tensor_b,
        sequence_lengths,
        initial_h,
        initial_c,
        '',  # P (optional) : No peep hole in Keras.
    ]

    # Attributes
    attrs = build_attributes(scope, operator, container, bidirectional)

    # Outputs
    output_names = [_name('Y'), _name('Y_h'), _name('Y_c')]

    # Reshape Keras input format into ONNX input format
    input_name = operator.inputs[0].full_name
    apply_transpose(scope, input_name, lstm_x, container, perm=[1, 0, 2])

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_lstm',
                              input_names,
                              output_names,
                              name=op.name,
                              output_seq=output_seq,
                              **attrs)

    build_output(scope, operator, container, output_names, bidirectional)
