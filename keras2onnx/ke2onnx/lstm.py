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
    apply_concat,
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
    """Builds the initial hidden and cell states for the LSTM layer.
    """
    _name = name_func(scope, operator)

    initial_h = simplernn.build_initial_states(scope, operator, container, bidirectional)

    # Determine if the cell states are set
    has_c = (
        (len(operator.inputs) > 1 and not bidirectional) or
        (len(operator.inputs) > 3 and bidirectional)
    )
    if not has_c:
        return initial_h, ''

    op = operator.raw_operator
    initial_c = _name('initial_c')

    if bidirectional:
        forward_layer = op.forward_layer
        hidden_size = forward_layer.units
        desired_shape = [1, -1, hidden_size]

        # Combine the forward and backward_layers
        forward_h = _name('initial_c_forward')
        backward_h = _name('initial_c_backward')
        apply_reshape(scope, operator.inputs[2].full_name, forward_h, container, desired_shape=desired_shape)
        apply_reshape(scope, operator.inputs[4].full_name, backward_h, container, desired_shape=desired_shape)

        apply_concat(scope, [forward_h, backward_h], initial_c, container)

    else:
        hidden_size = operator.raw_operator.units
        desired_shape = [1, -1, hidden_size]

        # Add a reshape after initial_c, 2d -> 3d
        input_c = operator.inputs[2].full_name
        apply_reshape(scope, input_c, initial_c, container, desired_shape=desired_shape)

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
    """Builds the output operators for the LSTM layer.
    """
    if bidirectional:
        return simplernn.build_output(scope, operator, container, output_names[:-1], bidirectional)

    lstm_y, lstm_h, lstm_c = output_names

    op = operator.raw_operator
    hidden_size = op.units
    output_seq = op.return_sequences
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

            seq_dim = _name('seq_dim')
            apply_slice(scope, input_shape_tensor, seq_dim, container, [1], [2], axes=[0])

            shape_tensor = oopb.add_node('Concat',
                                         [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                          seq_dim,
                                          ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                          ],
                                         input_name + '_output_seq_shape', axis=0)
            shape_tensor_output = oopb.add_node('Reshape',
                                                [transposed_y, shape_tensor],
                                                input_name + '_output_seq_shape_1')
            apply_identity(scope, shape_tensor_output, output_name, container)
    else:
        apply_reshape(scope, lstm_h, output_name, container, desired_shape=[-1, hidden_size])


def build_output_states(scope, operator, container, output_names, bidirectional=False):
    """Builds the output hidden states for the LSTM layer.
    """
    _, lstm_h, lstm_c = output_names
    op = operator.raw_operator

    if bidirectional:
        forward_layer = op.forward_layer
        output_state = forward_layer.return_state

        if not output_state:
            return

        # Split lstm_h and lstm_c into forward and backward components
        squeeze_names = []
        output_names = [o.full_name for o in operator.outputs[1:]]
        name_map = {lstm_h: output_names[::2], lstm_c: output_names[1::2]}

        for state_name, outputs in name_map.items():
            split_names = ['{}_{}'.format(state_name, d) for d in ('forward', 'backward')]

            apply_split(scope, state_name, split_names, container)
            squeeze_names.extend(list(zip(split_names, outputs)))

        for split_name, output_name in squeeze_names:
            apply_squeeze(scope, split_name, output_name, container)

    else:
        output_state = op.return_state

        if not output_state:
            return

        output_h = operator.outputs[1].full_name
        output_c = operator.outputs[2].full_name
        apply_squeeze(scope, lstm_h, output_h, container)
        apply_squeeze(scope, lstm_c, output_c, container)


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
    build_output_states(scope, operator, container, output_names, bidirectional)
