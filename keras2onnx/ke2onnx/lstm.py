###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numbers
import numpy as np
from collections.abc import Iterable
from ..common import cvtfunc
from ..common.onnx_ops import apply_transpose, apply_reshape, apply_identity, OnnxOperatorBuilder
from ..proto import onnx_proto
from . import simplernn

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


def _calculate_keras_lstm_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None
                                              for i in op.output_shape[0])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None for i in op.output_shape)


@cvtfunc(shape_infer=_calculate_keras_lstm_output_shapes)
def convert_keras_lstm(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    _, seq_length, input_size = simplernn.extract_input_shape(op)


    is_static_shape = seq_length is not None
    if not is_static_shape and container.target_opset < 9:
        raise ValueError('None seq_length is not supported in opset ' + str(container.target_opset))
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    input_name = operator.inputs[0].full_name
    get_name = lambda x: scope.get_unique_variable_name(operator.full_name + x)
    lstm_x_name = get_name('_X')
    tensor_w_name = get_name('_W')
    tensor_r_name = get_name('_R')
    tensor_b_name = get_name('_B')
    sequence_lengths = simplernn.build_sequence_lengths(scope, operator, container)
    initial_h_name = get_name('_initial_h')
    initial_c_name = get_name('_initial_c')

    # Extract the parameters for the LSTM
    W_x, W_h, b = extract_params(op, hidden_size, input_size)

    W = W_x.flatten()
    W_shape = [1, 4 * hidden_size, input_size]
    container.add_initializer(tensor_w_name, TensorProto.FLOAT, W_shape, W)

    R = W_h.flatten()
    R_shape = [1, 4 * hidden_size, hidden_size]
    container.add_initializer(tensor_r_name, TensorProto.FLOAT, R_shape, R)

    if b is not None:
        B = b.flatten()
        B_shape = [1, 8 * hidden_size]
        container.add_initializer(tensor_b_name, TensorProto.FLOAT, B_shape, B)
    else:
        tensor_b_name = ''

    # Output variable names
    lstm_y_name = get_name('_Y')
    lstm_h_name = get_name('_Y_h')
    lstm_c_name = get_name('_Y_c')

    attrs = {}
    # Extract the relevant activation information
    attrs.update(simplernn.extract_activations([
        op.recurrent_activation,
        op.activation,
        op.activation,
    ]))

    # Set up other attributes
    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['hidden_size'] = hidden_size

    # Reshape Keras input format into ONNX input format
    apply_transpose(scope, input_name, lstm_x_name, container, perm=[1, 0, 2])

    # inital_h
    if len(operator.inputs) <= 1:
        initial_h_name = ''
    else:
        # Add a reshape after initial_h, 2d -> 3d
        apply_reshape(scope, operator.inputs[1].full_name, initial_h_name, container,
                      desired_shape=[1, -1, hidden_size])
    # initial_c
    if len(operator.inputs) <= 2:
        initial_c_name = ''
    else:
        # Add a reshape after initial_h, 2d -> 3d
        apply_reshape(scope, operator.inputs[2].full_name, initial_c_name, container,
                      desired_shape=[1, -1, hidden_size])

    oopb = OnnxOperatorBuilder(container, scope)

    input_names = [
        lstm_x_name,
        tensor_w_name,
        tensor_r_name,
        tensor_b_name,
        sequence_lengths,
        initial_h_name,
        initial_c_name,
        '',  # P (optional) : No peep hole in Keras.
    ]

    output_names = [
        lstm_y_name,
        lstm_h_name,
        lstm_c_name,
    ]

    oopb.apply_op_with_output('apply_lstm',
                              input_names,
                              output_names,
                              name=operator.raw_operator.name,
                              output_seq=output_seq,
                              **attrs)

    # Create output-adjusting operators
    if output_seq:
        lstm_y_name_transposed = scope.get_unique_variable_name('lstm_y_transposed')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, lstm_y_name, lstm_y_name_transposed, container, perm=perm)
        if is_static_shape:
            apply_reshape(scope, lstm_y_name_transposed, operator.outputs[0].full_name, container,
                          desired_shape=[-1, seq_length, hidden_size])
        else:
            input_shape_tensor = oopb.add_node('Shape',
                                               [operator.input_full_names[0]],
                                               operator.inputs[0].full_name + '_input_shape_tensor')

            if container.target_opset >= 10:
                seq_len_tensor = oopb.add_node('Slice',
                                               [input_shape_tensor,
                                                ('_start', oopb.int64, np.array([1], dtype='int64')),
                                                ('_end', oopb.int64, np.array([2], dtype='int64')),
                                                ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                                ],
                                               operator.inputs[0].full_name + '_seq_len_tensor')
            else:
                seq_len_tensor = oopb.add_node('Slice',
                                               [input_shape_tensor],
                                               operator.inputs[0].full_name + '_seq_len_tensor', starts=[1], ends=[2],
                                               axes=[0])

            shape_tensor = oopb.add_node('Concat',
                                         [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                          seq_len_tensor,
                                          ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                          ],
                                         operator.inputs[0].full_name + '_output_seq_shape', axis=0)
            shape_tensor_output = oopb.add_node('Reshape',
                                                [lstm_y_name_transposed,
                                                 shape_tensor
                                                ],
                                                operator.inputs[0].full_name + '_output_seq_shape_1')
            apply_identity(scope, shape_tensor_output, operator.outputs[0].full_name, container)
    else:
        apply_reshape(scope, lstm_h_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        # state_h
        apply_reshape(scope, lstm_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
        # state_c
        apply_reshape(scope, lstm_c_name, operator.outputs[2].full_name, container, desired_shape=[-1, hidden_size])
