###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto
from ..common.onnx_ops import apply_reshape, apply_transpose
from .common import extract_recurrent_activation


def convert_keras_simple_rnn(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    input_shape = op.get_input_shape_at(0)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    input_size = input_shape[-1]
    seq_length = input_shape[-2]
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    attrs = {'name': operator.full_name}
    rnn_input_names = []
    rnn_output_names = []

    rnn_x_name = scope.get_unique_variable_name('rnn_x')
    apply_transpose(scope, operator.inputs[0].full_name, rnn_x_name, container, perm=[1, 0, 2])
    rnn_input_names.append(rnn_x_name)

    tensor_w_name = scope.get_unique_variable_name('tensor_w')
    W = op.get_weights()[0].T
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT, [1, hidden_size, input_size], W.flatten())
    rnn_input_names.append(tensor_w_name)

    tensor_r_name = scope.get_unique_variable_name('tensor_r')
    R = op.get_weights()[1].T
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT, [1, hidden_size, hidden_size], R.flatten())
    rnn_input_names.append(tensor_r_name)

    if op.use_bias:
        tensor_b_name = scope.get_unique_variable_name('tensor_b')
        B = np.concatenate([op.get_weights()[2], np.zeros(hidden_size)])
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 2 * hidden_size], B.flatten())
        rnn_input_names.append(tensor_b_name)
    else:
        rnn_input_names.append('')

    # sequence_lens is not able to be converted from input_length
    rnn_input_names.append('')
    # inital_h: none
    if len(operator.inputs) == 1:
        rnn_input_names.append('')
    else:
        # Add a reshape after initial_h, 2d -> 3d
        input_reshape_name = scope.get_unique_variable_name('input_reshape')
        apply_reshape(scope, operator.inputs[1].full_name, input_reshape_name, container,
                      desired_shape=[1, -1, hidden_size])
        rnn_input_names.append(input_reshape_name)

    if hasattr(op, 'activation'):
        activation_type, alpha, beta = extract_recurrent_activation(op.activation)
        attrs['activations'] = [activation_type.encode('utf-8')]
        if alpha is not None:
            attrs['activation_alpha'] = [alpha]
        if beta is not None:
            attrs['activation_beta'] = [beta]

    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['hidden_size'] = hidden_size

    # Set up version-dependent attributes
    if container.target_opset <= 5:
        attrs['output_sequence'] = 1 if output_seq else 0
        op_version = 1
    else:
        op_version = 7

    # We use the collected information to build ONNX's RNN. ONNX RNN's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    rnn_y_name = scope.get_unique_variable_name('rnn_y')
    rnn_h_name = scope.get_unique_variable_name('rnn_h')
    rnn_output_names.append(rnn_y_name)
    rnn_output_names.append(rnn_h_name)
    container.add_node('RNN', rnn_input_names, rnn_output_names, op_version=op_version, **attrs)

    # Create operators to adjust ONNX output to meet Keras format
    if output_seq:
        permuted_rnn_y_name = scope.get_unique_variable_name('rnn_y_permuted')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, rnn_y_name, permuted_rnn_y_name, container, perm=perm)
        apply_reshape(scope, permuted_rnn_y_name, operator.outputs[0].full_name, container,
                      desired_shape=[-1, seq_length, hidden_size])
    else:
        # Here we ingore ONNX RNN's first output because it's useless.
        apply_reshape(scope, rnn_h_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        apply_reshape(scope, rnn_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
