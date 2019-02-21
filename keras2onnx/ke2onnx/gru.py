# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..proto import onnx_proto
from ..common.onnx_ops import apply_reshape, apply_transpose
from .common import extract_recurrent_activation


def convert_keras_gru(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    input_shape = op.get_input_shape_at(0)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    input_size = input_shape[-1]
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    op_type = 'GRU'
    attrs = {'name': operator.full_name}
    gru_input_names = []

    gru_x_name = scope.get_unique_variable_name('gru_x')
    apply_reshape(scope, operator.inputs[0].full_name, gru_x_name, container, desired_shape=[-1, 1, input_size])
    gru_input_names.append(gru_x_name)

    tensor_w_name = scope.get_unique_variable_name('tensor_w')
    W = op.get_weights()[0].T
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, input_size], W.flatten())
    gru_input_names.append(tensor_w_name)

    tensor_r_name = scope.get_unique_variable_name('tensor_r')
    R = op.get_weights()[1].T
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, hidden_size], R.flatten())
    gru_input_names.append(tensor_r_name)

    B = op.get_weights()[2]
    if op.use_bias and len(B) > 0:
        tensor_b_name = scope.get_unique_variable_name('tensor_b')
        if B.size == 3 * hidden_size:
            B = np.concatenate([B, np.zeros(3 * hidden_size)])
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 6 * hidden_size], B.flatten())
        gru_input_names.append(tensor_b_name)
    else:
        gru_input_names.append('')

    # sequence lens
    gru_input_names.append('')
    # TODO: figure out keras way of inital_h
    if len(operator.inputs) == 1:
        gru_input_names.append('')
    else:
        gru_input_names.append(operator.inputs[1].full_name)

    activation_types = []
    alphas = []
    betas = []
    for (activation_type, alpha, beta) in \
            [extract_recurrent_activation(op.recurrent_activation), extract_recurrent_activation(op.activation)]:
        activation_types.append(activation_type.encode('utf-8'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)

    attrs['activations'] = activation_types
    if alphas:
        attrs['activation_alpha'] = alphas
    if betas:
        attrs['activation_beta'] = betas

    # Set up other attributes
    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['hidden_size'] = hidden_size

    # Set up version-dependent attributes
    if container.target_opset < 3:
        op_version = 1
        attrs['output_sequence'] = 1 if output_seq else 0
    elif container.target_opset <= 5:
        attrs['linear_before_reset'] = 0
        attrs['output_sequence'] = 1 if output_seq else 0
        op_version = 3
    else:
        attrs['linear_before_reset'] = 0
        op_version = 7

    # We use the collected information to build ONNX's GRU. ONNX GRU's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    gru_y_name = scope.get_unique_variable_name('gru_y')
    gru_h_name = scope.get_unique_variable_name('gru_h')
    gru_output_names = [gru_y_name, gru_h_name]
    container.add_node(op_type, gru_input_names, gru_output_names, op_version=op_version, **attrs)

    # Create output-adjusting operators
    if output_seq:
        intermediate_result_name = scope.get_unique_variable_name('intermediate_result')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, gru_y_name, intermediate_result_name, container, perm=perm)
        apply_reshape(scope, intermediate_result_name, operator.outputs[0].full_name, container,
                      desired_shape=[-1, 0, hidden_size])
    else:
        # Here we ignore ONNX GRU's first output because it's useless.
        intermediate_result_name = scope.get_unique_variable_name('intermediate_result')
        apply_transpose(scope, gru_h_name, intermediate_result_name, container, perm=[1, 0, 2])
        apply_reshape(scope, intermediate_result_name, operator.outputs[0].full_name, container,
                      desired_shape=[-1, hidden_size])

    if output_state:
        apply_reshape(scope, gru_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
