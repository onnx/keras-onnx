###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto
from ..common.onnx_ops import apply_reshape, apply_transpose, OnnxOperatorBuilder
from . import simplernn

TensorProto = onnx_proto.TensorProto


def extract_params(op):
    """Returns a tuple of the GRU paramters, and converts them into the format for ONNX.
    """
    params = op.get_weights()
    W = params[0].T
    R = params[1].T

    B = None
    if op.use_bias:
        B = params[2]

    return W, R, B


def convert_keras_gru(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    _, seq_length, input_size = simplernn.extract_input_shape(op)
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    attrs = {}

    input_name = operator.inputs[0].full_name
    get_name = lambda x: scope.get_unique_variable_name(operator.full_name + x)

    # Inputs
    gru_x_name = get_name('_X')
    tensor_w_name = get_name('_W')
    tensor_r_name = get_name('_R')
    tensor_b_name = ''
    sequence_lengths = simplernn.build_sequence_lengths(scope, operator, container)
    initial_h_name = simplernn.build_initial_states(scope, operator, container)

    W, R, B = extract_params(op)
    W_shape = [1, 3 * hidden_size, input_size]
    R_shape = [1, 3 * hidden_size, hidden_size]

    container.add_initializer(tensor_w_name, TensorProto.FLOAT, W_shape, W.flatten())
    container.add_initializer(tensor_r_name, TensorProto.FLOAT, R_shape, R.flatten())

    if B is not None and len(B) > 0:
        if B.size == 3 * hidden_size:
            B = np.concatenate([B, np.zeros(3 * hidden_size)])
        tensor_b_name = get_name('_B')
        B_shape = [1, 6 * hidden_size]
        container.add_initializer(tensor_b_name, TensorProto.FLOAT, B_shape, B.flatten())

    attrs.update(simplernn.extract_activations([op.recurrent_activation, op.activation]))

    # Set up other attributes
    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['hidden_size'] = hidden_size

    input_names = [
        gru_x_name,
        tensor_w_name,
        tensor_r_name,
        tensor_b_name,
        sequence_lengths,
        initial_h_name,
    ]

    # We use the collected information to build ONNX's GRU. ONNX GRU's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    gru_y_name = scope.get_unique_variable_name('gru_y')
    gru_h_name = scope.get_unique_variable_name('gru_h')
    gru_output_names = [gru_y_name, gru_h_name]

    apply_transpose(scope, input_name, gru_x_name, container, perm=[1, 0, 2])

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_gru',
                              input_names,
                              gru_output_names,
                              name=operator.raw_operator.name,
                              output_seq=output_seq,
                              reset_after=op.reset_after,
                              **attrs)

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
