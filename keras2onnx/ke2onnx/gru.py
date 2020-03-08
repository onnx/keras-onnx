###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto
from ..common.onnx_ops import apply_reshape, apply_transpose, OnnxOperatorBuilder
from .common import extract_recurrent_activation
from . import simplernn


def extract_params(op):
    """Returns a tuple of the GRU paramters, and converts them into the format for ONNX.
    """
    params = op.get_weights()
    W = params[0].T
    R = params[1].T
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
    gru_input_names = []

    gru_x_name = scope.get_unique_variable_name('gru_x')
    apply_transpose(scope, operator.inputs[0].full_name, gru_x_name, container, perm=[1, 0, 2])
    gru_input_names.append(gru_x_name)

    W, R, B = extract_params(op)

    tensor_w_name = scope.get_unique_variable_name('tensor_w')
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, input_size], W.flatten())
    gru_input_names.append(tensor_w_name)

    tensor_r_name = scope.get_unique_variable_name('tensor_r')
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, hidden_size], R.flatten())
    gru_input_names.append(tensor_r_name)

    if op.use_bias and len(B) > 0:
        tensor_b_name = scope.get_unique_variable_name('tensor_b')
        if B.size == 3 * hidden_size:
            B = np.concatenate([B, np.zeros(3 * hidden_size)])
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 6 * hidden_size], B.flatten())
        gru_input_names.append(tensor_b_name)
    else:
        gru_input_names.append('')

    # sequence lens
    uses_masking_layer = len(operator.input_masks) == 1
    if uses_masking_layer:
        # Mask using sequence_lens input
        sequence_lengths = scope.get_unique_variable_name(operator.full_name + '_seq_lens')
        gru_input_names.append(sequence_lengths)
    else:
        gru_input_names.append('')
    # inital_h
    if len(operator.inputs) == 1:
        gru_input_names.append('')
    else:
        # Add a reshape after initial_h, 2d -> 3d
        input_reshape_name = scope.get_unique_variable_name('input_reshape')
        apply_reshape(scope, operator.inputs[1].full_name, input_reshape_name, container,
                      desired_shape=[1, -1, hidden_size])
        gru_input_names.append(input_reshape_name)

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

    # We use the collected information to build ONNX's GRU. ONNX GRU's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    gru_y_name = scope.get_unique_variable_name('gru_y')
    gru_h_name = scope.get_unique_variable_name('gru_h')
    gru_output_names = [gru_y_name, gru_h_name]
    oopb = OnnxOperatorBuilder(container, scope)

    if uses_masking_layer:
        mask_cast = oopb.apply_cast(operator.input_masks[0].full_name, to=oopb.int32, name=operator.full_name + '_mask_cast')
        oopb.add_node_with_output('ReduceSum', mask_cast, sequence_lengths, keepdims=False, axes=[-1], name=operator.full_name + '_mask_sum')


    oopb.apply_op_with_output('apply_gru',
                              gru_input_names,
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
