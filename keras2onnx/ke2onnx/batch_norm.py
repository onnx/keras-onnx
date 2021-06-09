# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common.onnx_ops import apply_batch_norm, apply_transpose
from ..proto import onnx_proto


def convert_keras_batch_normalization(scope, operator, container):
    op = operator.raw_operator
    shape_len = len(operator.get_input_shape())

    if isinstance(op.axis, list):
        if len(op.axis) == 1:
            axis = op.axis[0]
        else:
            raise AttributeError('No support for more than one axis in: ' + operator.full_name)
    else:
        axis = op.axis

    skip_transpose = (axis != shape_len - 1 and axis != -1) or shape_len <= 2
    if not skip_transpose:
        perm_1 = list(range(1, shape_len - 1))
        perm_1 = [0, shape_len - 1] + perm_1
        perm_2 = list(range(2, shape_len))
        perm_2 = [0] + perm_2 + [1]

    if skip_transpose:
        adjusted_input_name = operator.inputs[0].full_name
    else:
        adjusted_input_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_transposed')
        apply_transpose(scope, operator.inputs[0].full_name, adjusted_input_name, container, perm=perm_1)

    input_tensor_names = [adjusted_input_name]

    params = op.get_weights()
    # If scale and/or center flag is set in keras node, use keras default values for gamma and/or beta
    if not op.scale:
        params.insert(0, np.ones(params[0].shape, dtype=float))
    if not op.center:
        params.insert(1, np.zeros(params[1].shape, dtype=float))

    gamma = params[0] / np.sqrt(params[3] + op.epsilon)
    beta = params[1] - params[0] * params[2] / np.sqrt(params[3] + op.epsilon)

    scale_tensor_name = scope.get_unique_variable_name('scale')
    container.add_initializer(scale_tensor_name, onnx_proto.TensorProto.FLOAT, params[0].shape, gamma)
    input_tensor_names.append(scale_tensor_name)

    bias_tensor_name = scope.get_unique_variable_name('bias')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, params[1].shape, beta)
    input_tensor_names.append(bias_tensor_name)

    mean_tensor_name = scope.get_unique_variable_name('mean')
    container.add_initializer(mean_tensor_name, onnx_proto.TensorProto.FLOAT, params[2].shape, 0 * params[2])
    input_tensor_names.append(mean_tensor_name)

    var_tensor_name = scope.get_unique_variable_name('var')
    container.add_initializer(var_tensor_name, onnx_proto.TensorProto.FLOAT, params[3].shape, 1 + 0 * params[3])
    input_tensor_names.append(var_tensor_name)

    epsilon = op.epsilon * 1e-3  # We use a much smaller epsilon because the original epsilon is absorbed in gamma
    is_test = 1
    momentum = op.momentum
    spatial = 1

    if skip_transpose:
        # If no transpose is required, we can simply use the output of ONNX BatchNorm as the final outcome
        # ORT assumes opitonal outputs indicate training mode. So we should use one output for inference.
        apply_batch_norm(scope, input_tensor_names, operator.output_full_names[0], container,
                         operator_name=operator.full_name, epsilon=epsilon, is_test=is_test,
                         momentum=momentum, spatial=spatial)
    else:
        # If transpose is required, we need to put BatchNorm's output to an intermediate tensor for applying a transpose
        intermediate_output_name = scope.get_unique_variable_name('batch_norm_output_buffer')
        apply_batch_norm(scope, input_tensor_names, intermediate_output_name, container,
                         operator_name=operator.full_name, epsilon=epsilon, is_test=is_test,
                         momentum=momentum, spatial=spatial)

        # For 4D case, this is to permute [N,C,H,W] to [N,H,W,C]
        apply_transpose(scope, intermediate_output_name, operator.outputs[0].full_name, container, perm=perm_2)
