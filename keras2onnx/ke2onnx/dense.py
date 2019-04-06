###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto, keras
from ..common.onnx_ops import apply_softmax, apply_add
from .activation import activation_map
activation_get = keras.activations.get


def convert_keras_dense(scope, operator, container):
    parameters = operator.raw_operator.get_weights()

    # Allocate weight matrix
    weight_name = scope.get_unique_variable_name('W')
    weight = parameters[0]
    container.add_initializer(weight_name, onnx_proto.TensorProto.FLOAT, weight.shape, weight.flatten())

    # Do a numpy matmul. If the input is 2-D, it will be a standard matrix multiplication. Otherwise, it follows Numpy's
    # matmul behavior.
    op_version = 1 if container.target_opset < 9 else 9
    transformed_tensor_name = scope.get_unique_variable_name('transformed_tensor')
    container.add_node('MatMul', [operator.inputs[0].full_name, weight_name], transformed_tensor_name,
                       name=operator.full_name, op_version=op_version)

    # Allocate bias vector
    bias = parameters[1] if len(parameters) > 1 else np.zeros((weight.shape[1],), dtype=np.float32)
    bias_name = scope.get_unique_variable_name('B')
    container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT, bias.shape, bias.flatten())

    # Add bias
    biased_tensor_name = scope.get_unique_variable_name('biased_tensor_name')
    apply_add(scope, [transformed_tensor_name, bias_name], biased_tensor_name, container,
              axis=-1, broadcast=1)

    # Create an activation function node and apply activation function to the intermediate tensor
    apply_activation_function = activation_map[operator.raw_operator.activation]
    if apply_activation_function in [activation_get('softmax'), keras.activations.softmax]:
        apply_softmax(scope, biased_tensor_name, operator.outputs[0].full_name, container, axis=-1)
    else:
        apply_activation_function(scope, biased_tensor_name, operator.outputs[0].full_name, container)
