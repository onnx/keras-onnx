# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .common import activation_process
from ..proto import onnx_proto, keras
from ..common.onnx_ops import apply_add, OnnxOperatorBuilder
activation_get = keras.activations.get


def convert_keras_dense(scope, operator, container):
    op = operator.raw_operator
    parameters = op.get_weights()

    # Allocate weight matrix
    weight = parameters[0]
    weight_name = container.add_initializer_by_name(scope, op.weights[0].name, onnx_proto.TensorProto.FLOAT,
                                                    weight.shape, weight.flatten())

    # Do a numpy matmul. If the input is 2-D, it will be a standard matrix multiplication. Otherwise, it follows Numpy's
    # matmul behavior.
    oopb = OnnxOperatorBuilder(container, scope)
    transformed_tensor_name = oopb.apply_matmul([operator.inputs[0].full_name, weight_name],
                                                name=operator.raw_operator.name)

    # Allocate bias vector
    if len(parameters) == 1:
        bias = np.zeros((weight.shape[1],), dtype=np.float32)
        bias_name = scope.get_unique_variable_name('B')
        container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT, bias.shape, bias.flatten())
    else:
        bias = parameters[1]
        bias_name = container.add_initializer_by_name(scope, op.weights[1].name, onnx_proto.TensorProto.FLOAT,
                                                      bias.shape, bias.flatten())

    # Add bias
    biased_tensor_name = scope.get_unique_variable_name('biased_tensor_name')
    apply_add(scope, transformed_tensor_name + [bias_name], biased_tensor_name, container,
              axis=-1, broadcast=1)

    activation_process(scope, operator, container, biased_tensor_name)
