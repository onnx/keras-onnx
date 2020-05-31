###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from ..proto import keras
from ..proto.tfcompat import tensorflow as tf
from ..common.onnx_ops import apply_relu6, apply_softmax
from .activation import activation_map
activation_get = keras.activations.get


def get_permutation_config(n_dims):
    input_perm_axes = [0, n_dims + 1] + list(range(1, n_dims + 1))
    output_perm_axes = [0] + list(range(2, n_dims + 2)) + [1]
    return input_perm_axes, output_perm_axes


def activation_process(scope, operator, container, biased_tensor_name):
    # Create an activation function node and apply activation function to the intermediate tensor
    apply_activation_function = activation_map[operator.raw_operator.activation]
    if operator.raw_operator.activation in [activation_get('softmax'), keras.activations.softmax]:
        apply_softmax(scope, biased_tensor_name, operator.outputs[0].full_name, container, axis=-1)
    elif operator.raw_operator.activation in [tf.nn.relu6]:
        apply_relu6(scope, biased_tensor_name, operator.outputs[0].full_name, container,
                    dtype=operator.raw_operator.input.dtype.as_numpy_dtype)
    else:
        apply_activation_function(scope, biased_tensor_name, operator.outputs[0].full_name, container)
