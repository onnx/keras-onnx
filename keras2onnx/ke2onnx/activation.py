###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
import tensorflow as tf
from ..proto import keras, is_tf_keras
from ..common.onnx_ops import apply_elu, apply_hard_sigmoid, apply_leaky_relu, apply_relu, apply_relu_6, \
    apply_tanh, apply_softmax, apply_identity, apply_selu, apply_mul, apply_prelu, apply_sigmoid
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

activation_get = keras.activations.get

relu6 = None
if not is_tf_keras:
    try:
        from keras_applications.mobilenet import relu6
    except ImportError:
        pass
if not relu6 and hasattr(keras.applications.mobilenet, 'relu6'):
    relu6 = keras.applications.mobilenet.relu6


def apply_leaky_relu_keras(scope, input_name, output_name, container, operator_name=None, alpha=0.2):
    apply_leaky_relu(scope, input_name, output_name, container, operator_name, alpha)


activation_map = {activation_get('sigmoid'): apply_sigmoid,
                  activation_get('softmax'): apply_softmax,
                  activation_get('linear'): apply_identity,
                  activation_get('relu'): apply_relu,
                  activation_get('elu'): apply_elu,
                  activation_get('selu'): apply_selu,
                  activation_get('tanh'): apply_tanh,
                  activation_get('hard_sigmoid'): apply_hard_sigmoid,
                  tf.nn.leaky_relu: apply_leaky_relu_keras,
                  tf.nn.sigmoid: apply_sigmoid,
                  tf.nn.softmax: apply_softmax,
                  tf.nn.relu: apply_relu,
                  tf.nn.relu6: apply_relu_6,
                  tf.nn.elu: apply_elu,
                  tf.nn.selu: apply_selu,
                  tf.nn.tanh: apply_tanh}

if hasattr(tf.compat, 'v1'):
    activation_map.update({tf.compat.v1.nn.sigmoid: apply_sigmoid})
    activation_map.update({tf.compat.v1.nn.softmax: apply_softmax})
    activation_map.update({tf.compat.v1.nn.leaky_relu: apply_leaky_relu_keras})
    activation_map.update({tf.compat.v1.nn.relu: apply_relu})
    activation_map.update({tf.compat.v1.nn.relu6: apply_relu_6})
    activation_map.update({tf.compat.v1.nn.elu: apply_elu})
    activation_map.update({tf.compat.v1.nn.selu: apply_selu})
    activation_map.update({tf.compat.v1.nn.tanh: apply_tanh})


def convert_keras_activation(scope, operator, container):
    input_name = operator.input_full_names[0]
    output_name = operator.output_full_names[0]
    activation = operator.raw_operator.activation
    activation_type = type(activation)
    if activation in [activation_get('sigmoid'), keras.activations.sigmoid]:
        apply_sigmoid(scope, input_name, output_name, container)
    elif activation in [activation_get('tanh'), keras.activations.tanh]:
        apply_tanh(scope, input_name, output_name, container)
    elif activation in [activation_get('relu'), keras.activations.relu] or \
            (hasattr(keras.layers.advanced_activations, 'ReLU') and
             activation_type == keras.layers.advanced_activations.ReLU):
        apply_relu(scope, input_name, output_name, container)
    elif activation in [activation_get('softmax'), keras.activations.softmax] or \
            activation_type == keras.layers.advanced_activations.Softmax:
        apply_softmax(scope, input_name, output_name, container, axis=-1)
    elif activation in [activation_get('elu'), keras.activations.elu] or \
            activation_type == keras.layers.advanced_activations.ELU:
        apply_elu(scope, input_name, output_name, container, alpha=1.0)
    elif activation in [activation_get('hard_sigmoid'), keras.activations.hard_sigmoid]:
        apply_hard_sigmoid(scope, input_name, output_name, container, alpha=0.2, beta=0.5)
    elif activation in [activation_get('linear'), keras.activations.linear]:
        apply_identity(scope, input_name, output_name, container)
    elif activation in [activation_get('selu'), keras.activations.selu]:
        apply_selu(scope, input_name, output_name, container, alpha=1.673263, gamma=1.050701)
    elif activation_type == keras.layers.advanced_activations.LeakyReLU:
        apply_leaky_relu(scope, input_name, output_name, container, alpha=activation.alpha.item(0))
    elif activation_type == keras.layers.advanced_activations.PReLU:
        apply_prelu(scope, input_name, output_name, container, slope=operator.raw_operator.get_weights()[0])
    elif activation in [relu6] or (hasattr(activation, '__name__') and activation.__name__ == 'relu6'):
        # relu6(x) = min(relu(x), 6)
        np_type = TENSOR_TYPE_TO_NP_TYPE[operator.inputs[0].type.to_onnx_type().tensor_type.elem_type]
        zero_value = np.zeros(shape=(1,), dtype=np_type)
        apply_relu_6(scope, input_name, output_name, container,
                     zero_value=zero_value)
    elif hasattr(activation, '__name__') and activation.__name__ == 'swish':
        apply_sigmoid(scope, input_name, output_name + '_sig', container)
        apply_mul(scope, [input_name, output_name + '_sig'], output_name, container)
    else:
        if activation in [activation_get('softsign'), keras.activations.softsign]:
            op_type = 'Softsign'
        elif activation in [activation_get('softplus'), keras.activations.softplus]:
            op_type = 'Softplus'
        else:
            raise RuntimeError("Unsupported activation method within Activation layer '{}'".format(activation))

        container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=operator.full_name)
