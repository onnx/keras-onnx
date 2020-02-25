###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import tensorflow as tf
from ..proto import keras, is_tf_keras, is_keras_older_than
from ..proto.tfcompat import is_tf2

_layer = keras.layers
_adv_activations = keras.layers.advanced_activations


def _default_layer_name_extractor(fstr_list, node_name):
    for fstr in fstr_list:
        idx = fstr.rfind('{}/')
        if node_name.endswith(fstr[idx + 3:]):
            klen = len(fstr) + idx - 2  # 2 = len('{}')
            return node_name[:len(node_name) - klen]

    return None


def _simple_layer_name_extractor(fstr_list, node_name):
    ri = node_name.rindex('/')
    return node_name[:ri]


def _conv_layer_spec_outputs(layer, node):
    if type(layer) == _layer.DepthwiseConv2D:
        ri = node.name.rindex('/')
        return node.name[:ri + 1] + 'BiasAdd'

    activation_map = {
        keras.activations.linear: '',
        tf.nn.sigmoid: 'Sigmoid',
        tf.nn.softmax: 'Softmax',
        tf.nn.relu: 'Relu',
        tf.nn.elu: 'Elu',
        tf.nn.tanh: 'Tanh'}

    node_act = activation_map.get(layer.activation, None)
    assert node_act is not None, "Unsupported activation in the layer({})".format(layer.activation)
    if node_act:
        ri = node.name.rindex('/')
        return node.name[:ri + 1] + node_act
    else:
        return node.name


def _relu_like_spec_outputs(layer, node):
    if isinstance(layer, _adv_activations.PReLU):
        ri = node.name.rindex('/')
        return node.name[:ri + 1] + 'add'

    return node.name


_keras_layer_spec = {
    # layer-type: ([pattern-list], [extract-layer-name, output-name-generator(optional)]
    _layer.AveragePooling1D: (["{}/AvgPool"], [_default_layer_name_extractor]),
    _layer.AveragePooling2D: (["{}/AvgPool"], [_default_layer_name_extractor]),
    _layer.AveragePooling3D: (["{}/AvgPool"], [_default_layer_name_extractor]),
    _layer.Conv2DTranspose: (["{}/conv2d_transpose"], [_simple_layer_name_extractor, _conv_layer_spec_outputs]),
    _layer.DepthwiseConv2D: (["{}/depthwise"], [_simple_layer_name_extractor, _conv_layer_spec_outputs]),
    _layer.LeakyReLU: (["{}/LeakyRelu"], [_default_layer_name_extractor]),
    _adv_activations.PReLU: (["{}/Relu"], [_simple_layer_name_extractor, _relu_like_spec_outputs])
}

if not is_keras_older_than('2.2.0'):
    _keras_layer_spec.update({
        _adv_activations.ReLU: (["{}/Relu"], [_simple_layer_name_extractor, _relu_like_spec_outputs]),
    })

if is_tf_keras and is_tf2:
    _keras_layer_spec.update({
        _layer.normalization_v2.BatchNormalization: (
            ["{}/FusedBatchNormV3", "{}/batchnorm/add_1"], [_default_layer_name_extractor])
    })


def keras_layer_spec(layer_type):
    return _keras_layer_spec.get(layer_type, (None, []))
