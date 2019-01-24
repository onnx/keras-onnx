###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six
from keras.layers import *
from keras.layers import advanced_activations as adv_activations

from ..common import with_variable
from ..common.onnx_ops import apply_identity, apply_reshape

from .activation import convert_keras_activation
from .adv_activation import convert_keras_advanced_activation
from .batch_norm import convert_keras_batch_normalization
from .upsample import *
from .conv import *
from .pooling import *
from .crop import *
from .simplernn import convert_keras_simple_rnn
from .gru import convert_keras_gru
from .lstm import convert_keras_lstm
from .bidirectional import convert_bidirectional


def extract_inbound_nodes(layer):
    if hasattr(layer, 'inbound_nodes'):
        return layer.inbound_nodes
    elif hasattr(layer, '_inbound_nodes'):
        return layer._inbound_nodes
    else:
        raise ValueError("Failed to find inbound_nodes and _inbound_nodes when parsing %s" % layer.name)


def convert_keras_reshape(scope, operator, container):
    iop = operator.raw_operator
    target_shape = iop.target_shape
    if operator.target_opset >= 7:
        # TODO: need extra the batch size from the input tensor.
        target_shape = (1, ) + target_shape  # adding the 'batch_size' to target shape.

    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.raw_operator.name, desired_shape=target_shape)


def convert_keras_dropout(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


keras_layer_to_operator = {
    UpSampling1D: convert_keras_upsample_1d,
    UpSampling2D: convert_keras_upsample_2d,
    UpSampling3D: convert_keras_upsample_3d,
    BatchNormalization: convert_keras_batch_normalization,

    adv_activations.LeakyReLU: convert_keras_advanced_activation,
    adv_activations.ThresholdedReLU: convert_keras_advanced_activation,
    adv_activations.ELU: convert_keras_advanced_activation,
    adv_activations.PReLU: convert_keras_advanced_activation,

    Activation: convert_keras_activation,

    Conv1D: convert_keras_conv1d,
    Conv2D: convert_keras_conv2d,
    Conv3D: convert_keras_conv3d,
    Conv2DTranspose: convert_keras_conv_transpose_2d,
    Conv3DTranspose: convert_keras_conv_transpose_3d,
    DepthwiseConv2D: convert_keras_depthwise_conv_2d,

    MaxPooling1D: convert_keras_max_pooling_1d,
    MaxPooling2D: convert_keras_max_pooling_2d,
    MaxPooling3D: convert_keras_max_pooling_3d,
    AveragePooling1D: convert_keras_average_pooling_1d,
    AveragePooling2D: convert_keras_average_pooling_2d,
    AveragePooling3D: convert_keras_average_pooling_3d,

    Cropping1D: convert_keras_crop_1d,
    Cropping2D: convert_keras_crop_2d,
    Cropping3D: convert_keras_crop_3d,

    Reshape: convert_keras_reshape,

    Dropout: convert_keras_dropout,

    SimpleRNN: convert_keras_simple_rnn,
    GRU: convert_keras_gru,
    LSTM: convert_keras_lstm,
    Bidirectional: convert_bidirectional
}


@with_variable('loaded')
def static_set_ke2onnx_converters(func_set_converter):
    for ky_, val_ in six.iteritems(keras_layer_to_operator):
        func_set_converter(ky_, val_)

