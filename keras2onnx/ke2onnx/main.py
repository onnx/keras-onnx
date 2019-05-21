###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six

from ..proto import keras
from ..common import with_variable
from ..common.onnx_ops import apply_identity, apply_reshape

from .activation import convert_keras_activation
from .adv_activation import convert_keras_advanced_activation
from .batch_norm import convert_keras_batch_normalization
from .dense import convert_keras_dense
from .dot import convert_keras_dot
from .upsample import convert_keras_upsample_1d, convert_keras_upsample_2d, convert_keras_upsample_3d
from .conv import convert_keras_conv1d, convert_keras_conv2d, convert_keras_conv3d
from .conv import convert_keras_conv_transpose_2d, convert_keras_conv_transpose_3d, convert_keras_depthwise_conv_2d
from .conv import convert_keras_separable_conv1d,convert_keras_separable_conv2d
from .pooling import convert_keras_max_pooling_1d, convert_keras_max_pooling_2d, convert_keras_max_pooling_3d
from .pooling import convert_keras_average_pooling_1d, convert_keras_average_pooling_2d, convert_keras_average_pooling_3d
from .crop import convert_keras_crop_1d, convert_keras_crop_2d, convert_keras_crop_3d
from .zeropad import convert_keras_zero_pad_1d, convert_keras_zero_pad_2d, convert_keras_zero_pad_3d
from .embedding import convert_keras_embed
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
    target_shape = tuple([-1 if i_ is None else i_ for i_ in iop.output_shape])

    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.raw_operator.name, desired_shape=target_shape)


def convert_keras_training_only_layer(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


def build_opdict_from_keras(model):
    # type: (keras.Model) -> {}

    output_dict = {}
    for l_ in model.layers:
        if hasattr(l_, 'layers'):
            submodel_dict = build_opdict_from_keras(l_)
            shared_layer = False
            for node_ in extract_inbound_nodes(l_):
                shared_layer |= any(
                    ts_.name not in submodel_dict for ts_ in node_.output_tensors)
                if shared_layer:
                    break
            if not shared_layer:  # shared layer(model) will be processed as a whole.
                output_dict.update(submodel_dict)
                continue

        for node_ in extract_inbound_nodes(l_):
            for ts_ in node_.output_tensors:
                output_dict[ts_.name] = (l_, model)

    return output_dict


_layer = keras.layers
_adv_activations = keras.layers.advanced_activations

keras_layer_to_operator = {
    _layer.UpSampling1D: convert_keras_upsample_1d,
    _layer.UpSampling2D: convert_keras_upsample_2d,
    _layer.UpSampling3D: convert_keras_upsample_3d,
    _layer.BatchNormalization: convert_keras_batch_normalization,

    _adv_activations.LeakyReLU: convert_keras_advanced_activation,
    _adv_activations.ThresholdedReLU: convert_keras_advanced_activation,
    _adv_activations.ELU: convert_keras_advanced_activation,
    _adv_activations.PReLU: convert_keras_advanced_activation,

    _layer.Activation: convert_keras_activation,

    _layer.Conv1D: convert_keras_conv1d,
    _layer.Conv2D: convert_keras_conv2d,
    _layer.Conv3D: convert_keras_conv3d,
    _layer.Conv2DTranspose: convert_keras_conv_transpose_2d,
    _layer.Conv3DTranspose: convert_keras_conv_transpose_3d,
    _layer.DepthwiseConv2D: convert_keras_depthwise_conv_2d,
    _layer.SeparableConv1D: convert_keras_separable_conv1d,
    _layer.SeparableConv2D: convert_keras_separable_conv2d,

    _layer.Dense: convert_keras_dense,
    _layer.Dot: convert_keras_dot,
    _layer.Embedding: convert_keras_embed,

    _layer.MaxPooling1D: convert_keras_max_pooling_1d,
    _layer.MaxPooling2D: convert_keras_max_pooling_2d,
    _layer.MaxPooling3D: convert_keras_max_pooling_3d,
    _layer.AveragePooling1D: convert_keras_average_pooling_1d,
    _layer.AveragePooling2D: convert_keras_average_pooling_2d,
    _layer.AveragePooling3D: convert_keras_average_pooling_3d,

    _layer.Cropping1D: convert_keras_crop_1d,
    _layer.Cropping2D: convert_keras_crop_2d,
    _layer.Cropping3D: convert_keras_crop_3d,

    _layer.ZeroPadding1D: convert_keras_zero_pad_1d,
    _layer.ZeroPadding2D: convert_keras_zero_pad_2d,
    _layer.ZeroPadding3D: convert_keras_zero_pad_3d,

    _layer.Reshape: convert_keras_reshape,

    _layer.Dropout: convert_keras_training_only_layer,

    _layer.SimpleRNN: convert_keras_simple_rnn,
    _layer.GRU: convert_keras_gru,
    _layer.LSTM: convert_keras_lstm,
    _layer.Bidirectional: convert_bidirectional
}


@with_variable('loaded')
def static_set_ke2onnx_converters(func_set_converter):
    for ky_, val_ in six.iteritems(keras_layer_to_operator):
        func_set_converter(ky_, val_)

