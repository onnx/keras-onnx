###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six

from ..proto import keras
from ..common import with_variable
from ..common.onnx_ops import apply_identity, apply_reshape, apply_concat, apply_transpose

from .activation import convert_keras_activation
from .adv_activation import convert_keras_advanced_activation
from .batch_norm import convert_keras_batch_normalization
from .merge import convert_keras_merge_layer
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


def list_input_tensors(node):
    """
    Since Tensorflow 1.4, sometimes the node.input_tensors may not be a list, though the word is plural.
    """
    return [node.input_tensors] if hasattr(node.input_tensors, 'dtype') else node.input_tensors


def list_output_tensors(node):
    """
    Since Tensorflow 1.4, sometimes the node.output_tensors may not be a list, though the output_tensors is plural.
    """
    return [node.output_tensors] if hasattr(node.output_tensors, 'dtype') else node.output_tensors


def convert_keras_reshape(scope, operator, container):
    iop = operator.raw_operator
    target_shape = tuple([-1 if i_ is None else i_ for i_ in iop.output_shape])

    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.raw_operator.name, desired_shape=target_shape)


def convert_keras_concat(scope, operator, container):
    axis = operator.raw_operator.axis
    if axis < 0:
        axis += len(operator.raw_operator.output.shape)
    apply_concat(scope, operator.input_full_names, operator.output_full_names, container,
                 operator_name=operator.full_name, axis=axis)


def convert_keras_flatten(scope, operator, container):
    iop = operator.raw_operator
    target_shape = 1
    for idx, val in enumerate(iop.output_shape):
        if idx > 0:
            target_shape = target_shape * val
    target_shape = (-1, target_shape)
    shape_len = len(iop.input_shape)
    input_name_sector = operator.input_full_names[0].split('/')
    op_prefix = input_name_sector[0] + '/' if len(input_name_sector) > 1 else ''
    op_name = op_prefix + operator.raw_operator.name
    if iop.data_format == 'channels_last' or shape_len < 3:
        apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                      operator_name=op_name, desired_shape=target_shape)
    else:
        perm = list(range(2, shape_len))
        perm = [0] + perm + [1]
        input_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_permuted')
        apply_transpose(scope, operator.inputs[0].full_name, input_tensor_name, container,
                      operator_name=op_name+"_transpose", perm=perm)
        apply_reshape(scope, input_tensor_name, operator.outputs[0].full_name, container,
                      operator_name=op_name, desired_shape=target_shape)



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
                    ts_.name not in submodel_dict for ts_ in list_output_tensors(node_))
                if shared_layer:
                    break
            if not shared_layer:  # shared layer(model) will be processed as a whole.
                output_dict.update(submodel_dict)
                continue

        for node_ in extract_inbound_nodes(l_):
            for ts_ in list_output_tensors(node_):
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

    _layer.Add: convert_keras_merge_layer,
    _layer.Multiply: convert_keras_merge_layer,
    _layer.Subtract: convert_keras_merge_layer,
    _layer.Average: convert_keras_merge_layer,
    _layer.Maximum: convert_keras_merge_layer,
    _layer.Concatenate: convert_keras_concat,

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

    _layer.Flatten: convert_keras_flatten,
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
