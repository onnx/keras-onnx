###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import collections
from ..common.onnx_ops import apply_transpose, apply_upsample
from .common import get_permutation_config


def convert_keras_upsample(scope, operator, container, n_dims):
    op = operator.raw_operator
    # op.size type is tuple, even if we set a int in keras.layers API
    if n_dims == 1:
        scales = [1] + list(d for d in op.size)
    elif n_dims == 2 or n_dims == 3:
        # Always create the list of sampling factors in channels_first format because the input will be converted into
        # channels_first if it's in channels_last
        scales = [1, 1] + list(d for d in op.size)
    else:
        raise ValueError('Unsupported dimension %s when converting Keras Upsampling layer' % n_dims)

    mode = 'nearest'
    if hasattr(op, 'interpolation'):
        mode = 'linear' if op.interpolation.endswith('linear') else op.interpolation

    # Derive permutation configuration. If the Keras input format is not channels_first, this configuration may be used
    # to manipulate the input and output of ONNX Upsample.
    input_perm_axes, output_perm_axes = get_permutation_config(n_dims)
    channels_first = n_dims > 1 and op.data_format == 'channels_first'
    no_permutation_required = channels_first or n_dims < 2

    # Before creating the main Upsample operator, we need to permute the input tensor if the original operator is
    # working under channels_last mode.
    if no_permutation_required:
        # No permutation is required. Use input as it is.
        input_tensor_name = operator.inputs[0].full_name
    else:
        # Permute the original input and then use the permuted result as the input of ONNX Upsample
        input_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_permuted')
        apply_transpose(scope, operator.inputs[0].full_name, input_tensor_name, container, perm=input_perm_axes)

    # If no_permutation_required is True, we don't need to permute the output of ONNX Upsample. Otherwise, similar to Crop's
    # conversion, a Transpose would be added.
    if no_permutation_required:
        apply_upsample(scope, input_tensor_name, operator.outputs[0].full_name, container, mode=mode, scales=scales)
    else:
        upsampled_tensor_name = scope.get_unique_variable_name(input_tensor_name + '_upsampled')
        apply_upsample(scope, input_tensor_name, upsampled_tensor_name, container, mode=mode, scales=scales)
        apply_transpose(scope, upsampled_tensor_name, operator.outputs[0].full_name, container, perm=output_perm_axes)


def convert_keras_upsample_1d(scope, operator, container):
    convert_keras_upsample(scope, operator, container, n_dims=1)


def convert_keras_upsample_2d(scope, operator, container):
    convert_keras_upsample(scope, operator, container, n_dims=2)


def convert_keras_upsample_3d(scope, operator, container):
    convert_keras_upsample(scope, operator, container, n_dims=3)
