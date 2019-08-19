###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import keras
from ..common.onnx_ops import apply_transpose, apply_identity, OnnxOperatorBuilder
from .common import get_permutation_config


def convert_keras_crop_1(scope, operator, container, n_dims):
    op = operator.raw_operator
    op_type = 'Crop'
    attrs = {'name': operator.full_name}

    input_perm_axes, output_perm_axes = get_permutation_config(n_dims)
    channels_first = n_dims > 1 and op.data_format == 'channels_first'

    # Before creating the main Crop operator, we need to permute the input tensor if the original operator is working
    # under channels_last mode.
    if channels_first:
        input_tensor_name = operator.inputs[0].full_name
    else:
        input_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_permuted')
        apply_transpose(scope, operator.inputs[0].full_name, input_tensor_name, container, perm=input_perm_axes)

    param = op.cropping
    if isinstance(param, int):
        param = [param, param]

    if len(param) == 2:
        if isinstance(param[0], int):
            attrs['scale'] = param
        elif len(param[0]) == 2 and len(param[1]) == 2:
            # If tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
            top = param[0][0]
            bottom = param[0][1]
            left = param[1][0]
            right = param[1][1]
            attrs['border'] = [left, top, right, bottom]
        else:
            raise RuntimeError('Unknown crop parameter %s in CroppingLayer' % str(param))
    else:
        raise RuntimeError('Unknown crop parameter %s in CroppingLayer' % str(param))

    if not channels_first:
        cropped_tensor_name = scope.get_unique_variable_name(input_tensor_name + '_cropped')
        container.add_node(op_type, input_tensor_name, cropped_tensor_name, **attrs)
        apply_transpose(scope, cropped_tensor_name, operator.outputs[0].full_name, container, perm=output_perm_axes)
    else:
        container.add_node(op_type, input_tensor_name, operator.outputs[0].full_name, **attrs)


def convert_keras_crop_9(scope, operator, container, n_dims):
    op = operator.raw_operator
    channels_first = n_dims > 1 and op.data_format == 'channels_first'
    param = op.cropping

    if isinstance(param, int):
        param = [param, param]

    ori_shape = list(op.input_shape[1:])
    if isinstance(op, keras.layers.Cropping1D):
        start_border = [param[0][0], 0]
        end_border = [param[0][1], 0]
        axes_v = [0, 1, 2]
    elif isinstance(op, keras.layers.Cropping2D):
        axes_v = [0, 1, 2, 3]
        if isinstance(param[0], int): # tuple of ints
            start_border = [param]
            end_border = start_border
        else: # tuple of typle of ints
            start_border = [param[0][0], param[1][0]]
            end_border = [param[0][1], param[1][1]]
        if channels_first:
            start_border = [0] + start_border
            end_border = [0] + end_border
        else:
            start_border = start_border + [0]
            end_border = end_border + [0]
    else:
        axes_v = [0, 1, 2, 3, 4]
        if isinstance(param[0], int):  # tuple of ints
            start_border = [param]
            end_border = start_border
        else:  # tuple of typle of ints
            start_border = [param[0][0], param[1][0], param[2][0]]
            end_border = [param[0][1], param[1][1], param[2][1]]
        if channels_first:
            start_border = [0] + start_border
            end_border = [0] + end_border
        else:
            start_border = start_border + [0]
            end_border = end_border + [0]

    start_v = [0] + start_border
    end_v = [np.iinfo(np.int32).max] + list(np.array(ori_shape) - np.array(end_border))

    oopb = OnnxOperatorBuilder(container, scope)

    cropped_tensor_name = oopb.add_node('Slice' if container.target_opset >= 10 else 'DynamicSlice',
                                        [operator.inputs[0].full_name,
                                         ('_start', oopb.int64, np.array(start_v, dtype='int64')),
                                         ('_end', oopb.int64, np.array(end_v, dtype='int64')),
                                         ('_axes', oopb.int64, np.array(axes_v, dtype='int64'))
                                         ],
                                        operator.inputs[0].full_name + '_cropping')
    apply_identity(scope, cropped_tensor_name, operator.outputs[0].full_name, container)


def convert_keras_crop(scope, operator, container, n_dims):
    if container.target_opset >= 9:
        convert_keras_crop_9(scope, operator, container, n_dims)
    else:
        convert_keras_crop_1(scope, operator, container, n_dims)

def convert_keras_crop_1d(scope, operator, container):
    convert_keras_crop(scope, operator, container, n_dims=1)


def convert_keras_crop_2d(scope, operator, container):
    convert_keras_crop(scope, operator, container, n_dims=2)


def convert_keras_crop_3d(scope, operator, container):
    convert_keras_crop(scope, operator, container, n_dims=3)
