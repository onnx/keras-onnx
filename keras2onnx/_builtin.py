###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import sys
import numbers
import tensorflow
import numpy as np

from keras2onnx._consts import TYPES, NCHW_TO_NHWC, NHWC_TO_NCHW, HWCN_TO_NCHW, \
    NCDHW_TO_NDHWC, NDHWC_TO_NCDHW, DHWCN_TO_NCDHW
from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from .common.utils import count_dynamic_dim
from .common.onnx_ops import apply_identity, apply_reshape, OnnxOperatorBuilder
from .funcbook import converter_func, set_converters
from .proto import keras
from .proto.tfcompat import is_tf2
from ._tf_utils import (is_nhwc as _is_nhwc,
                        tf_attrs_to_onnx as _to_onnx_attrs,
                        cal_tensor_value as _cal_tensor_value,
                        cal_tensor_shape as _cal_tensor_shape,
                        to_onnx_type as _to_onnx_type)


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name,
                   operator.outputs[0].full_name, container, operator_name=operator.full_name)


@converter_func(TYPES.Identity)
def convert_tf_identity(scope, operator, container):
    default_convert(scope, operator, container)


@converter_func(TYPES.AddN)
def convert_tf_addn(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output("apply_sum",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name + '_sum')


def _convert_tf_argmax_argmin_helper(scope, operator, container, arg_str):
    node = operator.raw_operator
    axis = _cal_tensor_value(node.inputs[1]).item(0)
    dtype = _to_onnx_type(node.outputs[0].dtype)
    oopb = OnnxOperatorBuilder(container, scope)
    arg_func = oopb.apply_argmax if arg_str == 'argmax' else oopb.apply_argmin
    if dtype == oopb.int64:
        oopb.apply_op_with_output("apply_" + arg_str,
                                  operator.input_full_names[0],
                                  operator.output_full_names,
                                  name=operator.full_name + '_' + arg_str,
                                  axis=axis,
                                  keepdims=0)
    else:
        arg_output = arg_func(operator.input_full_names[0],
                              name=operator.full_name + '_' + arg_str,
                              axis=axis,
                              keepdims=0)
        oopb.apply_op_with_output("apply_cast",
                                  arg_output,
                                  operator.output_full_names,
                                  name=operator.full_name + '_cast',
                                  to=dtype)


@converter_func(TYPES.ArgMax)
def convert_tf_argmax(scope, operator, container):
    _convert_tf_argmax_argmin_helper(scope, operator, container, 'argmax')


@converter_func(TYPES.ArgMin)
def convert_tf_argmin(scope, operator, container):
    _convert_tf_argmax_argmin_helper(scope, operator, container, 'argmin')


def _spatial_map(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=False, new_kernel_shape=None,
                         output_indices=None, input_perm=NHWC_TO_NCHW, kernel_perm=HWCN_TO_NCHW,
                         output_perm=NCHW_TO_NHWC, op_type='Conv', is_nhwc=False):
    if output_indices is None:
        output_indices = [0]

    if is_nhwc:
        # transpose input if needed, no need to record shapes on input
        transpose_node_1 = oopb.apply_transpose(node.inputs[0].name,
                                                name=operator.full_name + '_transpose_1',
                                                perm=input_perm)
    else:
        transpose_node_1 = [node.inputs[0].name]

    if op_type == 'Conv':
        # kernel must to be transposed
        if with_kernel:
            val = _cal_tensor_value(node.inputs[1])
            if val is not None:
                val = val.transpose(kernel_perm)
                onnx_type = _to_onnx_type(node.inputs[1].dtype)
                transpose_node_kernel = oopb.apply_identity([('_start', onnx_type, val)],
                                                            name=operator.full_name + '_transpose_kernel')
            else:
                transpose_node_kernel = oopb.apply_transpose(node.inputs[1].name,
                                                             name=operator.full_name + '_transpose_kernel',
                                                             perm=kernel_perm)
            # TODO, some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        else:
            transpose_node_kernel = [node.inputs[1].name]

        conv_node = oopb.apply_conv(transpose_node_1 + transpose_node_kernel,
                                    name=operator.full_name + '_conv',
                                    **attrs)
    else:
        conv_node = oopb.add_node(op_type,
                                  transpose_node_1,
                                  name=operator.full_name + '_conv',
                                  **attrs)

    # transpose outputs if needed
    if is_nhwc:
        for idx in output_indices:
            oopb.add_node_with_output("Transpose",
                                      conv_node,
                                      operator.outputs[idx].full_name,
                                      name=operator.full_name + '_transpose_2_' + str(idx),
                                      perm=output_perm)
    else:
        for idx in output_indices:
            oopb.apply_op_with_output("apply_identity",
                                      conv_node,
                                      operator.outputs[idx].full_name,
                                      name=operator.full_name + '_identity_' + str(idx))


def _conv_dims_attr(node, dims):
    if _is_nhwc(node):
        if len(dims) == 2:
            return dims
        else:
            return dims[1:-1]
    else:
        return dims[2:]


def _add_padding(node, padding, dilations, spatial, pad_perm, strides, kernel_shape, is_nhwc):
    attrs_pad = {}
    if padding:
        if dilations is None:
            dilations = [1] * spatial * 2
        if padding == b'SAME':
            pads = [0] * spatial * 2
            input_shape = _cal_tensor_shape(node.inputs[0])
            output_shape = _cal_tensor_shape(node.outputs[0])
            # transpose shape to nchw
            if is_nhwc:
                input_shape = _spatial_map(input_shape, pad_perm)
                output_shape = _spatial_map(output_shape, pad_perm)
            # calculate pads
            if any(input_shape[i + 2] is None or output_shape[i + 2] is None for i in range(spatial)):
                attrs_pad["auto_pad"] = "SAME_UPPER"
            else:
                for i in range(spatial):
                    pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * kernel_shape[i] - input_shape[i + 2]
                    pad = max(pad, 0)
                    pads[i] = pad // 2
                    pads[i + spatial] = pad - pad // 2
                attrs_pad["pads"] = pads
    return attrs_pad


def _convert_tf_pool(scope, operator, container, arg_str):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    spatial = len(node.inputs[0].shape) - 2
    pad_perm = NHWC_TO_NCHW if spatial < 3 else NDHWC_TO_NCDHW
    is_nhwc = arg_str == 'MaxPoolWithArgmax' or _is_nhwc(node)
    if len(node.inputs) < 3:
        kernel_shape_tf = node.get_attr('ksize')
        strides_tf = node.get_attr('strides')
    else:
        kernel_shape_tf = _cal_tensor_value(node.inputs[1])
        strides_tf = _cal_tensor_value(node.inputs[2])

    if is_nhwc:
        kernel_shape_hw = kernel_shape_tf[1:-1]
        strides_hw = strides_tf[1:-1]
    else:
        kernel_shape_hw = kernel_shape_tf[2:]
        strides_hw = strides_tf[2:]

    dilations = None
    attrs = {"kernel_shape": kernel_shape_hw, "strides": strides_hw}
    padding = node.get_attr('padding')
    attrs_pads = _add_padding(node, padding, dilations, spatial, pad_perm, strides_hw, kernel_shape_hw, is_nhwc)
    attrs.update(attrs_pads)
    output_indices = None
    op_str = arg_str
    if arg_str == 'MaxPoolWithArgmax':
        output_indices = [0, 1]
        op_str = 'MaxPool'
    if spatial < 3:
        _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=False, output_indices=output_indices,
                             input_perm=NHWC_TO_NCHW,
                             kernel_perm=HWCN_TO_NCHW, output_perm=NCHW_TO_NHWC, op_type=op_str,
                             is_nhwc=is_nhwc)
    else:
        _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=False, output_indices=output_indices,
                             input_perm=NDHWC_TO_NCDHW,
                             kernel_perm=DHWCN_TO_NCDHW, output_perm=NCDHW_TO_NDHWC, op_type=op_str,
                             is_nhwc=is_nhwc)


@converter_func(TYPES.AvgPool)
def convert_tf_avgpool(scope, operator, container):
    _convert_tf_pool(scope, operator, container, 'AveragePool')


@converter_func(TYPES.AvgPool3D)
def convert_tf_avgpool3d(scope, operator, container):
    _convert_tf_pool(scope, operator, container, 'AveragePool')


@converter_func(TYPES.MaxPool)
def convert_tf_maxpool(scope, operator, container):
    _convert_tf_pool(scope, operator, container, 'MaxPool')


@converter_func(TYPES.MaxPoolWithArgmax)
def convert_tf_maxpool_argmax(scope, operator, container):
    _convert_tf_pool(scope, operator, container, 'MaxPoolWithArgmax')


@converter_func(TYPES.MaxPoolV2)
def convert_tf_maxpoolv2(scope, operator, container):
    _convert_tf_pool(scope, operator, container, 'MaxPool')


@converter_func(TYPES.MaxPool3D)
def convert_tf_maxpool3d(scope, operator, container):
    _convert_tf_pool(scope, operator, container, 'MaxPool')


@converter_func(TYPES.BatchToSpaceND)
def convert_tf_batch_to_space(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    blocksize = _cal_tensor_value(node.inputs[1])
    crops = _cal_tensor_value(node.inputs[2])
    if operator.target_opset <= 10 or (blocksize is not None and crops is not None):
        input_shape = _cal_tensor_shape(node.outputs[0])
        assert len(input_shape) in (4, 3)
        assert len(blocksize) == 2 and blocksize[0] == blocksize[1]

        if len(input_shape) == 3:
            unsqueeze_node_1 = oopb.apply_unsqueeze(operator.inputs[0].full_name,
                                                    name=operator.full_name + '_unsqueeze_0',
                                                    axes=[3])
            transpose_node_1 = oopb.apply_transpose(unsqueeze_node_1,
                                                    name=operator.full_name + '_transpose_1',
                                                    perm=[3, 0, 1, 2])
        else:
            transpose_node_1 = oopb.apply_transpose(operator.inputs[0].full_name,
                                                    name=operator.full_name + '_transpose_1',
                                                    perm=[3, 0, 1, 2])
        depth_to_space_node = oopb.add_node('DepthToSpace',
                                            transpose_node_1,
                                            operator.inputs[0].full_name + '_depth_to_space',
                                            blocksize=blocksize[0])
        transpose_node_2 = oopb.apply_transpose(depth_to_space_node,
                                                name=operator.full_name + '_transpose_2',
                                                perm=[1, 2, 3, 0])

        if np.count_nonzero(crops) == 0:
            oopb.apply_op_with_output("apply_identity",
                                      transpose_node_2,
                                      operator.output_full_names,
                                      name=operator.full_name + '_slice')
            return

        slice_axis = [1, 2]
        top, bottom = crops[0]
        left, right = crops[1]
        starts = [top, left]
        ends = []
        for end in [bottom, right]:
            if end != 0:
                ends.append(-end)
            else:
                ends.append(np.iinfo(np.int32).max)

        if len(input_shape) == 3:
            slice_node = oopb.apply_slice(transpose_node_2,
                                          name=operator.full_name + '_slice',
                                          starts=starts, ends=ends, axes=slice_axis)
            oopb.apply_op_with_output("apply_squeeze",
                                      slice_node,
                                      operator.output_full_names,
                                      name=operator.full_name + '_squeeze_output',
                                      axes=[3])
        else:
            oopb.apply_op_with_output("apply_slice",
                                      transpose_node_2,
                                      operator.output_full_names,
                                      name=operator.full_name + '_slice',
                                      starts=starts, ends=ends, axes=slice_axis)

    else:
        shape_x = oopb.add_node('Shape', [operator.inputs[0].full_name],
                                operator.full_name + '_input_0_shape')
        block_shape = oopb.apply_cast(operator.inputs[1].full_name,
                                      to=oopb.int64,
                                      name=operator.full_name + '_input_1_cast')
        crop = oopb.apply_cast(operator.inputs[2].full_name,
                               to=oopb.int64,
                               name=operator.full_name + '_input_2_cast')
        block_size = oopb.apply_slice(block_shape,
                                      name=operator.full_name + '_slice_0',
                                      starts=[0], ends=[1])
        block_prod = oopb.apply_mul(block_size + block_size,
                                    name=operator.full_name + '_mul_0')
        padded_block_prod = oopb.apply_pad(block_prod,
                                           name=operator.full_name + '_pad_0',
                                           pads=[0, 3],
                                           value=1)
        new_shape_x = oopb.apply_div([shape_x] + padded_block_prod,
                                     name=operator.full_name + '_div')
        concat_new_shape_x = oopb.apply_concat(block_shape + new_shape_x,
                                               name=operator.full_name + '_concat',
                                               axis=0)
        reshaped_x = oopb.apply_reshape([operator.inputs[0].full_name],
                                        name=operator.full_name + '_reshape_0',
                                        desired_shape=concat_new_shape_x[0])
        transposed_x = oopb.apply_transpose(reshaped_x,
                                            name=operator.full_name + '_transpose_0',
                                            perm=[2, 3, 0, 4, 1, 5])
        padded_block_shape = oopb.apply_pad(block_shape,
                                            name=operator.full_name + '_pad_1',
                                            pads=[1, 1],
                                            value=1)
        new_shape_x_v2 = oopb.apply_mul(new_shape_x + padded_block_shape,
                                        name=operator.full_name + '_mul_1')
        reshaped_x_v2 = oopb.apply_reshape(transposed_x,
                                           name=operator.full_name + '_reshape_1',
                                           desired_shape=new_shape_x_v2[0])
        transposed_crop = oopb.apply_transpose(crop,
                                               name=operator.full_name + '_transpose_1',
                                               perm=[1, 0])
        slice_crop_starts = oopb.apply_slice(transposed_crop,
                                             name=operator.full_name + '_slice_starts',
                                             starts=[0, 0], ends=[1, 2])
        reshaped_slice_crop_starts = oopb.apply_reshape(slice_crop_starts,
                                                        name=operator.full_name + '_reshape_starts',
                                                        desired_shape=[2])
        slice_crop_ends = oopb.apply_slice(transposed_crop,
                                           name=operator.full_name + '_slice_ends',
                                           starts=[1, 0], ends=[2, 2])
        reshaped_slice_crop_ends = oopb.apply_reshape(slice_crop_ends,
                                                      name=operator.full_name + '_reshape_ends',
                                                      desired_shape=[2])
        sliced_new_shape_x_v2 = oopb.apply_slice(new_shape_x_v2,
                                                 name=operator.full_name + '_slice_3',
                                                 starts=[1], ends=[3])
        neged_reshaped_slice_crop_ends = oopb.apply_sub(sliced_new_shape_x_v2 + reshaped_slice_crop_ends,
                                                        name=operator.full_name + '_sub')
        oopb.apply_op_with_output("apply_slice",
                                  reshaped_x_v2,
                                  operator.output_full_names,
                                  name=operator.full_name + '_slice_final',
                                  starts=reshaped_slice_crop_starts[0],
                                  ends=neged_reshaped_slice_crop_ends[0],
                                  axes=[1, 2])


@converter_func(TYPES.SpaceToBatchND)
def convert_tf_space_to_batch(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    blocksize = _cal_tensor_value(node.inputs[1])
    paddings = _cal_tensor_value(node.inputs[2])
    if operator.target_opset <= 10 or (blocksize is not None and paddings is not None):
        input_shape = _cal_tensor_shape(node.outputs[0])
        assert len(input_shape) == 4
        assert len(blocksize) == 2 and blocksize[0] == blocksize[1]

        top, bottom = paddings[0]
        left, right = paddings[1]
        pads = [0, top, left, 0,
                0, bottom, right, 0]

        if np.count_nonzero(pads) > 0:
            pad_op = oopb.apply_pad(operator.inputs[0].full_name,
                                    name=operator.full_name + '_pad_1',
                                    pads=pads)
        else:
            pad_op = operator.inputs[0].full_name

        transpose_node_1 = oopb.apply_transpose(pad_op,
                                                name=operator.full_name + '_transpose_1',
                                                perm=[3, 0, 1, 2])
        space_to_depth_node = oopb.add_node('SpaceToDepth',
                                            transpose_node_1,
                                            operator.inputs[0].full_name + '_space_to_depth',
                                            blocksize=blocksize[0])
        oopb.apply_op_with_output("apply_transpose",
                                  space_to_depth_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_transpose_2',
                                  perm=[1, 2, 3, 0])
    else:
        shape_x = oopb.add_node('Shape', [operator.inputs[0].full_name],
                                operator.full_name + '_input_0_shape')
        block_shape = oopb.apply_cast(operator.inputs[1].full_name,
                                      to=oopb.int64,
                                      name=operator.full_name + '_input_1_cast')
        pad_x = oopb.apply_cast(operator.inputs[2].full_name,
                                to=oopb.int64,
                                name=operator.full_name + '_input_2_cast')
        concated_pad_x = oopb.apply_concat(
            [('_const_zero_zero', oopb.int64, np.array([[0, 0]], dtype='int64'))] + pad_x,
            name=operator.full_name + '_concat_1',
            axis=0)
        concated_pad_x_v2 = oopb.apply_concat(
            concated_pad_x + [('_const_zero_zero', oopb.int64, np.array([[0, 0]], dtype='int64'))],
            name=operator.full_name + '_concat_2',
            axis=0)
        transposed_concated_pad_x_v2 = oopb.apply_transpose(concated_pad_x_v2,
                                                            name=operator.full_name + '_transpose_0',
                                                            perm=[1, 0])
        reshaped_transposed_pad_x = oopb.apply_reshape(transposed_concated_pad_x_v2,
                                                       name=operator.full_name + '_reshape_0',
                                                       desired_shape=[8])
        padded_input_x = oopb.apply_pad(operator.inputs[0].full_name,
                                        name=operator.full_name + '_pad_1',
                                        pads=reshaped_transposed_pad_x)
        padded_block_shape = oopb.apply_pad(block_shape,
                                            name=operator.full_name + '_pad_2',
                                            pads=[1, 1], value=1)
        new_shape_x = oopb.apply_div([shape_x] + padded_block_shape,
                                     name=operator.full_name + '_div')
        first_row_new_shape_x = oopb.apply_slice(new_shape_x,
                                                 name=operator.full_name + '_slice_0',
                                                 starts=[0], ends=[2])
        block_size = oopb.apply_slice(block_shape,
                                      name=operator.full_name + '_slice_1',
                                      starts=[0], ends=[1])
        new_first_row_new_shape_x = oopb.apply_concat(first_row_new_shape_x + block_size,
                                                      name=operator.full_name + '_concat_3',
                                                      axis=0)
        second_row_new_shape_x_first_half = oopb.apply_slice(new_shape_x,
                                                             name=operator.full_name + '_slice_second_first',
                                                             starts=[2], ends=[3])
        second_row_new_shape_x_second_half = oopb.apply_slice(new_shape_x,
                                                              name=operator.full_name + '_slice_second_second',
                                                              starts=[3], ends=[4])
        new_second_row_new_shape_x_first_half = oopb.apply_concat(second_row_new_shape_x_first_half + block_size,
                                                                  name=operator.full_name + '_concat_second_first',
                                                                  axis=0)
        new_second_row_new_shape_x = oopb.apply_concat(
            new_second_row_new_shape_x_first_half + second_row_new_shape_x_second_half,
            name=operator.full_name + '_concat_second_shape',
            axis=0)
        new_shape_x_v2 = oopb.apply_concat(new_first_row_new_shape_x + new_second_row_new_shape_x,
                                           name=operator.full_name + '_concat_shape',
                                           axis=0)
        new_x = oopb.apply_reshape(padded_input_x[0],
                                   name=operator.full_name + '_reshape_new_x',
                                   desired_shape=new_shape_x_v2[0])
        transposed_new_x = oopb.apply_transpose(new_x,
                                                name=operator.full_name + '_transpose_new',
                                                perm=[2, 4, 0, 1, 3, 5])
        block_size_prod = oopb.apply_mul(block_size + block_size,
                                         name=operator.full_name + '_mul_0')
        padded_block_size_prod = oopb.apply_pad(block_size_prod,
                                                name=operator.full_name + '_pad_block_size',
                                                pads=[0, 3], value=1)
        new_shape_x_v3 = oopb.apply_mul(new_shape_x + padded_block_size_prod,
                                        name=operator.full_name + '_mul_shape_v3')
        oopb.apply_op_with_output("apply_reshape",
                                  transposed_new_x,
                                  operator.output_full_names,
                                  name=operator.full_name + '_transpose_2',
                                  desired_shape=new_shape_x_v3)


@converter_func(TYPES.BiasAdd, TYPES.BiasAddV1)
def convert_tf_bias_add(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    if not _is_nhwc(node):
        shape0 = _cal_tensor_shape(node.inputs[0])
        shape1 = _cal_tensor_shape(node.inputs[1])
        if node.inputs[1].op.type == 'Const':
            new_broadcast_shape = [shape1[0]] + [1] * (len(shape0) - 2)
            reshape_node = oopb.apply_reshape(operator.inputs[1].full_name,
                                              name=operator.full_name + '_reshape',
                                              desired_shape=new_broadcast_shape)
            oopb.apply_op_with_output("apply_add",
                                      [node.inputs[0].name, reshape_node[0]],
                                      operator.output_full_names,
                                      name=operator.full_name + '_add')
            return

    oopb.apply_op_with_output("apply_add",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name + '_add')


@converter_func(TYPES.Cumsum)
def convert_tf_cum_sum(scope, operator, container):
    if operator.target_opset < 11:
        raise ValueError("CumSum op is not supported for opset < 11")
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    attrs = {'exclusive': node.get_attr('exclusive'), 'reverse': node.get_attr('reverse')}
    oopb.add_node_with_output('CumSum',
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              **attrs)


def _calc_explicit_padding(input_size, output_shape, output_padding, kernel_shape, stride, dilation,
                           perm):
    def to_nchw(x, perm):
        return [x[perm[n_]] for n_ in range(len(x))]
    input_size = to_nchw(input_size, perm)[2:]

    spatial = len(kernel_shape)
    total_padding = []
    pads = [None] * 2 * spatial
    for i in range(spatial):
        total_padding[i:] = [stride[i] * ((input_size[i] - 1) // stride[i]) + 1 +
                             output_padding[i] + (kernel_shape[i] - 1) * dilation[i] - input_size[i]]
        total_padding[i] = max(total_padding[i], 0)
        pads[i] = total_padding[i] // 2
        pads[i + spatial] = total_padding[i] - (total_padding[i] // 2)

    return pads


@converter_func(TYPES.DepthToSpace)
def convert_tf_depth_to_space(scope, operator, container):
    node = operator.raw_operator
    block_size = node.get_attr('block_size')
    oopb = OnnxOperatorBuilder(container, scope)
    if _is_nhwc(node):
        adjusted_input_name = oopb.apply_transpose(operator.input_full_names,
                                                   name=operator.full_name + '_pre_transpose',
                                                   perm=[0, 3, 1, 2])
        depth_to_space_result = oopb.add_node("DepthToSpace",
                                              adjusted_input_name,
                                              name=operator.full_name,
                                              blocksize=node.get_attr('block_size'),
                                              mode="DCR",
                                              op_version=11)
        oopb.apply_op_with_output("apply_transpose",
                                  depth_to_space_result,
                                  operator.output_full_names,
                                  name=operator.full_name + '_post_transpose',
                                  perm=[0, 2, 3, 1])
    else:
        oopb.add_node_with_output("DepthToSpace",
                                  operator.input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name,
                                  blocksize=block_size,
                                  mode="DCR",
                                  op_version=11)


@converter_func(TYPES.DepthwiseConv2dNative)
def convert_tf_depthwise_conv2d(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)

    channels_first = node.get_attr('data_format') == b'NCHW'

    if channels_first:
        adjusted_input_name = [operator.inputs[0].full_name]
    else:
        adjusted_input_name = oopb.apply_transpose(operator.inputs[0].full_name,
                                                   name=operator.full_name + '_transpose_0',
                                                   perm=[0, 3, 1, 2])

    weight_perm_axes = [3, 2, 0, 1]
    weight_shape = _cal_tensor_shape(node.inputs[1])
    new_shape = weight_shape[:2] + [1, weight_shape[2] * weight_shape[3]]
    weight_reshape = oopb.apply_reshape(operator.inputs[1].full_name,
                                        name=operator.full_name + '_reshape_ends',
                                        desired_shape=new_shape)
    transposed_weight = oopb.apply_transpose(weight_reshape,
                                             name=operator.full_name + '_transpose_new',
                                             perm=weight_perm_axes)

    attrs = {}
    dilation_rate = node.get_attr('dilations')
    dilation_rate = dilation_rate[2:] if channels_first else dilation_rate[1:3]
    attrs['dilations'] = dilation_rate
    strides = node.get_attr('strides')
    strides = strides[2:] if channels_first else strides[1:3]
    attrs['strides'] = strides
    kernel_size = weight_shape[:2]
    input_channels, output_channels = weight_shape[-2:]
    group = input_channels
    attrs['group'] = group

    input_shape = _cal_tensor_shape(node.inputs[0])
    output_shape = _cal_tensor_shape(node.outputs[0])

    if node.get_attr('padding') == b'VALID':
        attrs['auto_pad'] = 'VALID'
    elif node.get_attr('padding') == b'SAME':
        if count_dynamic_dim(input_shape) > 1:
            attrs['auto_pad'] = 'SAME_UPPER'
        else:
            attrs['auto_pad'] = 'NOTSET'
            output_padding = [0] * len(kernel_size)
            attrs['pads'] = _calc_explicit_padding(input_shape,
                                                   output_shape,
                                                   output_padding,
                                                   kernel_size,
                                                   strides,
                                                   dilation_rate,
                                                   list(range(
                                                       len(input_shape))) if channels_first else [0, 2, 3, 1])

    intermediate_output_name = oopb.apply_conv(adjusted_input_name + transposed_weight,
                                               name=operator.full_name + '_conv',
                                               **attrs)

    if not channels_first:
        oopb.apply_op_with_output("apply_transpose",
                                  intermediate_output_name,
                                  operator.output_full_names,
                                  name=operator.full_name + '_transpose_final',
                                  perm=[0, 2, 3, 1])
    else:
        oopb.apply_op_with_output("apply_identity",
                                  intermediate_output_name,
                                  operator.output_full_names,
                                  name=operator.full_name + '_identity_final')


@converter_func(TYPES.MatMul, TYPES.BatchMatMul, TYPES.BatchMatMulV2)
def convert_tf_batchmatmul(scope, operator, container):
    node = operator.raw_operator  # type: tensorflow.Operation
    oopb = OnnxOperatorBuilder(container, scope)

    tranpose_a = node.get_attr('transpose_a') if node.type == TYPES.MatMul else node.get_attr('adj_x')
    tranpose_b = node.get_attr('transpose_b') if node.type == TYPES.MatMul else node.get_attr('adj_y')

    input_names = operator.input_full_names
    for idx_, flag in enumerate([tranpose_a, tranpose_b]):
        if flag:
            shape_len = len(node.inputs[idx_].shape)
            perm = list(range(0, shape_len))[:-2] + [shape_len - 1, shape_len - 2]
            input_names[idx_] = oopb.apply_transpose(input_names[idx_],
                                                     name=operator.full_name + '_transpose_%d' % idx_,
                                                     perm=perm)[0]

    oopb.apply_op_with_output("apply_matmul",
                              input_names,
                              operator.output_full_names,
                              name=operator.full_name + '_add')


@converter_func(TYPES.SquaredDifference)
def convert_tf_squared_difference(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    sub_node = oopb.apply_sub(operator.input_full_names, name=operator.full_name + '_sub')
    oopb.apply_op_with_output('apply_mul', sub_node + sub_node, operator.output_full_names, name=operator.full_name)


@converter_func(TYPES.ConcatV2)
def convert_tf_concat_v2(scope, operator, container):
    node = operator.raw_operator
    input_name_idx = []
    original_input_number = len(operator.input_full_names) - 1
    for idx in range(original_input_number):
        val = _cal_tensor_value(node.inputs[idx])
        if not (val is not None and len(val) == 0):
            input_name_idx.append(idx)

    input_full_names = [operator.input_full_names[idx] for idx in input_name_idx]

    axis_val = _cal_tensor_value(node.inputs[-1]).item(0)
    if axis_val < 0 and operator.target_opset < 11:
        input_shape = _cal_tensor_shape(node.inputs[0])
        axis_val = len(input_shape) + axis_val

    oopb = OnnxOperatorBuilder(container, scope)
    need_casting = False
    if operator.target_opset < 8:
        supported_types = [oopb.float, oopb.float16]
        dtype = _to_onnx_type(node.outputs[0].dtype)
        need_casting = dtype not in supported_types

    if need_casting:
        concat_node = oopb.apply_concat(input_full_names,
                                        name=operator.full_name + '_concat',
                                        axis=axis_val)
        oopb.apply_op_with_output("apply_cast",
                                  concat_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_cast',
                                  to=oopb.float)
    else:
        oopb.apply_op_with_output("apply_concat",
                                  input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name + '_concat',
                                  axis=axis_val)


@converter_func(TYPES.Const)
def convert_tf_const(scope, operator, container):
    node = operator.raw_operator
    np_arr = _cal_tensor_value(node.outputs[0])
    onnx_tensor = numpy_helper.from_array(np_arr, operator.outputs[0].onnx_name)
    container.add_initializer_from_tensor(onnx_tensor)


def _convert_tf_conv(scope, operator, container, spatial, pad_perm):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    kernel_shape = _cal_tensor_shape(node.inputs[1])[0:-2]
    strides = _conv_dims_attr(node, node.get_attr('strides'))
    dilations = _conv_dims_attr(node, node.get_attr('dilations'))
    padding = node.get_attr('padding')
    attrs = {'strides': strides, 'dilations': dilations, 'kernel_shape': kernel_shape}
    is_nhwc = _is_nhwc(node)
    attrs_pad = _add_padding(node, padding, dilations, spatial, pad_perm, strides, kernel_shape, is_nhwc)
    attrs.update(attrs_pad)
    is_nhwc = _is_nhwc(node)

    if spatial < 3:
        _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=True, input_perm=NHWC_TO_NCHW,
                             kernel_perm=HWCN_TO_NCHW, output_perm=NCHW_TO_NHWC, is_nhwc=is_nhwc)
    else:
        _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=True, input_perm=NDHWC_TO_NCDHW,
                             kernel_perm=DHWCN_TO_NCDHW, output_perm=NCDHW_TO_NDHWC, is_nhwc=is_nhwc)


@converter_func(TYPES.Conv1D)
def convert_tf_conv1d(scope, operator, container):
    _convert_tf_conv(scope, operator, container, 2, NHWC_TO_NCHW)


@converter_func(TYPES.Conv2D)
def convert_tf_conv2d(scope, operator, container):
    _convert_tf_conv(scope, operator, container, 2, NHWC_TO_NCHW)


@converter_func(TYPES.Conv3D)
def convert_tf_conv3d(scope, operator, container):
    _convert_tf_conv(scope, operator, container, 3, NDHWC_TO_NCDHW)


@converter_func(TYPES.Einsum)
def convert_tf_einsum(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    equation_str = node.get_attr('equation').decode("utf-8")
    oopb.add_node_with_output("Einsum",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              equation=equation_str,
                              op_version=12)


@converter_func(TYPES.ExpandDims)
def convert_tf_expand_dims(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = _cal_tensor_value(node.inputs[1]).tolist()
    rank = len(_cal_tensor_shape(node.inputs[0]))
    oopb.apply_op_with_output("apply_unsqueeze",
                              [operator.inputs[0].full_name],
                              operator.output_full_names,
                              name=operator.full_name,
                              axes=[axis],
                              rank=rank)


def _convert_tf_fused_batch_norm_core(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    input_dim = len(_cal_tensor_shape(node.inputs[0]))
    epsilon = node.get_attr('epsilon')
    attrs = {'epsilon': epsilon, 'momentum': 0.9, 'spatial': 1}
    outputs_num = min(5, len(node.outputs))

    if _is_nhwc(node):
        input_perm = [0, input_dim - 1] + list(range(1, input_dim - 1))
        transpose_node_1 = oopb.apply_transpose(operator.inputs[0].full_name, name=operator.full_name + '_transpose_1',
                                                perm=input_perm)
        for idx in range(1, 5):
            transpose_node_1.append(operator.inputs[idx].full_name)
        batch_norm = oopb.apply_batch_norm(transpose_node_1, name=operator.full_name + '_batch_norm',
                                           outputs_num=outputs_num, **attrs)
        output_perm = [0] + list(range(2, input_dim)) + [1]
        final_node = oopb.apply_transpose(batch_norm[0], name=operator.full_name + '_transpose_2',
                                          perm=output_perm)
    else:
        transpose_node_1 = []
        for idx in range(5):
            transpose_node_1.append(operator.inputs[idx].full_name)
        batch_norm = oopb.apply_batch_norm(transpose_node_1, name=operator.full_name + '_batch_norm',
                                           outputs_num=outputs_num, **attrs)
        final_node = batch_norm[0]

    oopb.apply_op_with_output("apply_identity",
                              final_node,
                              operator.outputs[0].full_name,
                              name=operator.full_name)


@converter_func(TYPES.Fill)
def convert_tf_fill(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    if operator.target_opset < 9:
        fill_shape = _cal_tensor_shape(node.inputs[0])
        fill_shape_dims = fill_shape[0]
        val_dtype = _to_onnx_type(node.inputs[1].dtype)
        need_cast = val_dtype != oopb.float and operator.target_opset < 9
        if need_cast:
            cast_input_val = oopb.apply_cast(operator.inputs[1].full_name,
                                             to=oopb.float,
                                             name=operator.full_name + '_input_value_cast')
        else:
            cast_input_val = [operator.inputs[1].full_name]
        idx = 0
        for _ in range(fill_shape_dims):
            cast_input_val = oopb.apply_unsqueeze(cast_input_val,
                                                  name=operator.full_name + '_unsqueeze_' + str(idx),
                                                  axes=[0])
            idx += 1
        cast_input_dim = oopb.apply_cast(operator.inputs[0].full_name,
                                         to=oopb.int64,
                                         name=operator.full_name + '_input_dim_cast')
        if need_cast:
            tile_node = oopb.apply_tile(cast_input_val + cast_input_dim,
                                        name=operator.full_name + '_tile')
            oopb.apply_op_with_output("apply_cast",
                                      tile_node,
                                      operator.output_full_names,
                                      name=operator.full_name)
        else:
            oopb.apply_op_with_output("apply_tile",
                                      cast_input_val,
                                      operator.output_full_names,
                                      name=operator.full_name,
                                      repeats=cast_input_dim[0])
    else:
        val_dtype = _to_onnx_type(node.inputs[0].dtype)
        if val_dtype != oopb.int64:
            cast_input_dim = oopb.apply_cast(operator.inputs[0].full_name,
                                             to=oopb.int64,
                                             name=operator.full_name + '_input_dim_cast')
        else:
            cast_input_dim = [operator.inputs[0].full_name]

        val = _cal_tensor_value(node.inputs[1])
        value = np.array([val])
        attrs = {"value": numpy_helper.from_array(value)}
        oopb.add_node_with_output('ConstantOfShape',
                                  cast_input_dim,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name,
                                  **attrs)


@converter_func(TYPES.FloorDiv)
def convert_tf_floor_div(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    dtype = _to_onnx_type(node.outputs[0].dtype)
    if dtype in [oopb.float16, oopb.float, oopb.double]:
        div_node = oopb.apply_div(operator.input_full_names,
                                  name=operator.full_name + '_div')[0]
        oopb.apply_op_with_output('apply_floor', div_node,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name)
    else:
        oopb.apply_op_with_output('apply_div', operator.input_full_names,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name)


@converter_func(TYPES.FloorMod)
def convert_tf_floor_mod(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    div_node = oopb.apply_div(operator.input_full_names,
                              name=operator.full_name + '_div')
    input0_dtype = _to_onnx_type(node.inputs[0].dtype)
    if input0_dtype in [oopb.float16, oopb.float, oopb.double]:
        div_floor_node = oopb.apply_floor(div_node, name=operator.full_name + '_floor')
    else:
        div_floor_node = div_node
    mul_node = oopb.apply_mul(div_floor_node + [operator.input_full_names[1]],
                              name=operator.full_name + '_mul')
    oopb.apply_op_with_output('apply_sub', [operator.input_full_names[0]] + mul_node,
                              operator.outputs[0].full_name,
                              name=operator.full_name + '_sub')


@converter_func(TYPES.FusedBatchNorm)
def convert_tf_fused_batch_norm(scope, operator, container):
    _convert_tf_fused_batch_norm_core(scope, operator, container)


@converter_func(TYPES.FusedBatchNormV2)
def convert_tf_fused_batch_norm_v2(scope, operator, container):
    _convert_tf_fused_batch_norm_core(scope, operator, container)


@converter_func(TYPES.FusedBatchNormV3)
def convert_tf_fused_batch_norm_v3(scope, operator, container):
    _convert_tf_fused_batch_norm_core(scope, operator, container)


@converter_func(TYPES.GatherV2)
def convert_tf_gather_v2(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = _cal_tensor_value(node.inputs[2]).tolist()
    oopb.apply_op_with_output("apply_gather",
                              [operator.inputs[0].full_name, operator.inputs[1].full_name],
                              operator.output_full_names,
                              name=operator.full_name,
                              axis=axis)


@converter_func(TYPES.GatherNd)
def convert_tf_gather_nd(scope, operator, container):
    if operator.target_opset < 11:
        raise ValueError("GatherND op is not supported for opset < 11")
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    indices_dtype = _to_onnx_type(node.inputs[1].dtype)
    if indices_dtype != oopb.int64:
        cast_node = oopb.apply_cast(operator.inputs[1].full_name,
                                    to=oopb.int64,
                                    name=operator.full_name + '_cast')[0]
    else:
        cast_node = operator.inputs[1].full_name
    oopb.add_node_with_output('GatherND',
                              [operator.inputs[0].full_name, cast_node],
                              operator.outputs[0].full_name,
                              name=operator.full_name)


@converter_func(TYPES.IdentityN)
def convert_tf_identity_n(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    for idx_ in range(len(operator.input_full_names)):
        oopb.apply_op_with_output('apply_identity', operator.input_full_names[idx_],
                                  operator.output_full_names[idx_],
                                  name=operator.full_name + '_' + str(idx_))


@converter_func(TYPES.GreaterEqual)
def convert_tf_greater_equal(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_greater_or_equal', operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.LessEqual)
def convert_tf_less_equal(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_less_or_equal', operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.LinSpace)
def convert_tf_linspace(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    sub_value = oopb.apply_sub([operator.input_full_names[1], operator.input_full_names[0]],
                               name=operator.full_name + '_sub')
    sub_1_value = oopb.apply_sub([operator.input_full_names[2],
                                  ('_minus_one', _to_onnx_type(node.inputs[2].dtype),
                                   np.array(1, dtype=node.inputs[2].dtype.name))],
                                 name=operator.full_name + '_sub_1')
    cast_sub_1 = oopb.apply_cast(sub_1_value,
                                 to=_to_onnx_type(node.inputs[0].dtype),
                                 name=operator.full_name + '_sub_1_cast')
    div_value = oopb.apply_div(sub_value + cast_sub_1,
                               name=operator.full_name + '_div')

    if _to_onnx_type(node.inputs[1].dtype) in [oopb.float, oopb.double, oopb.float16]:
        delta_value = 0.0000001
    else:
        delta_value = 1
    add_delta_value = oopb.apply_add([operator.input_full_names[1],
                                      ('_add_delta', _to_onnx_type(node.inputs[1].dtype),
                                       np.array(delta_value, dtype=node.inputs[1].dtype.name))],
                                     name=operator.full_name + '_add_delta')

    oopb.add_node_with_output('Range',
                              [operator.input_full_names[0]] + add_delta_value + div_value,
                              operator.output_full_names,
                              name=operator.full_name,
                              op_version=11)


@converter_func(TYPES.LogicalAnd)
def convert_tf_logical_and(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.add_node_with_output('And',
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.LogicalNot)
def convert_tf_logical_not(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.add_node_with_output('Not',
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.LogSoftmax)
def convert_tf_logsoftmax(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    logits_rank = len(_cal_tensor_shape(node.inputs[0]))
    attrs = _to_onnx_attrs(node)
    axis = attrs['axis'] if hasattr(attrs, 'axis') else -1
    if operator.target_opset < 11 and axis < 0:
        axis += logits_rank

    oopb.add_node_with_output('LogSoftmax',
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              axis=axis)


def _convert_tf_maximum_minimum(scope, operator, container, oopb, apply_func):
    node = operator.raw_operator
    supported_types = [oopb.double, oopb.float, oopb.float16]
    if container.target_opset >= 12:
        supported_types.extend([oopb.int32, oopb.int64])
    output_type = _to_onnx_type(node.outputs[0].dtype)
    need_cast = False
    cast_inputs = []

    for idx, inp in enumerate(node.inputs):
        inp_type = _to_onnx_type(inp.dtype)
        if inp_type not in supported_types:
            diff_output = oopb.apply_cast(inp.name,
                                          to=oopb.float,
                                          name=operator.full_name + '_input_' + str(idx))
            cast_inputs.extend(diff_output)
            need_cast = True
        else:
            cast_inputs.append(inp.name)

    # tensorflow minimum/maximum does support broadcast, onnx < opset 8 does not.
    # handle this by doing something like:
    # y = min(x1, add(x2, sub(x1, x1))), where x1, x2 are the inputs and x2 is a scalar
    # this will create a tensor of zeros of the shape of x1, adds x2 to it (which broadcasts) and use that for min.
    broadcast_inputs = []
    needs_broadcast_op = []
    if operator.target_opset < 8:
        output_shape = _cal_tensor_shape(node.outputs[0])
        has_correct_shape = []
        for i, input_name in enumerate(node.inputs):
            input_shape = _cal_tensor_shape(node.inputs[i])
            if input_shape != output_shape:
                needs_broadcast_op.append(i)
            else:
                has_correct_shape.append(cast_inputs[i])

    if needs_broadcast_op:
        has_correct_shape = has_correct_shape[0]
        for i in range(len(cast_inputs)):
            if i in needs_broadcast_op:
                # get a tensor with zeros (since there is no Fill op as of opset8)
                sub_node = oopb.apply_sub([has_correct_shape, has_correct_shape],
                                          name=operator.full_name + '_diff_' + str(i))
                # use add as 'broadcast' op
                add_node = oopb.apply_add([cast_inputs[i]] + sub_node,
                                          name=operator.full_name + '_add_' + str(i))
                broadcast_inputs.extend(add_node)
            else:
                broadcast_inputs.append(cast_inputs[i])
    else:
        broadcast_inputs = cast_inputs

    op_postfix = '_max' if apply_func == oopb.apply_max else '_min'
    max_node = apply_func(broadcast_inputs,
                          name=operator.full_name + op_postfix)

    if need_cast:
        oopb.apply_op_with_output("apply_cast",
                                  max_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_castback',
                                  to=output_type)
    else:
        oopb.apply_op_with_output("apply_identity",
                                  max_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_identity')


@converter_func(TYPES.Maximum)
def convert_tf_maximum(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    _convert_tf_maximum_minimum(scope, operator, container, oopb, oopb.apply_max)


@converter_func(TYPES.Minimum)
def convert_tf_minimum(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    _convert_tf_maximum_minimum(scope, operator, container, oopb, oopb.apply_min)


@converter_func(TYPES.NonMaxSuppressionV2, TYPES.NonMaxSuppressionV3)
def convert_tf_nonmaxsuppression(scope, operator, container):
    if operator.target_opset < 10:
        raise ValueError("NonMaxSuppression op is not supported for opset < 10")
    else:
        oopb = OnnxOperatorBuilder(container, scope)
        input_0 = oopb.apply_unsqueeze(operator.inputs[0].full_name,
                                       name=operator.full_name + '_unsqueeze_0',
                                       axes=[0])
        input_1 = oopb.apply_unsqueeze(operator.inputs[1].full_name,
                                       name=operator.full_name + '_unsqueeze_1',
                                       axes=[0, 1])
        input_2 = oopb.apply_cast(operator.inputs[2].full_name,
                                  to=oopb.int64,
                                  name=operator.full_name + '_cast_0')
        non_max_v = 10 if operator.target_opset < 11 else 11
        nonmaxsuppress = oopb.add_node('NonMaxSuppression',
                                       input_0 + input_1 + input_2 + operator.input_full_names[3:],
                                       operator.full_name + '_nonmax',
                                       op_version=non_max_v)
        slice_node = oopb.apply_slice(nonmaxsuppress,
                                      name=operator.full_name + '_slice',
                                      starts=[2], ends=[3], axes=[1])
        squeeze_node = oopb.apply_squeeze(slice_node,
                                          name=operator.full_name + '_squeeze',
                                          axes=[1])
        oopb.apply_op_with_output("apply_cast",
                                  squeeze_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_castback',
                                  to=oopb.int32)


def _make_range_const(scope, operator, container, start, limit, delta, onnx_type):
    start = _cal_tensor_value(start).tolist()
    limit = _cal_tensor_value(limit).tolist()
    delta = _cal_tensor_value(delta).tolist()
    val = np.arange(start, limit, delta)
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.add_node_with_output('Identity',
                              [('_start', onnx_type, val)],
                              operator.outputs[0].full_name,
                              name=operator.full_name + '_range')


def _make_range_non_const(scope, operator, container, start, limit, delta, onnx_type):
    oopb = OnnxOperatorBuilder(container, scope)
    diff_node = oopb.apply_sub([limit.name, start.name],
                               name=operator.full_name + '_diff')

    if onnx_type in [oopb.int32, oopb.int64]:
        diff_output = oopb.apply_cast(diff_node,
                                      to=oopb.float,
                                      name=operator.full_name + '_cast_diff')
        delta_cast = oopb.apply_cast(delta.name,
                                     to=oopb.float,
                                     name=operator.full_name + '_cast_delta')
    else:
        diff_output = diff_node
        delta_cast = [delta.name]

    div_node = oopb.apply_div(diff_output + delta_cast,
                              name=operator.full_name + '_div')
    ceil_node = oopb.add_node("Ceil",
                              div_node,
                              name=operator.full_name + '_ceil')
    trip_count_node = oopb.apply_cast(ceil_node,
                                      to=oopb.int64,
                                      name=operator.full_name + '_trip_cnt')
    loop_inputs = [trip_count_node[0],
                   # TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE maps BOOL to INT32
                   # so we need change np.array(True, dtype='bool') to int32 here
                   ('_cond', oopb.bool, np.array(1, dtype='int32')),
                   start.name]
    from onnx import helper
    n1 = helper.make_node("Identity", ["cond"], ["cond_out"], name="n1")
    n2 = helper.make_node("Add", ["prev", delta.name], ["current"], name="n2")
    n3 = helper.make_node("Identity", ["prev"], ["range"], name="n3")

    graph_proto = helper.make_graph(
        nodes=[n1, n2, n3],
        name="test",
        inputs=[helper.make_tensor_value_info("i", oopb.int64, []),
                helper.make_tensor_value_info("cond", oopb.bool, []),
                helper.make_tensor_value_info("prev", onnx_type, [])],
        outputs=[helper.make_tensor_value_info("cond_out", oopb.bool, []),
                 helper.make_tensor_value_info("current", onnx_type, []),
                 helper.make_tensor_value_info("range", onnx_type, [])],
        initializer=[]
    )
    loop_node = oopb.add_node_all("Loop",
                                  loop_inputs,
                                  name=operator.full_name + '_loop',
                                  outputs_num=2,
                                  body=graph_proto)
    oopb.apply_op_with_output("apply_identity",
                              loop_node[1],
                              operator.output_full_names,
                              name=operator.full_name + '_identity')


def _make_range(scope, operator, container, start, limit, delta, onnx_type):
    if all(_cal_tensor_value(n) is not None for n in [start, limit, delta]) is True:
        _make_range_const(scope, operator, container, start, limit, delta, onnx_type)
    else:
        _make_range_non_const(scope, operator, container, start, limit, delta, onnx_type)


@converter_func(TYPES.Range)
def convert_tf_range(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    if operator.target_opset < 11:
        onnx_type = _to_onnx_type(node.outputs[0].dtype)
        _make_range(scope, operator, container, node.inputs[0], node.inputs[1], node.inputs[2], onnx_type)
    else:
        oopb.add_node_with_output("Range",
                                  operator.input_full_names,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name + '_range',
                                  op_version=11)


@converter_func(TYPES.TD_Reshape)
def convert_reshape_timedistributed(scope, operator, container):
    input_name = operator.get_attr('input_name')
    target_shape = operator.get_attr('target_shape')
    if input_name is None:
        apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                      operator_name=operator.full_name, desired_shape=target_shape)
    else:
        oopb = OnnxOperatorBuilder(container, scope)
        shape0 = oopb.apply_shape(input_name, name=operator.full_name + '_shape')
        cropped_tensor_name = oopb.add_node('Slice',
                                            [shape0[0],
                                             ('_start', oopb.int64, np.array([0], dtype=np.int64)),
                                             ('_end', oopb.int64, np.array([2], dtype=np.int64))
                                             ],
                                            operator.inputs[0].full_name + '_cropping',
                                            op_version=11)
        concat = oopb.apply_concat([cropped_tensor_name,
                                    ('_start', oopb.int64, np.array(target_shape, dtype=np.int64)),
                                    ], name=operator.full_name + '_concat')
        apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                      operator_name=operator.full_name, desired_shape=concat[0])


@converter_func(TYPES.All, TYPES.Any)
def convert_tf_any_all(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = _cal_tensor_value(node.inputs[1]).tolist()
    axis = [axis] if np.isscalar(axis) else axis

    # It is fine to have nagative reduce_dim.
    cast_op = oopb.apply_cast(operator.input_full_names[0],
                              to=oopb.float,
                              name=operator.full_name + '_cast')
    keepdims = node.get_attr("keep_dims")
    op_type = "ReduceMin" if node.type == "All" else "ReduceSum"
    reduce_op = oopb.add_node(op_type, cast_op,
                              axes=axis,
                              keepdims=keepdims,
                              name=operator.full_name + '_reduce')
    oopb.apply_op_with_output('apply_greater',
                              [reduce_op, np.array(0, dtype=np.float32)],
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.Pack)
def convert_tf_pack(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = node.get_attr('axis')
    if axis < 0 and operator.target_opset < 11:
        axis += len(_cal_tensor_shape(node.inputs[0])) + 1

    inputs = []
    for i in range(len(node.inputs)):
        unsqueeze = oopb.add_node('Unsqueeze',
                                  operator.inputs[i].full_name,
                                  operator.full_name + '_unsqueeze' + str(i), axes=[axis])
        inputs.append(unsqueeze)

    oopb.apply_op_with_output("apply_concat",
                              inputs,
                              operator.outputs[0].full_name,
                              name=operator.full_name + '_concat',
                              axis=axis)


def _convert_tf_pad(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    paddings_value = _cal_tensor_value(node.inputs[1])
    if paddings_value is None:
        padding_dtype = _to_onnx_type(node.inputs[1].dtype)
        if padding_dtype != oopb.int64:
            cast_node = oopb.apply_cast(operator.input_full_names[1],
                                        to=oopb.int64,
                                        name=operator.full_name + '_paddings_cast')
        else:
            cast_node = operator.input_full_names[1]
        transpose_node_1 = oopb.apply_transpose(cast_node,
                                                name=operator.full_name + '_transpose_1',
                                                perm=[1, 0])
        paddings = oopb.apply_reshape(transpose_node_1,
                                      name=operator.full_name + '_reshape',
                                      desired_shape=[-1])[0]
    else:
        paddings = np.array(_cal_tensor_value(node.inputs[1])).transpose().flatten()
    attrs = _to_onnx_attrs(node)
    mode = attrs["mode"] if "mode" in attrs else None

    if mode:
        mode = mode.decode("utf-8").lower()
    if mode not in [None, "constant", "reflect"]:
        raise ValueError(mode + " pad mode is not supported")

    origin_dtype = _to_onnx_type(node.outputs[0].dtype)
    if origin_dtype not in [oopb.float16, oopb.float,
                            oopb.double]:
        cast_op = oopb.apply_cast(operator.input_full_names[0],
                                  to=oopb.float,
                                  name=operator.full_name + '_cast')
    else:
        cast_op = operator.input_full_names[0]

    if mode in [None, "constant"] and len(node.inputs) == 3:
        const_val = _cal_tensor_value(node.inputs[2]).tolist()
    else:
        const_val = None

    if operator.target_opset < 11:
        pad_node = oopb.apply_pad(cast_op,
                                  name=operator.full_name + '_pad',
                                  mode=mode,
                                  pads=paddings,
                                  value=const_val)
    else:
        pad_node = oopb.apply_pad(cast_op,
                                  name=operator.full_name + '_pad',
                                  mode=mode,
                                  pads=paddings,
                                  value=const_val,
                                  onnx_type=_to_onnx_type(node.inputs[0].dtype))

    if origin_dtype not in [oopb.float16, oopb.float,
                            oopb.double]:
        oopb.apply_op_with_output("apply_cast",
                                  pad_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_castback',
                                  to=origin_dtype)
    else:
        oopb.apply_op_with_output("apply_identity",
                                  pad_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_identity')


@converter_func(TYPES.Pad)
def convert_tf_pad(scope, operator, container):
    _convert_tf_pad(scope, operator, container)


@converter_func(TYPES.PadV2)
def convert_tf_pad_v2(scope, operator, container):
    _convert_tf_pad(scope, operator, container)


@converter_func(TYPES.MirrorPad)
def convert_tf_mirror_pad(scope, operator, container):
    _convert_tf_pad(scope, operator, container)


def _convert_tf_reduce_op(scope, operator, container, onnx_op):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axes = _cal_tensor_value(node.inputs[1]).tolist()
    axes = [axes] if np.isscalar(axes) else axes

    if operator.target_opset < 11:
        input_shape = _cal_tensor_shape(node.inputs[0])
        if input_shape is None:
            if any([val < 0 for val in axes]):
                raise ValueError("reduce_op: cannot have negative axis because we don't know input rank")
        else:
            input_rank = len(input_shape)
            axes = [val + input_rank if val < 0 else val for val in axes]

    keepdims = node.get_attr("keep_dims")
    oopb.add_node_with_output(onnx_op,
                              operator.inputs[0].full_name,
                              operator.outputs[0].full_name,
                              name=operator.full_name + '_reduce_min',
                              axes=axes, keepdims=keepdims)


@converter_func(TYPES.Max)
def convert_tf_max(scope, operator, container):
    _convert_tf_reduce_op(scope, operator, container, 'ReduceMax')


@converter_func(TYPES.Min)
def convert_tf_min(scope, operator, container):
    _convert_tf_reduce_op(scope, operator, container, 'ReduceMin')


@converter_func(TYPES.Mean)
def convert_tf_mean(scope, operator, container):
    _convert_tf_reduce_op(scope, operator, container, 'ReduceMean')


@converter_func(TYPES.Sum)
def convert_tf_sum(scope, operator, container):
    _convert_tf_reduce_op(scope, operator, container, 'ReduceSum')


@converter_func(TYPES.Prod)
def convert_tf_prod(scope, operator, container):
    _convert_tf_reduce_op(scope, operator, container, 'ReduceProd')


@converter_func(TYPES.Reshape)
def convert_tf_reshape(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    if _cal_tensor_value(node.inputs[1]) is None:
        temp_shape_value = node.inputs[1].name
        shape_value = temp_shape_value
        shape_dtype = _to_onnx_type(node.inputs[0].dtype)
        if shape_dtype != oopb.int64:
            shape_value = oopb.apply_cast(temp_shape_value,
                                          to=oopb.int64,
                                          name=operator.full_name + '_cast')[0]
    else:
        shape_value = _cal_tensor_value(node.inputs[1]).tolist()

    oopb.apply_op_with_output("apply_reshape",
                              operator.inputs[0].full_name,
                              operator.outputs[0].full_name,
                              name=operator.full_name + '_reshape',
                              desired_shape=shape_value)


@converter_func(TYPES.ScatterNd)
def convert_tf_scatter_nd(scope, operator, container):
    if operator.target_opset < 11:
        raise ValueError("ScatterNd op is not supported for opset = " + str(operator.target_opset))
    else:
        oopb = OnnxOperatorBuilder(container, scope)
        node = operator.raw_operator

        const_shape_dtype = _to_onnx_type(node.inputs[2].dtype)
        if const_shape_dtype != oopb.int64:
            const_of_shape_input = oopb.apply_cast(operator.inputs[2].full_name,
                                                   to=oopb.int64,
                                                   name=operator.full_name + '_const_of_shape_input')
        else:
            const_of_shape_input = [operator.inputs[2].full_name]

        np_type = TENSOR_TYPE_TO_NP_TYPE[operator.inputs[1].type.to_onnx_type().tensor_type.elem_type]
        np_val = np.array([0], dtype=np_type)
        onnx_tensor = numpy_helper.from_array(np_val, operator.inputs[2].full_name + '_value')
        const_of_shape = oopb.add_node('ConstantOfShape',
                                       const_of_shape_input,
                                       operator.inputs[2].full_name + '_const_of_shape',
                                       value=onnx_tensor)

        node_input_0_dtype = _to_onnx_type(node.inputs[0].dtype)
        if node_input_0_dtype != oopb.int64:
            node_input_0_cast = oopb.apply_cast(operator.inputs[0].full_name,
                                                to=oopb.int64,
                                                name=operator.full_name + '_input_0')
        else:
            node_input_0_cast = [operator.inputs[0].full_name]

        oopb.add_node_with_output('ScatterND',
                                  [const_of_shape] + node_input_0_cast + [operator.inputs[1].full_name],
                                  operator.outputs[0].full_name,
                                  name=operator.full_name + '_scatter_nd')


@converter_func(TYPES.Select)
def convert_tf_select(scope, operator, container):
    if operator.target_opset < 9:
        raise ValueError("Select op is not supported for opset = " + str(operator.target_opset))
    else:
        oopb = OnnxOperatorBuilder(container, scope)
        node = operator.raw_operator
        cond_shape = _cal_tensor_shape(node.inputs[0])
        input_shape = _cal_tensor_shape(node.inputs[1])
        if input_shape is None:
            input_shape = _cal_tensor_shape(node.inputs[2])
        input_rank = len(input_shape)
        if len(cond_shape) == 1 and input_rank > 1:
            broadcast_shape = [cond_shape[0]] + [1] * (input_rank - 1)
            reshape_node = oopb.apply_reshape(operator.inputs[0].full_name,
                                              name=operator.full_name + '_reshape',
                                              desired_shape=broadcast_shape)
            input_nodes = reshape_node + operator.input_full_names[1:]
        else:
            input_nodes = operator.input_full_names

        oopb.add_node_with_output('Where',
                                  input_nodes,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name + '_where',
                                  op_version=9)


@converter_func(TYPES.Size)
def convert_tf_size(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    dtype = _to_onnx_type(node.outputs[0].dtype)
    if dtype != oopb.int64:
        size_node = oopb.add_node('Size',
                                  operator.inputs[0].full_name,
                                  operator.inputs[0].full_name + '_size')
        oopb.apply_op_with_output("apply_cast",
                                  size_node,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name + '_size_cast',
                                  to=dtype)
    else:
        oopb.add_node_with_output('Size',
                                  operator.inputs[0].full_name,
                                  operator.output_full_names,
                                  name=operator.inputs[0].full_name + '_size')


def _convert_tf_resize(scope, operator, container, mode):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    shape = _cal_tensor_shape(node.inputs[0])
    target_shape = _cal_tensor_value(node.inputs[1])

    if shape and shape[1] is not None and shape[2] is not None and target_shape is not None:
        n, h, w, c = shape
        nh, nw = target_shape
        scale_val = np.array([1.0, 1.0, float(nh) / h, float(nw) / w]).astype(np.float32)
        scales = ('_scale', oopb.float, scale_val)
    else:
        if operator.target_opset < 10:
            raise ValueError("dynamic shape is not supported for Upsample when opset = " + str(operator.target_opset))
        input_shape = oopb.add_node('Shape',
                                    operator.inputs[0].full_name,
                                    operator.inputs[0].full_name + '_input_shape')
        sliced_score = oopb.add_node('Slice',
                                     [input_shape,
                                      ('_start', oopb.int64, np.array([1], dtype='int64')),
                                      ('_end', oopb.int64, np.array([3], dtype='int64')),
                                      ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                      ],
                                     operator.inputs[0].full_name + '_sliced')
        ori_cast = oopb.add_node('Cast',
                                 sliced_score,
                                 operator.inputs[0].full_name + '_ori_cast', to=oopb.float)
        target_cast = oopb.add_node('Cast',
                                    operator.inputs[1].full_name,
                                    operator.inputs[1].full_name + '_target_cast', to=oopb.float)
        scales_hw = oopb.add_node('Div',
                                  [target_cast, ori_cast],
                                  operator.inputs[1].full_name + '_scales_hw')
        scales = oopb.add_node('Concat',
                               [('_concat', oopb.float, np.array([1.0, 1.0], dtype='float32')),
                                scales_hw
                                ],
                               operator.inputs[0].full_name + '_concat',
                               axis=0)

    input_nchw = oopb.add_node('Transpose',
                               operator.inputs[0].full_name,
                               operator.inputs[0].full_name + '_transpose',
                               perm=[0, 3, 1, 2])
    attrs = {"mode": mode}
    if operator.target_opset >= 11:
        if node.get_attr('align_corners'):
            attrs['coordinate_transformation_mode'] = 'align_corners'
        else:
            attrs['coordinate_transformation_mode'] = 'asymmetric'

    if attrs['mode'] == 'nearest':
        attrs['nearest_mode'] = 'floor'
    if operator.target_opset < 10:
        op_type = 'Upsample'
    else:
        op_type = 'Resize'

    if operator.target_opset < 8:
        scale_h = float(nh) / h
        scale_w = float(nw) / w
        if scale_h < 1.0 or scale_w < 1.0:
            raise ValueError("Upsample op need scale value >= 1.0")
        attrs = {"mode": mode, "scales": [1.0, 1.0, scale_h, scale_w]}
        upsample = oopb.add_node(op_type,
                                 input_nchw,
                                 operator.inputs[0].full_name + '_upsample',
                                 **attrs)
    elif operator.target_opset < 11:
        upsample = oopb.add_node(op_type,
                                 [input_nchw,
                                  scales],
                                 operator.inputs[0].full_name + '_upsample',
                                 mode=mode)
    else:
        upsample = oopb.add_node(op_type,
                                 [input_nchw,
                                  ('_rois', oopb.float, np.array([0.0, 0.0, 1.0, 1.0], dtype='float32')),
                                  scales],
                                 operator.inputs[0].full_name + '_upsample',
                                 **attrs)
    oopb.add_node_with_output('Transpose',
                              upsample,
                              operator.output_full_names,
                              name=operator.inputs[0].full_name + '_transpose_2',
                              perm=[0, 2, 3, 1])


@converter_func(TYPES.ResizeBilinear)
def convert_tf_resize_bilinear(scope, operator, container):
    _convert_tf_resize(scope, operator, container, "linear")


@converter_func(TYPES.ResizeNearestNeighbor)
def convert_tf_resize_nearest_neighbor(scope, operator, container):
    _convert_tf_resize(scope, operator, container, "nearest")


@converter_func(TYPES.Round)
def convert_tf_round(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    if operator.target_opset < 11:
        add_output_name = oopb.add_node('Add',
                                        [operator.inputs[0].full_name,
                                         ('_add', oopb.float, np.array(-0.5, dtype=np.float32))
                                         ],
                                        operator.inputs[0].full_name + '_add')
        cast_0 = oopb.add_node('Cast',
                               add_output_name,
                               operator.inputs[0].full_name + '_0_cast', to=oopb.float)
        oopb.add_node_with_output("Ceil",
                                  cast_0,
                                  operator.output_full_names,
                                  name=operator.full_name)
    else:
        oopb.add_node_with_output("Round",
                                  operator.input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name)


@converter_func(TYPES.Rsqrt)
def convert_tf_rsqrt(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    sqrt_node = oopb.add_node('Sqrt',
                              operator.inputs[0].full_name,
                              operator.inputs[0].full_name + '_sqrt')
    oopb.apply_op_with_output("apply_reciprocal",
                              sqrt_node,
                              operator.output_full_names,
                              name=operator.full_name + '_cast')


@converter_func(TYPES.Shape)
def convert_tf_shape(scope, operator, container):
    node = operator.raw_operator
    dtype = _to_onnx_type(node.outputs[0].dtype)
    oopb = OnnxOperatorBuilder(container, scope)
    shape_node = oopb.add_node('Shape',
                               operator.input_full_names[0],
                               operator.input_full_names[0] + '_shape')
    if dtype == oopb.int64:
        oopb.add_node_with_output('Identity',
                                  shape_node,
                                  operator.output_full_names,
                                  operator.inputs[0].full_name + '_identity')
    else:
        oopb.apply_op_with_output("apply_cast",
                                  shape_node,
                                  operator.output_full_names,
                                  name=operator.full_name + '_cast',
                                  to=dtype)


@converter_func(TYPES.Split)
def convert_tf_split(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    split_dims = _cal_tensor_value(node.inputs[0]).tolist()
    oopb.apply_op_with_output('apply_split',
                              operator.input_full_names[1:],
                              operator.output_full_names,
                              operator.inputs[0].full_name + '_split',
                              axis=split_dims)


@converter_func(TYPES.SplitV)
def convert_tf_splitv(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    split = _cal_tensor_value(node.inputs[1]).tolist()
    split_dims = _cal_tensor_value(node.inputs[2]).tolist()
    oopb.apply_op_with_output('apply_split',
                              operator.input_full_names[0],
                              operator.output_full_names,
                              operator.inputs[0].full_name + '_split',
                              split=split,
                              axis=split_dims)


@converter_func(TYPES.Squeeze)
def convert_tf_squeeze(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    shape = _cal_tensor_shape(node.inputs[0])
    axis = node.get_attr('squeeze_dims')

    if axis:
        neg_axis = any([val < 0 for val in axis])
        if neg_axis and operator.target_opset < 11:
            shape_len = len(shape)
            axis = [a + shape_len if a < 0 else a for a in axis]
    else:
        axis = [i for i, j in enumerate(shape) if j == 1]

    if shape is None:
        raise ValueError("Squeeze input shape cannot be None for node {}".format(node.name))

    oopb.add_node_with_output('Squeeze',
                              operator.input_full_names[0],
                              operator.output_full_names,
                              operator.inputs[0].full_name + '_squeeze',
                              axes=axis)


@converter_func(TYPES.Tile)
def convert_tf_tile(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    cast_1 = oopb.add_node('Cast',
                           operator.inputs[1].full_name,
                           operator.inputs[1].full_name + '_1_cast', to=oopb.int64)
    oopb.add_node_with_output('Tile',
                              [operator.input_full_names[0], cast_1],
                              operator.output_full_names,
                              operator.inputs[0].full_name + '_tile')


@converter_func(TYPES.TopKV2)
def convert_tf_topkv2(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    cast_0 = oopb.add_node('Cast',
                           operator.inputs[0].full_name,
                           operator.inputs[0].full_name + '_0_cast', to=oopb.float)
    k = _cal_tensor_value(node.inputs[1])
    if k is None:
        if operator.target_opset < 10:
            raise ValueError("TopK op k need be static until opset 10")
        cast_1 = oopb.add_node('Cast',
                               operator.inputs[1].full_name,
                               operator.inputs[1].full_name + '_1_cast', to=oopb.int64)
        unsqueeze = oopb.add_node('Unsqueeze',
                                  cast_1,
                                  operator.inputs[1].full_name + '_unsqueeze', axes=[0])
        k_value = unsqueeze
    else:
        k_value = k.item(0)
    output_1_dtype = _to_onnx_type(node.outputs[1].dtype)
    if output_1_dtype == oopb.int64:
        oopb.apply_op_with_output('apply_topk',
                                  [cast_0],
                                  operator.output_full_names,
                                  operator.inputs[0].full_name + '_topk',
                                  k=k_value)
    else:
        topk = oopb.apply_topk([cast_0],
                               operator.inputs[0].full_name + '_topk',
                               outputs_num=2,
                               k=k_value)
        oopb.apply_op_with_output("apply_identity",
                                  topk[0],
                                  operator.output_full_names[0],
                                  name=operator.full_name + '_identity')
        oopb.apply_op_with_output("apply_cast",
                                  topk[1],
                                  operator.output_full_names[1],
                                  name=operator.full_name + '_cast',
                                  to=output_1_dtype)


@converter_func(TYPES.Transpose)
def convert_tf_transpose(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    perm = _cal_tensor_value(node.inputs[1])
    oopb.apply_op_with_output("apply_transpose",
                              operator.inputs[0].full_name,
                              operator.output_full_names,
                              name=operator.full_name,
                              perm=perm)


@converter_func(TYPES.Cast)
def convert_tf_cast(scope, operator, container):
    node = operator.raw_operator
    to = _to_onnx_type(node.get_attr("DstT"))
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output("apply_cast",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              to=to)


@converter_func(TYPES.NotEqual)
def convert_tf_not_equal(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    if operator.target_opset >= 11:
        equal_out = oopb.add_node('Equal', [operator.inputs[0].full_name, operator.inputs[1].full_name],
                                  operator.full_name + '_mask')
        oopb.add_node_with_output('Not', equal_out,
                                  operator.output_full_names,
                                  name=operator.full_name + '_not')
    else:
        equal_input_0 = oopb.add_node('Cast', [operator.inputs[0].full_name],
                                      operator.full_name + '_input_0_cast', to=6)
        equal_input_1 = oopb.add_node('Cast', [operator.inputs[1].full_name],
                                      operator.full_name + '_input_1_cast', to=6)
        equal_out = oopb.add_node('Equal', [equal_input_0, equal_input_1],
                                  operator.full_name + '_mask')
        oopb.add_node_with_output('Not', equal_out,
                                  operator.output_full_names,
                                  name=operator.full_name + '_not')


@converter_func(TYPES.OneHot)
def convert_tf_one_hot(scope, operator, container):
    if operator.target_opset < 9:
        raise ValueError("OneHot op is not supported until opset 9")
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = node.get_attr('axis')

    depth = oopb.apply_unsqueeze(operator.inputs[1].full_name,
                                 name=operator.full_name + '_unsqueeze_1',
                                 axes=[0])
    on_value = oopb.apply_unsqueeze(operator.inputs[2].full_name,
                                    name=operator.full_name + '_unsqueeze_2',
                                    axes=[0])
    off_value = oopb.apply_unsqueeze(operator.inputs[3].full_name,
                                     name=operator.full_name + '_unsqueeze_3',
                                     axes=[0])
    off_on_value = oopb.apply_concat(off_value + on_value,
                                     name=operator.full_name + '_concat',
                                     axis=0)
    oopb.add_node_with_output('OneHot', [operator.inputs[0].full_name] + depth + off_on_value,
                              operator.output_full_names,
                              name=operator.full_name + '_one_hot', axis=axis)


@converter_func(TYPES.Pow)
def convert_tf_pow(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    if container.target_opset < 12:
        supported_types = [oopb.float16, oopb.float, oopb.double]
        for input_idx_ in range(2):
            dtype = _to_onnx_type(node.inputs[input_idx_].dtype)
            if dtype not in supported_types:
                raise ValueError("The input type of Pow is not supported for opset < 12.")
        dtype = _to_onnx_type(node.outputs[0].dtype)
        if dtype not in supported_types:
            raise ValueError("The output type of Pow is not supported for opset < 12.")

    if operator.raw_operator.inputs[1].op.type == 'Const':
        val_tensor = operator.raw_operator.inputs[1].op.get_attr('value')
        float_delta = 1e-6
        if ((len(val_tensor.float_val) > 0 and abs(val_tensor.float_val[0] - 1.0) < float_delta) or
           (len(val_tensor.int_val) > 0 and val_tensor.int_val == 1)):
            oopb.apply_op_with_output("apply_identity",
                                      operator.input_full_names[0],
                                      operator.output_full_names,
                                      name=operator.full_name)
            return

    oopb.apply_op_with_output("apply_pow",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.ReadVariableOp)
def convert_tf_read_variable_op(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    if len(node.inputs) == 1 and len(node.outputs) == 1:
        oopb.apply_op_with_output("apply_identity",
                                  operator.input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name)


@converter_func(TYPES.Relu6)
def convert_tf_relu6(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    np_type = TENSOR_TYPE_TO_NP_TYPE[operator.inputs[0].type.to_onnx_type().tensor_type.elem_type]
    zero_value = np.zeros(shape=(1,), dtype=np_type)
    oopb.apply_op_with_output("apply_relu_6",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name + '_clip',
                              zero_value=zero_value)


@converter_func(TYPES.Slice)
def convert_tf_slice(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    begin = _cal_tensor_value(node.inputs[1])
    size = _cal_tensor_value(node.inputs[2])

    if begin is not None and size is not None:
        begin_value = begin.tolist()
        size_value = size.tolist()
        end_value = []
        for begin_, size_ in zip(begin_value, size_value):
            if size_ == -1 or (begin_ < 0 and (begin_ + size_) >= 0):
                end_value.append(np.iinfo(np.int64).max)
            else:
                end_value.append(begin_ + size_)
    else:
        if operator.target_opset < 10:
            raise ValueError("Dynamic inputs for tf.slice is not supported until opset 10")

        dtype = _to_onnx_type(node.inputs[1].dtype)
        if dtype != oopb.int64:
            cast_begin = oopb.apply_cast(operator.inputs[1].full_name,
                                         to=oopb.int64,
                                         name=operator.full_name + '_begin_cast')
        else:
            cast_begin = [operator.inputs[1].full_name]

        dtype = _to_onnx_type(node.inputs[2].dtype)
        if dtype != oopb.int64:
            cast_size = oopb.apply_cast(operator.inputs[2].full_name,
                                        to=oopb.int64,
                                        name=operator.full_name + '_size_cast')
        else:
            cast_size = [operator.inputs[2].full_name]

        neg_one_size = oopb.add_node('Equal',
                                     cast_size + [('_neg_one', oopb.int64, np.array(-1, dtype=np.int64))],
                                     operator.full_name + '_equal_neg_one')
        cast_equal = oopb.apply_cast(neg_one_size,
                                     to=oopb.int64,
                                     name=operator.full_name + '_equal_cast')
        value_offset = oopb.apply_mul(
            cast_equal + [('_max_int', oopb.int64, np.array(np.iinfo(np.int64).max, dtype=np.int64))],
            name=operator.full_name + '_mul_max')
        size_adjust = oopb.apply_add(cast_size + value_offset,
                                     name=operator.full_name + '_size_adjust')
        begin_value = cast_begin[0]
        end_value = oopb.apply_add(cast_begin + size_adjust,
                                   name=operator.full_name + '_ends')[0]

    oopb.apply_op_with_output("apply_slice",
                              operator.inputs[0].full_name,
                              operator.output_full_names,
                              name=operator.full_name,
                              starts=begin_value,
                              ends=end_value)


@converter_func(TYPES.Softmax)
def convert_tf_softmax(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    logits_rank = len(_cal_tensor_shape(node.inputs[0]))
    attrs = _to_onnx_attrs(node)
    axis = attrs['axis'] if hasattr(attrs, 'axis') else -1
    if operator.target_opset < 11 and axis < 0:
        axis += logits_rank

    oopb.apply_op_with_output("apply_softmax",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              axis=axis)


@converter_func(TYPES.Square)
def convert_tf_square(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_mul',
                              operator.input_full_names + operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name)


def _process_begin_end(new_begin, new_end, stride):
    if stride >= 0:
        new_begin.append(0)
        new_end.append(sys.maxsize)
    else:
        new_begin.append(-1)
        new_end.append(-sys.maxsize)


def _prepare_StridedSlice(node, target_opset):
    max_size = sys.maxsize
    begin = _cal_tensor_value(node.inputs[1])
    if begin is None:
        begin = [0] * node.inputs[1].shape[0]
    end = _cal_tensor_value(node.inputs[2])
    if end is None:
        dynamic_end = True
        end = [max_size] * node.inputs[2].shape[0]  # this is dummy and not really used.
    else:
        # for ResNext model, end = [0, 0, 0, 64], it still works using dynamic_end=False
        # so will not set dynamic_end=True for simplicity.
        dynamic_end = False
    strides = _cal_tensor_value(node.inputs[3])
    if strides is None:
        strides = [1] * node.inputs[3].shape[0]
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask if end_mask is not None else 0
    end_mask_array = [0] * node.inputs[2].shape[0]
    end_mask_temp = end_mask
    end_mask_array_idx = 0
    while end_mask_temp > 0:
        if end_mask_temp & 1:
            end_mask_array[end_mask_array_idx] = 1
        end_mask_temp = end_mask_temp >> 1
        end_mask_array_idx += 1

    new_axis_mask = node.get_attr("new_axis_mask")
    new_axis_mask = new_axis_mask if new_axis_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask if shrink_axis_mask is not None else 0
    ellipsis_mask = node.get_attr("ellipsis_mask")
    ellipsis_mask = ellipsis_mask if ellipsis_mask is not None else 0
    extra_mask = new_axis_mask or shrink_axis_mask or ellipsis_mask
    new_begin = []
    new_end = []
    axes = []
    steps = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    ellipsis_gap = 0

    new_axis_len = 0
    cur_new_axis_mask = new_axis_mask
    while cur_new_axis_mask > 0:
        if cur_new_axis_mask & 1:
            new_axis_len += 1
        cur_new_axis_mask = cur_new_axis_mask >> 1
    new_axis_axes = []

    for idx, begin_item in enumerate(begin):
        if target_opset < 10 and strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 are supported, current stride =" + str(strides[idx]))

        if (ellipsis_mask >> idx) & 1:
            input_shape = node.inputs[0].shape  # ctx.get_shape(node.input[0])
            if input_shape is None:
                raise ValueError("StridedSlice op {} requires the shape of input".format(node.name))
            ellipsis_gap = len(input_shape) + new_axis_len - len(begin)
            for ellipsis_start_idx in range(idx, idx + ellipsis_gap + 1):
                new_begin.append(0)
                new_end.append(max_size)
                axes.append(ellipsis_start_idx)
                steps.append(1)
            continue

        if (new_axis_mask >> idx) & 1:
            new_axis_axes.append(idx + ellipsis_gap)
            continue

        end_item = end[idx]
        axes.append(idx + ellipsis_gap)
        steps.append(strides[idx])

        if (begin_mask >> idx) & 1 != 0 and (end_mask >> idx) & 1 != 0:
            _process_begin_end(new_begin, new_end, strides[idx])
            continue

        if begin_item == 0 and end_item == 0:
            _process_begin_end(new_begin, new_end, strides[idx])
            continue

        shrink_mask = (shrink_axis_mask >> idx) & 1
        if shrink_mask != 0:
            new_begin.append(begin_item)
            if begin_item == -1:
                new_end.append(max_size)
            else:
                new_end.append(begin_item + 1)
            needs_squeeze.append(idx + ellipsis_gap)
            continue

        if (begin_mask >> idx) & 1 != 0:
            new_begin.append(0) if strides[idx] >= 0 else new_begin.append(-1)
            new_end.append(end_item)
            continue

        if (end_mask >> idx) & 1 != 0:
            new_begin.append(begin_item)
            new_end.append(max_size) if strides[idx] >= 0 else new_begin.append(-max_size)
            continue

        new_begin.append(begin_item)
        new_end.append(end_item)

    return new_begin, new_end, axes, steps, needs_squeeze, \
        begin_mask, end_mask, extra_mask, new_axis_axes, end_mask_array, dynamic_end


@converter_func(TYPES.StridedSlice)
def convert_tf_strided_slice(scope, operator, container):
    node = operator.raw_operator
    new_begin, new_end, axes, steps, needs_squeeze, \
        begin_mask, end_mask, extra_mask, new_axis_axes, end_mask_array, dynamic_end = _prepare_StridedSlice(
            node, operator.target_opset)
    oopb = OnnxOperatorBuilder(container, scope)

    if len(new_axis_axes) > 0:
        new_axis_unsqueeze = oopb.add_node('Unsqueeze',
                                           operator.inputs[0].full_name,
                                           operator.inputs[0].full_name + '_unsqueeze',
                                           axes=new_axis_axes)
    else:
        new_axis_unsqueeze = operator.inputs[0].full_name

    if operator.target_opset < 10:
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
        if dynamic_end:
            raise ValueError("Slice op does not support dynamic input for opset < 10.")
        cropped_tensor_name = oopb.add_node('Slice',
                                            new_axis_unsqueeze,
                                            operator.inputs[0].full_name + '_cropping',
                                            starts=new_begin, ends=new_end, axes=axes)
    else:
        if extra_mask or begin_mask:
            cast_node_begin = True
        else:
            start_cast = oopb.add_node('Cast',
                                       operator.inputs[1].full_name,
                                       operator.inputs[1].full_name + '_start_cast', to=7)
            cast_node_begin = False

        if extra_mask or end_mask:
            cast_node_end = True
        else:
            end_cast = oopb.add_node('Cast',
                                     operator.inputs[2].full_name,
                                     operator.inputs[2].full_name + '_end_cast', to=7)
            cast_node_end = False

        data_shape = oopb.add_node('Shape',
                                   operator.inputs[0].full_name,
                                   operator.inputs[0].full_name + '_shape',
                                   op_version=9)
        data_shape_mul = oopb.apply_mul([data_shape,
                                         ('_start', oopb.int64, np.array(end_mask_array, dtype=np.int64))],
                                        name=operator.inputs[0].full_name + '_shape_mul')
        end_mask_array_neg = 1 - np.array(end_mask_array, dtype=np.int64)
        end_cast_0 = oopb.apply_cast(node.inputs[2].name,
                                     name=node.inputs[2].name + '_end_cast_0',
                                     to=7)
        end_cast_0_mul = oopb.apply_mul(end_cast_0 +
                                        [('_start', oopb.int64, np.array(end_mask_array_neg, dtype=np.int64))],
                                        name=operator.inputs[0].full_name + '_end_cast_0_mul')
        end_combine = oopb.apply_add(data_shape_mul + end_cast_0_mul,
                                     name=operator.inputs[0].full_name + '_end_combine')

        if cast_node_end:
            if dynamic_end:
                end_point = end_combine[0]
            else:
                end_point = ('_end', oopb.int64, np.array(new_end, dtype=np.int64))
        else:
            end_point = end_cast

        cropped_tensor_name = oopb.add_node('Slice',
                                            [new_axis_unsqueeze,
                                             ('_start', oopb.int64,
                                              np.array(new_begin, dtype=np.int64)) if cast_node_begin else start_cast,
                                             end_point,
                                             ('_axes', oopb.int64, np.array(axes, dtype=np.int64)),
                                             ('_steps', oopb.int64, np.array(steps, dtype=np.int64))
                                             ],
                                            operator.inputs[0].full_name + '_cropping')

    if needs_squeeze:
        oopb.add_node_with_output('Squeeze',
                                  cropped_tensor_name,
                                  operator.output_full_names,
                                  operator.inputs[0].full_name + '_squeeze',
                                  axes=needs_squeeze)
    else:
        oopb.add_node_with_output('Identity',
                                  cropped_tensor_name,
                                  operator.output_full_names,
                                  operator.inputs[0].full_name + '_identity')


@converter_func(TYPES.TensorScatterUpdate)
def convert_tf_tensor_scatter_update(scope, operator, container):
    if operator.target_opset < 11:
        raise ValueError("TensorScatterUpdate op is not supported for opset = " + str(operator.target_opset))
    else:
        oopb = OnnxOperatorBuilder(container, scope)
        node = operator.raw_operator

        indices = _cal_tensor_value(node.inputs[1])
        indices_arr = np.array(indices)
        indices_tensor = numpy_helper.from_array(indices_arr, operator.inputs[1].full_name + '_value')
        container.add_initializer_from_tensor(indices_tensor)

        updates_name = operator.inputs[2].full_name
        updates = _cal_tensor_value(node.inputs[2])
        if (updates is not None):
            updates_arr = np.array(updates)
            updates_tensor = numpy_helper.from_array(updates_arr, operator.inputs[2].full_name + '_value')
            container.add_initializer_from_tensor(updates_tensor)
            updates_name = operator.inputs[2].full_name + '_value'

        cast_indices = oopb.apply_cast(indices_tensor.name,
                                       to=oopb.int64,
                                       name=operator.full_name + '_input_1_cast')
        oopb.add_node_with_output('ScatterND',
                                  [operator.inputs[0].full_name, cast_indices[0], updates_name],
                                  operator.outputs[0].full_name,
                                  name=operator.full_name + '_tensor_scatter_nd',
                                  op_version=11)


@converter_func(TYPES.Unpack)
def convert_tf_unpack(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis_val = node.get_attr('axis')
    input_shape = _cal_tensor_shape(node.inputs[0])
    if axis_val < 0 and operator.target_opset < 11:
        axis_val = len(input_shape) + axis_val

    split_node = oopb.add_node_all('Split',
                                   operator.inputs[0].full_name,
                                   operator.full_name + '_split',
                                   outputs_num=input_shape[axis_val],
                                   axis=axis_val)

    for i in range(len(split_node)):
        oopb.apply_op_with_output("apply_squeeze",
                                  split_node[i],
                                  operator.outputs[i].full_name,
                                  name=operator.full_name + '_squeeze_' + str(i),
                                  axes=[axis_val])


def _convert_tf_var_handle_helper(scope, operator, container, var_handle_name, graph_op_type):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator

    if is_tf2:
        v_output = node.outputs[0].name
        get_assign_value = False
        for graph_node_name in node.graph._nodes_by_name:
            graph_op = node.graph._nodes_by_name[graph_node_name]
            if graph_op.type == graph_op_type and len(graph_op.inputs) > 1 and v_output == graph_op.inputs[0].name:
                cur_i = graph_op.inputs[1].op
                if cur_i.type == 'Const':
                    val_type = cur_i.get_attr('dtype')
                    val_shape = [dim.size for dim in cur_i.get_attr('value').tensor_shape.dim]
                    if cur_i.get_attr('value').tensor_content != b'':
                        val_arr = np.frombuffer(cur_i.get_attr('value').tensor_content,
                                                val_type.as_numpy_dtype).reshape(*val_shape)
                    else:
                        val = cur_i.get_attr('value').float_val[0]
                        val_arr = np.full(tuple(val_shape), val)
                    node_input = [('_identity', _to_onnx_type(val_type), val_arr)]
                    get_assign_value = True
                    break
    else:
        sess = keras.backend.get_session()
        if node.type == 'VarHandleOp':
            val_arr = sess.run([node.name + "/Read/ReadVariableOp:0"])[0]
            graph_op = node.graph._nodes_by_name[node.name + "/Read/ReadVariableOp"]
        else:
            val_arr = sess.run([node.name + ":0"])[0]
            graph_op = node.graph._nodes_by_name[node.name]
        val_type = graph_op.get_attr('dtype')
        node_input = [('_identity', _to_onnx_type(val_type), val_arr)]
        get_assign_value = True

    if get_assign_value:
        oopb.add_node_with_output('Identity',
                                  node_input,
                                  operator.output_full_names,
                                  operator.outputs[0].full_name + '_identity')
    else:
        raise ValueError(var_handle_name + " op " + node.name + " is not properly processed")


@converter_func(TYPES.VarHandleOp)
def convert_tf_var_handle_op(scope, operator, container):
    _convert_tf_var_handle_helper(scope, operator, container, "VarHandleOp", "AssignVariableOp")


@converter_func(TYPES.VariableV2)
def convert_tf_variable_v2(scope, operator, container):
    _convert_tf_var_handle_helper(scope, operator, container, "VariableV2", "Assign")


@converter_func(TYPES.Where)
def convert_tf_where(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    where_node = oopb.add_node('NonZero',
                               operator.inputs[0].full_name,
                               operator.inputs[0].full_name + '_non_zero',
                               op_version=9)
    oopb.apply_op_with_output("apply_transpose",
                              where_node,
                              operator.output_full_names,
                              name=operator.full_name + '_transpose',
                              perm=list(reversed(range(len(node.outputs[0].shape)))))


@converter_func(TYPES.ZerosLike)
def convert_tf_zeros_like(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    dtype = _to_onnx_type(node.outputs[0].dtype)
    oopb.apply_op_with_output('apply_mul',
                              [operator.inputs[0].full_name,
                               ('_zero', dtype, np.zeros((), dtype=np.int64))],
                              operator.outputs[0].full_name,
                              name=operator.full_name)


@converter_func(TYPES.ReverseV2)
def convert_tf_reverse(scope, operator, container):
    node = operator.raw_operator
    oopb = OnnxOperatorBuilder(container, scope)
    ip = node.inputs[0]
    axes = _cal_tensor_value(node.inputs[1])

    len_axes = len(axes)
    if len_axes > 1:
        raise ValueError("Currently no support for more than 1 axis for ReverseV2 op")
    elif len_axes == 0:
        # Replace ReverseV2 with an identity block.
        oopb.apply_op_with_output('apply_identity', [ip], operator.outputs, name=node.name + '_Identity')
        return

    # Store input and output parameters of the ReverseV2 node.
    rv2_in_names = [node.inputs[0].name]

    input_shape = _cal_tensor_shape(ip)
    input_rank = len(input_shape)
    input_shape_node = oopb.add_node('Shape', ip.name, ip.name + '_shape')

    if input_shape is None:
        raise ValueError("ReverseV2 op {} requires the shape of input".format(node.name))

    rv2_node_name = node.name

    inputs = rv2_in_names

    # Supports only one axis as of now
    axis = axes[0]
    if axis < 0:
        axis = input_rank + axis

    batch_axis = 1 if axis != 1 else 0

    const_batch = numpy_helper.from_array(np.array([batch_axis], dtype=np.int64), rv2_node_name + '_const_batch')
    container.add_initializer_from_tensor(const_batch)
    const_axis = numpy_helper.from_array(np.array([axis], dtype=np.int64), rv2_node_name + '_const_axis')
    container.add_initializer_from_tensor(const_axis)

    batch_size = oopb.add_node('Gather', [input_shape_node, const_batch.name], rv2_node_name + '_gather_batch')
    axis_dim = oopb.add_node('Gather', [input_shape_node, const_axis.name], rv2_node_name + '_gather_axis')

    seq_array = oopb.add_node('Expand', [axis_dim, batch_size], rv2_node_name + '_expand')
    inputs.append(seq_array)

    res_seq_node = oopb.add_node('ReverseSequence', inputs, name=rv2_node_name + '_rev_seq', batch_axis=batch_axis,
                                 time_axis=axis)

    oopb.apply_op_with_output('apply_identity', [res_seq_node], [operator.outputs[0].full_name],
                              name=rv2_node_name + '_Identity')


direct_ops = {
    "Abs": ("apply_abs",),
    "Acos": 7,
    "Acosh": 9,
    "Add": ("apply_add",),
    "AddV2": ("apply_add",),
    "Asin": 7,
    "Asinh": 9,
    "Atan": 7,
    "Atanh": 9,
    "Ceil": ("apply_ceil",),
    "Cos": 7,
    "Cosh": 9,
    "Div": ("apply_div",),
    "Elu": ("apply_elu",),
    "Equal": 7,
    "Erf": 9,
    "Exp": ("apply_exp",),
    "Floor": ("apply_floor",),
    "Greater": ("apply_greater",),
    "Less": ("apply_less",),
    "Log": ("apply_log",),
    "Mul": ("apply_mul",),
    "Neg": ("apply_neg",),
    "RealDiv": ("apply_div",),
    "Reciprocal": ("apply_reciprocal",),
    "Relu": ("apply_relu",),
    "Sigmoid": ("apply_sigmoid",),
    "Sin": 7,
    "Sinh": 9,
    "Softplus": 1,
    "Softsign": 1,
    "Sqrt": ("apply_sqrt",),
    "StopGradient": ("apply_identity",),
    "Sub": ("apply_sub",),
    "Tan": 7,
    "Tanh": ("apply_tanh",)
}


def direct_tf_op_convert(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    type = operator.raw_operator.type
    item = direct_ops[type]
    assert item is not None, "Can't find the tf op item."
    if isinstance(item, numbers.Integral):
        oopb.add_node_with_output(type,
                                  [var_.full_name for var_ in operator.inputs],
                                  [var_.full_name for var_ in operator.outputs],
                                  name=operator.raw_operator.name,
                                  op_version=item
                                  )
    else:
        apply_func_name = item[0]
        oopb.apply_op_with_output(apply_func_name,
                                  [var_.full_name for var_ in operator.inputs],
                                  [var_.full_name for var_ in operator.outputs],
                                  name=operator.raw_operator.name,
                                  )


def register_direct_tf_ops():
    set_converters({k: direct_tf_op_convert for k in direct_ops.keys()})
