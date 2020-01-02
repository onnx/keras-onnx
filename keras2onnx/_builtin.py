###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import sys
import numbers
import tensorflow
import numpy as np
from onnx import numpy_helper, mapping
from .common.onnx_ops import apply_identity, apply_reshape, OnnxOperatorBuilder
from .funcbook import converter_func, set_converters


class TYPES:
    # tf-node types:
    Identity = 'Identity'
    Const = 'Const'
    Any = 'Any'
    All = 'All'
    BiasAdd = 'BiasAdd'
    BiasAddV1 = 'BiasAddV1'
    Cast = 'Cast'
    ConcatV2 = 'ConcatV2'
    Conv1D = 'Conv1D'
    Conv2D = 'Conv2D'
    ExpandDims = 'ExpandDims'
    Fill = 'Fill'
    FusedBatchNorm = 'FusedBatchNorm'
    FusedBatchNormV2 = 'FusedBatchNormV2'
    FusedBatchNormV3 = 'FusedBatchNormV3'
    GatherV2 = 'GatherV2'
    Max = 'Max'
    Maximum = 'Maximum'
    Mean = 'Mean'
    Min = 'Min'
    Minimum = 'Minimum'
    NotEqual = 'NotEqual'
    Pack = 'Pack'
    Pad = 'Pad'
    PadV2 = 'PadV2'
    Prod = 'Prod'
    Range = 'Range'
    Reshape = 'Reshape'
    ResizeBilinear = 'ResizeBilinear'
    ResizeNearestNeighbor = 'ResizeNearestNeighbor'
    Round = 'Round'
    Rsqrt = 'Rsqrt'
    Shape = 'Shape'
    Size = 'Size'
    Squeeze = 'Squeeze'
    StridedSlice = 'StridedSlice'
    Sum = 'Sum'
    Tile = 'Tile'
    TopKV2 = 'TopKV2'
    Transpose = 'Transpose'
    Unpack = 'Unpack'

    # converter internal types:
    TD_Reshape = '_reshape_timedistributed'


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]


def _is_nhwc(node):
    return node.get_attr('data_format') == b'NHWC'


def _cal_tensor_value(tensor):  # type: (tensorflow.Tensor)->np.ndarray
    node = tensor.op
    if node.type in ['Placeholder']:
        return None
    elif node.type in ["Const", "ConstV2"]:
        make_ndarray = tensorflow.make_ndarray
        np_arr = make_ndarray(node.get_attr("value"))
        return np_arr
    else:
        try:
            cls_sess = tensorflow.Session if hasattr(tensorflow, 'Session') else tensorflow.compat.v1.Session
            with cls_sess(graph=node.graph) as sess:
                np_arr = sess.run(tensor)
                return np_arr
        except (ValueError, tensorflow.errors.InvalidArgumentError, tensorflow.errors.OpError):
            return None


def _cal_tensor_shape(tensor):
    if len(tensor.shape) > 0 and hasattr(tensor.shape[0], 'value'):
        return [x.value for x in tensor.shape]
    else:
        return list(tensor.shape)


def _to_onnx_type(dt_type):
    # TensorFlow data types integrate seamlessly with numpy
    return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dt_type.as_numpy_dtype)]


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


@converter_func(TYPES.Identity)
def convert_tf_identity(scope, operator, container):
    default_convert(scope, operator, container)


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

    axis_val = _cal_tensor_value(node.inputs[-1]).tolist()
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
    onnx_tensor = numpy_helper.from_array(np_arr, node.outputs[0].name)
    container.add_initializer_from_tensor(onnx_tensor)


def _spatial_map(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=False, new_kernel_shape=None,
                         output_indices=None):
    if output_indices is None:
        output_indices = [0]

    if _is_nhwc(node):
        # transpose input if needed, no need to record shapes on input
        transpose_node_1 = oopb.apply_transpose(node.inputs[0].name,
                                                name=operator.full_name + '_transpose_1',
                                                perm=NHWC_TO_NCHW)
    else:
        transpose_node_1 = [ node.inputs[0].name ]

    # kernel must to be transposed
    if with_kernel:
        val = _cal_tensor_value(node.inputs[1])
        if val is not None:
            val = val.transpose(HWCN_TO_NCHW)
            onnx_type = _to_onnx_type(node.inputs[1].dtype)
            transpose_node_kernel = oopb.apply_identity([('_start', onnx_type, val)],
                                                        name=operator.full_name + '_transpose_kernel')
        else:
            transpose_node_kernel = oopb.apply_transpose(node.inputs[1].name,
                                                         name=operator.full_name + '_transpose_kernel',
                                                         perm=HWCN_TO_NCHW)
        # TODO, some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
    else:
        transpose_node_kernel = [ node.inputs[1].name ]

    conv_node = oopb.apply_conv(transpose_node_1 + transpose_node_kernel,
                                name=operator.full_name + '_conv',
                                **attrs)

    # transpose outputs if needed
    if _is_nhwc(node):
        for idx in output_indices:
            oopb.add_node_with_output("Transpose",
                                      conv_node,
                                      operator.outputs[idx].full_name,
                                      name=operator.full_name + '_transpose_2_' + str(idx),
                                      perm=NCHW_TO_NHWC)
    else:
        for idx in output_indices:
            oopb.apply_op_with_output("apply_identity",
                                      conv_node,
                                      operator.outputs[idx].full_name,
                                      name=operator.full_name+ '_identity_' + str(idx))


def _conv_dims_attr(node, dims):
    if _is_nhwc(node):
        if len(dims) == 2:
            h, w = dims
        else:
            n, h, w, c = dims
    else:
        n, c, h, w = dims
    dims = [h, w]
    return dims


def _convert_tf_conv2d(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    kernel_shape = _cal_tensor_shape(node.inputs[1])[0:2]
    strides = _conv_dims_attr(node, node.get_attr('strides'))
    dilations = _conv_dims_attr(node, node.get_attr('dilations'))
    padding = node.get_attr('padding')
    spatial = 2
    attrs = {'strides': strides, 'dilations': dilations, 'kernel_shape': kernel_shape}
    if padding:
        if dilations is None:
            dilations = [1] * spatial * 2
        if padding == b'SAME':
            pads = [0] * spatial * 2
            input_shape = _cal_tensor_shape(node.inputs[0])
            output_shape = _cal_tensor_shape(node.outputs[0])
            # transpose shape to nchw
            if _is_nhwc(node):
                input_shape = _spatial_map(input_shape, NHWC_TO_NCHW)
                output_shape = _spatial_map(output_shape, NHWC_TO_NCHW)
            # calculate pads
            if any(input_shape[i + 2] == None or output_shape[i + 2] == None for i in range(spatial)):
                attrs["auto_pad"] = "SAME_UPPER"
            else:
                for i in range(spatial):
                    pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * kernel_shape[i] - input_shape[i + 2]
                    pad = max(pad, 0)
                    pads[i] = pad // 2
                    pads[i + spatial] = pad - pad // 2
                attrs["pads"] = pads

    _conv_convert_inputs(oopb, operator, node, attrs, with_kernel=True)


@converter_func(TYPES.Conv1D)
def convert_tf_conv1d(scope, operator, container):
    _convert_tf_conv2d(scope, operator, container)


@converter_func(TYPES.Conv2D)
def convert_tf_conv2d(scope, operator, container):
    _convert_tf_conv2d(scope, operator, container)


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
                              axis=axis,
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
        batch_norm = oopb.apply_batch_norm(transpose_node_1, name=operator.full_name + '_batch_norm', outputs_num=outputs_num, **attrs)
        output_perm = [0] + list(range(2, input_dim)) + [1]
        final_node = oopb.apply_transpose(batch_norm[0], name=operator.full_name + '_transpose_2',
                                          perm=output_perm)
    else:
        transpose_node_1 = []
        for idx in range(5):
            transpose_node_1.append(operator.inputs[idx].full_name)
        batch_norm = oopb.apply_batch_norm(transpose_node_1, name=operator.full_name + '_batch_norm', outputs_num=outputs_num, **attrs)
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
            cast_input_val = [ operator.inputs[1].full_name ]
        idx = 0
        for _ in range(fill_shape_dims):
            cast_input_val = oopb.apply_unsqueeze(cast_input_val,
                                                  name=operator.full_name + '_unsqueeze_' + str(idx),
                                                  axis=0)
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
            cast_input_dim = [ operator.inputs[0].full_name ]

        val = _cal_tensor_value(node.inputs[1])
        value = np.array([val])
        attrs = {"value": numpy_helper.from_array(value)}
        oopb.add_node_with_output('ConstantOfShape',
                                  cast_input_dim,
                                  operator.outputs[0].full_name,
                                  name=operator.full_name,
                                  **attrs)

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


def _convert_tf_maximum_minimum(scope, operator, container, oopb, apply_func):
    node = operator.raw_operator
    supported_types = [oopb.double, oopb.float, oopb.float16]
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
    delta_cast = delta.name
    if onnx_type in [oopb.int32, oopb.int64]:
        diff_output = oopb.apply_cast(diff_node,
                                      to=oopb.float,
                                      name=operator.full_name + '_cast_diff')
        delta_cast = oopb.apply_cast(delta.name,
                                     to=oopb.float,
                                     name=operator.full_name + '_cast_delta')

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
    target_shape = operator.get_attr('target_shape')
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.full_name, desired_shape=target_shape)


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
    paddings = np.array(_cal_tensor_value(node.inputs[1])).transpose().flatten()
    mode = node.get_attr("mode") if hasattr(node, 'mode') else None

    if mode:
        mode = mode.s.decode("utf-8").lower()
    if mode not in [None, "constant"]:
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
def convert_tf_min(scope, operator, container):
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
    attrs['coordinate_transformation_mode'] = 'asymmetric'
    if attrs['mode'] == 'nearest':
        attrs['nearest_mode'] = 'floor'
    if operator.target_opset < 10:
        op_type = 'Upsample'
    else:
        op_type = 'Resize'

    if operator.target_opset < 8:
        attrs = {"mode": mode, "scales": [1.0, 1.0, float(nh) / h, float(nw) / w]}
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
    cast_0 = oopb.add_node('Cast',
                           operator.inputs[0].full_name,
                           operator.inputs[0].full_name + '_0_cast', to=oopb.float)
    cast_1 = oopb.add_node('Cast',
                           operator.inputs[1].full_name,
                           operator.inputs[1].full_name + '_1_cast', to=oopb.int64)
    unsqueeze = oopb.add_node('Unsqueeze',
                              cast_1,
                              operator.inputs[1].full_name + '_unsqueeze', axes=[0])
    oopb.add_node_with_output("TopK",
                              [cast_0, unsqueeze],
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.Transpose)
def convert_tf_transpose(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    perm = _cal_tensor_value(node.inputs[1])
    input_value = _cal_tensor_value(node.inputs[0])
    if input_value is None:
        oopb.apply_op_with_output("apply_transpose",
                                  operator.inputs[0].full_name,
                                  operator.output_full_names,
                                  name=operator.full_name,
                                  perm=perm)
    else:
        output_value = np.transpose(input_value, perm)
        oopb.apply_op_with_output("apply_identity",
                                  [('_transpose_value', mapping.NP_TYPE_TO_TENSOR_TYPE[output_value.dtype], output_value)],
                                  operator.output_full_names,
                                  name=operator.full_name)


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
                                  operator.full_name + 'mask')
        oopb.add_node_with_output('Not', equal_out,
                                  operator.output_full_names,
                                  name=operator.full_name + '_not')
    else:
        equal_input_0 = oopb.add_node('Cast', [operator.inputs[0].full_name],
                                      operator.full_name + '_input_0_cast', to=6)
        equal_input_1 = oopb.add_node('Cast', [operator.inputs[1].full_name],
                                      operator.full_name + '_input_1_cast', to=6)
        equal_out = oopb.add_node('Equal', [equal_input_0, equal_input_1],
                                  operator.full_name + 'mask')
        oopb.add_node_with_output('Not', equal_out,
                                  operator.output_full_names,
                                  name=operator.full_name + '_not')


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
        end = [max_size] * node.inputs[2].shape[0]
    strides = _cal_tensor_value(node.inputs[3])
    if strides is None:
        strides = [1] * node.inputs[3].shape[0]
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask if end_mask is not None else 0
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
    data_input = node.inputs[0]
    for idx, begin_item in enumerate(begin):
        if target_opset < 10 and strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 are supported, current stride =" + str(strides[idx]))

        if (ellipsis_mask >> idx) & 1:
            input_shape = node.inputs[0].shape  # ctx.get_shape(node.input[0])
            if input_shape is None:
                raise ValueError("StridedSlice op {} requires the shape of input".format(node.name))
            ellipsis_gap = len(input_shape) - len(begin)
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
            shrink_begin = begin_item + _cal_tensor_shape(data_input)[idx] if begin_item < 0 else begin_item
            new_begin.append(shrink_begin)
            new_end.append(shrink_begin + 1)
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

    return new_begin, new_end, axes, steps, needs_squeeze, begin_mask, end_mask, extra_mask, new_axis_mask


@converter_func(TYPES.StridedSlice)
def convert_tf_strided_slice(scope, operator, container):
    node = operator.raw_operator
    new_begin, new_end, axes, steps, needs_squeeze, begin_mask, end_mask, extra_mask, new_axis_mask = _prepare_StridedSlice(
        node, operator.target_opset)
    oopb = OnnxOperatorBuilder(container, scope)

    new_axis_axes = []
    cur_idx = 0
    while new_axis_mask > 0:
        if new_axis_mask & 1:
            new_axis_axes.append(cur_idx)
        new_axis_mask = new_axis_mask >> 1
        cur_idx = cur_idx + 1

    if len(new_axis_axes) > 0:
        new_axis_unsqueeze = oopb.add_node('Unsqueeze',
                                           operator.inputs[0].full_name,
                                           operator.inputs[0].full_name + '_unsqueeze',
                                           axes=new_axis_axes)
    else:
        new_axis_unsqueeze = operator.inputs[0].full_name

    if operator.target_opset < 10:
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
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

        cropped_tensor_name = oopb.add_node('Slice',
                                            [new_axis_unsqueeze,
                                             ('_start', oopb.int64,
                                              np.array(new_begin, dtype=np.int64)) if cast_node_begin else start_cast,
                                             ('_end', oopb.int64,
                                              np.array(new_end, dtype=np.int64)) if cast_node_end else end_cast,
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
                                  axis=axis_val)


direct_ops = {"Abs": ("apply_abs",),
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
              "Exp": ("apply_exp",),
              "Floor": ("apply_floor",),
              "Log": ("apply_log",),
              "MatMul": ("apply_matmul",),
              "Mul": ("apply_mul",),
              "Neg": ("apply_neg",),
              "Pow": ("apply_pow",),
              "RealDiv": ("apply_div",),
              "Reciprocal": ("apply_reciprocal",),
              "Relu": ("apply_relu",),
              "Sigmoid": ("apply_sigmoid",),
              "Sin": 7,
              "Sinh": 9,
              "Softplus": 1,
              "Softsign": 1,
              "Softmax": ("apply_softmax", 1),
              "Sqrt": ("apply_sqrt",),
              "Sub": ("apply_sub",),
              "Tan": 7,
              "Tanh": ("apply_tanh",)
              }


def tf_op_convert(scope, operator, container):
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


set_converters({k: tf_op_convert for k in direct_ops.keys()})
