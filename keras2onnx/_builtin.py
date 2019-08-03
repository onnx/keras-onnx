###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import sys
import numpy as np
import tf2onnx

from .common.onnx_ops import apply_identity, apply_reshape
from .funcbook import set_converter
from .proto import onnx_proto, helper


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


def convert_reshape_timedistributed(scope, operator, container):
    target_shape = operator.get_attr('target_shape')
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.full_name, desired_shape=target_shape)


set_converter('identity', default_convert)
set_converter('reshape_timedistributed', convert_reshape_timedistributed)


def process_begin_end(new_begin, new_end, stride):
    if stride >= 0:
        new_begin.append(0)
        new_end.append(sys.maxsize)
    else:
        new_begin.append(-1)
        new_end.append(-sys.maxsize)


def on_StridedSlice(ctx, node, name, args):
    begin = node.inputs[1].get_tensor_value()
    end = node.inputs[2].get_tensor_value()
    strides = node.inputs[3].get_tensor_value()
    max_size = sys.maxsize
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask.i if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    new_axis_mask = node.get_attr("new_axis_mask")
    new_axis_mask = new_axis_mask.i if new_axis_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
    ellipsis_mask = node.get_attr("ellipsis_mask")
    ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
    new_begin = []
    new_end = []
    axes = []
    steps = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    ellipsis_gap = 0
    for idx, begin_item in enumerate(begin):
        if (ellipsis_mask >> idx) & 1:
            input_shape = ctx.get_shape(node.input[0])
            tf2onnx.utils.make_sure(
                input_shape is not None,
                "StridedSlice op {} requires the shape of input".format(node.name)
            )
            ellipsis_gap = len(input_shape) - len(begin)
            continue

        end_item = end[idx]
        axes.append(idx + ellipsis_gap)
        steps.append(strides[idx])

        if (begin_mask >> idx) & 1 != 0 and (end_mask >> idx) & 1 != 0:
            process_begin_end(new_begin, new_end, strides[idx])
            continue

        if begin_item == 0 and end_item == 0:
            process_begin_end(new_begin, new_end, strides[idx])
            continue

        mask = (shrink_axis_mask >> idx) & 1
        if mask != 0:
            new_begin.append(begin_item)
            new_end.append(end_item)
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

    start_name = tf2onnx.utils.make_name(node.name)
    start_node = ctx.make_const(start_name, np.array(new_begin, dtype=np.int64))
    end_name = tf2onnx.utils.make_name(node.name)
    end_node = ctx.make_const(end_name, np.array(new_end, dtype=np.int64))
    axes_name = tf2onnx.utils.make_name(node.name)
    axes_node = ctx.make_const(axes_name, np.array(axes, dtype=np.int64))
    step_name = tf2onnx.utils.make_name(node.name)
    step_node = ctx.make_const(step_name, np.array(steps, dtype=np.int64))

    node.input[1] = start_node.output[0]
    node.input[2] = end_node.output[0]
    node.input[3] = axes_node.output[0]
    node.input.append(step_node.output[0])
    node.type = "Slice"
    nodes = [node]

    new_axis_axes = []
    cur_idx = 0
    while new_axis_mask > 0:
        if new_axis_mask & 1:
            new_axis_axes.append(cur_idx)
        new_axis_mask = new_axis_mask >> 1
        cur_idx = cur_idx + 1

    if len(new_axis_axes) > 0:
        unsqueeze_node = ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[0])
        unsqueeze_node.set_attr("axes", new_axis_axes)
        nodes.append(unsqueeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(unsqueeze_node.output[0], input_dtype)

    if needs_squeeze:
        name = tf2onnx.utils.make_name(node.name)
        squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        squeeze_node.set_attr("axes", needs_squeeze)
        nodes.append(squeeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(squeeze_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], squeeze_node.output[0])

    return nodes


def on_StridedSlice_9(ctx, node, name, args):
    # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
    begin = node.inputs[1].get_tensor_value()
    end = node.inputs[2].get_tensor_value()
    strides = node.inputs[3].get_tensor_value()
    max_size = sys.maxsize
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask.i if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    new_axis_mask = node.get_attr("new_axis_mask")
    new_axis_mask = new_axis_mask.i if new_axis_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
    ellipsis_mask = node.get_attr("ellipsis_mask")
    ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
    new_begin = []
    new_end = []
    axes = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    ellipsis_gap = 0
    for idx, begin_item in enumerate(begin):
        if strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 are supported, current stride =" + str(strides[idx]))

        if (ellipsis_mask >> idx) & 1:
            input_shape = ctx.get_shape(node.input[0])
            tf2onnx.utils.make_sure(
                input_shape is not None,
                "StridedSlice op {} requires the shape of input".format(node.name)
            )
            ellipsis_gap = len(input_shape) - len(begin)
            continue

        end_item = end[idx]
        axes.append(idx + ellipsis_gap)
        if (begin_mask >> idx) & 1 != 0 and (end_mask >> idx) & 1 != 0:
            new_begin.append(0)
            new_end.append(max_size)
            continue

        if begin_item == 0 and end_item == 0:
            new_begin.append(0)
            new_end.append(max_size)
            continue

        # an implicit condition is stride == 1 (checked in above)
        if begin_item < 0 and end_item == 0:
            end_item = max_size

        mask = (shrink_axis_mask >> idx) & 1
        if mask != 0:
            new_begin.append(begin_item)
            new_end.append(end_item)
            needs_squeeze.append(idx + ellipsis_gap)
            continue

        if (begin_mask >> idx) & 1 != 0:
            new_begin.append(0)
            new_end.append(end_item)
            continue

        if (end_mask >> idx) & 1 != 0:
            new_begin.append(begin_item)
            new_end.append(max_size)
            continue

        new_begin.append(begin_item)
        new_end.append(end_item)

    node.set_attr("starts", new_begin)
    node.set_attr("ends", new_end)
    node.set_attr("axes", axes)
    node.type = "Slice"
    ctx.remove_input(node, node.input[3])
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    nodes = [node]

    new_axis_axes = []
    cur_idx = 0
    while new_axis_mask > 0:
        if new_axis_mask & 1:
            new_axis_axes.append(cur_idx)
        new_axis_mask = new_axis_mask >> 1
        cur_idx = cur_idx + 1

    if len(new_axis_axes) > 0:
        unsqueeze_node = ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[0])
        unsqueeze_node.set_attr("axes", new_axis_axes)
        nodes.append(unsqueeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(unsqueeze_node.output[0], input_dtype)

    if needs_squeeze:
        name = tf2onnx.utils.make_name(node.name)
        squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        squeeze_node.set_attr("axes", needs_squeeze)
        nodes.append(squeeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(squeeze_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], squeeze_node.output[0])

    return nodes


def on_Round(ctx, node, name, args):
    const_name = tf2onnx.utils.make_name(node.name)
    const_node = ctx.make_const(const_name, (-0.5 * np.ones((), dtype=np.float32)))
    cast_name = tf2onnx.utils.make_name(node.name)
    cast_node = ctx.insert_new_node_on_output("Cast", const_node.output[0], cast_name)
    cast_node.set_attr("to", onnx_proto.TensorProto.FLOAT)
    ctx.set_dtype(cast_node.output[0], onnx_proto.TensorProto.FLOAT)
    add_output_name = tf2onnx.utils.make_name(node.name) + ':0'
    add_node = ctx.make_node("Add", [node.input[0], cast_node.output[0]], shapes=[node.output_shapes[0]], dtypes=[node.output_dtypes], outputs=[add_output_name])
    node.input[0] = add_output_name
    node.type = "Ceil"
    return [const_node, add_node, node]


def on_TopKV2(ctx, node, name, args):
    # onnx only supports input K as a 1D tesor with dtype int64
    # while in tf, K is a 0D tensor with dtype int32
    k_0d = node.input[1]
    cast = ctx.make_node("Cast", [k_0d], attr={"to": onnx_proto.TensorProto.INT64})
    k_1d = ctx.make_node("Unsqueeze", cast.output, attr={"axes": [0]})
    ctx.replace_input(node, k_0d, k_1d.output[0])

    k_0 = node.input[0]
    cast_0 = ctx.make_node("Cast", [k_0], attr={"to": onnx_proto.TensorProto.FLOAT})
    ctx.replace_input(node, k_0, cast_0.output[0])
    node.type = "TopK"


def on_AllAny(ctx, node, name, args):
    # T output = All(T x, list(int) reduce_indices, @bool keepdims)
    # T output = Any(T x, list(int) reduce_indices, @bool keepdims)
    reduce_dim = node.inputs[1].get_tensor_value()

    # for Any, the reduce_indices can be scalar as observed.
    if np.isscalar(reduce_dim):
        reduce_dim = [reduce_dim]

    # It is fine to have nagative reduce_dim.
    cast = ctx.make_node(op_type="Cast", inputs=[node.input[0]], attr={"to": onnx_proto.TensorProto.FLOAT})
    keepdims = helper.get_attribute_value(node.get_attr("keep_dims"))
    op_type = "ReduceMin" if node.type == "All" else "ReduceSum"
    reduce_node = ctx.make_node(op_type=op_type, inputs=cast.output,
                                attr={"axes": reduce_dim, "keepdims": keepdims})

    zero_node = ctx.make_const(tf2onnx.utils.make_name("zero_reduce"), np.array(0, dtype=np.float32))

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(node.name)
    ctx.make_node(op_type="Greater", inputs=[reduce_node.output[0], zero_node.output[0]],
                  name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


def tf2onnx_builtin_conversion(opset):
    return {
        'Round': (on_Round, []),
        'StridedSlice': (on_StridedSlice_9 if opset <= 9 else on_StridedSlice, []),
        'TopKV2': (on_TopKV2, []),
        'All': (on_AllAny, []),
        'Any': (on_AllAny, []),
    }
