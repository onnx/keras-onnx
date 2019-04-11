###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .common.onnx_ops import apply_identity, apply_reshape
from .funcbook import set_converter
import sys
import numpy as np
import tf2onnx


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


def convert_reshape_timedistributed(scope, operator, container):
    iop = operator.raw_operator
    target_shape = iop.target_shape
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.raw_operator.name, desired_shape=target_shape)


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
    not_supported_attr = ["ellipsis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
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
    new_begin = []
    new_end = []
    axes = []
    steps = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    for idx, begin_item in enumerate(begin):
        end_item = end[idx]
        axes.append(idx)
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
            needs_squeeze.append(idx)
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

    # onnx slice as of opset 7 does only take float tensors ... cast if needed
    '''
    input_dtype = ctx.get_dtype(node.input[0])
    if input_dtype != onnx_pb.TensorProto.FLOAT:
        if node.inputs[0].type == "Cast":
            # override the previous cast
            cast_node = node.inputs[0]
        else:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
            nodes.insert(0, cast_node)
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.input[0], cast_node.output[0])
        # undo the cast afer slice
        name = tf2onnx.utils.make_name(node.name)
        cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
        cast_node.set_attr("to", input_dtype)
        ctx.set_dtype(cast_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], cast_node.output[0])
        nodes.append(cast_node)
    '''
    return nodes


# TODO, need clean up (remove reverse op) after opset 10 release
def on_StridedSlice_9(ctx, node, name, args):
    # for now we implement common cases. Things like strides!=1, -1 are not mappable to onnx.
    not_supported_attr = ["ellipsis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
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
    new_begin = []
    new_end = []
    axes = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    reverse_axes = []
    for idx, begin_item in enumerate(begin):
        end_item = end[idx]
        if strides[idx] == -1:
            reverse_axes.append(idx)
        if strides[idx] != 1 and strides[idx] != -1:
            raise ValueError("StridedSlice: only strides=1, -1 are supported, current stride =" + str(strides[idx]))
        axes.append(idx)

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
            needs_squeeze.append(idx)
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

    use_reverse_op = True
    reverse_flag = False
    if use_reverse_op and len(reverse_axes) > 0:
        name = tf2onnx.utils.make_name(node.name)
        name = name + '_reverse'
        reverse_node = ctx.insert_new_node_on_output("Reverse", node.output[0], name)
        reverse_node.set_attr("axes", reverse_axes)
        reverse_node.domain = 'com.microsoft'
        nodes.append(reverse_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(reverse_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], reverse_node.output[0])
        reverse_flag = True

    if needs_squeeze:
        name = tf2onnx.utils.make_name(node.name)
        if use_reverse_op:
            if reverse_flag:
                squeeze_node = ctx.insert_new_node_on_output("Squeeze", reverse_node.output[0], name)
            else:
                squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        else:
            squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        squeeze_node.set_attr("axes", needs_squeeze)
        nodes.append(squeeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(squeeze_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], squeeze_node.output[0])

    # onnx slice as of opset 7 does only take float tensors ... cast if needed
    '''
    input_dtype = ctx.get_dtype(node.input[0])
    if input_dtype != onnx_pb.TensorProto.FLOAT:
        if node.inputs[0].type == "Cast":
            # override the previous cast
            cast_node = node.inputs[0]
        else:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
            nodes.insert(0, cast_node)
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.input[0], cast_node.output[0])
        # undo the cast afer slice
        name = tf2onnx.utils.make_name(node.name)
        cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
        cast_node.set_attr("to", input_dtype)
        ctx.set_dtype(cast_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], cast_node.output[0])
        nodes.append(cast_node)
    '''
    return nodes


def on_Round(ctx, node, name, args):
    from onnx import onnx_pb
    const_name = tf2onnx.utils.make_name(node.name)
    const_node = ctx.make_const(const_name, (-0.5 * np.ones((), dtype=np.float32)))
    cast_name = tf2onnx.utils.make_name(node.name)
    cast_node = ctx.insert_new_node_on_output("Cast", const_node.output[0], cast_name)
    cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
    add_output_name = tf2onnx.utils.make_name(node.name) + ':0'
    add_node = ctx.make_node("Add", [node.input[0], cast_node.output[0]], shapes=[node.output_shapes[0]], dtypes=[node.output_dtypes], outputs=[add_output_name])
    node.input[0] = add_output_name
    node.type = "Ceil"
    return [const_node, add_node, node]
