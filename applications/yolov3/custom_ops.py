import sys
import numpy as np
from tf2onnx import utils

def on_StridedSlice(ctx, node, name, args):
    # node.type = "Reverse"
    # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
    not_supported_attr = ["new_axis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
    input_shape = ctx.get_shape(node.input[0])
    begin = node.inputs[1].get_tensor_value()
    end = node.inputs[2].get_tensor_value()
    strides = node.inputs[3].get_tensor_value()
    max_size = sys.maxsize
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask.i if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    ellipsis_mask = node.get_attr("ellipsis_mask")
    ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
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
        # if strides[idx] != 1:
        # raise ValueError("StridedSlice: only strides=1 is supported, current stride =" + str(strides[idx]))
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
    use_reverse_op = True
    reverse_flag = False
    if use_reverse_op and len(reverse_axes) > 0:
        name = utils.make_name(node.name)
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
        name = utils.make_name(node.name)
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
        name = utils.make_name(node.name)
        cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
        cast_node.set_attr("to", input_dtype)
        ctx.set_dtype(cast_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], cast_node.output[0])
        nodes.append(cast_node)
    '''
    return nodes


def on_Round(ctx, node, name, args):
    from onnx import onnx_pb
    const_name = utils.make_name(node.name)
    const_node = ctx.make_const(const_name, (-0.5 * np.ones((), dtype=np.float32)))
    cast_name = utils.make_name(node.name)
    cast_node = ctx.insert_new_node_on_output("Cast", const_node.output[0], cast_name)
    cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
    add_output_name = utils.make_name(node.name) + ':0'
    add_node = ctx.make_node("Add", [node.input[0], cast_node.output[0]], shapes=[node.output_shapes[0]], dtypes=[node.output_dtypes], outputs=[add_output_name])
    node.input[0] = add_output_name
    node.type = "Ceil"
    return [const_node, add_node, node]