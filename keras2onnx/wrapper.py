###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import sys
import importlib
import numpy as np
from distutils.version import StrictVersion
from onnx import numpy_helper
from .common import k2o_logger
from .funcbook import set_converter
from .proto import onnx_proto, helper

try:
    tf2onnx = importlib.import_module('tf2onnx')
    process_tf_graph = tf2onnx.tfonnx.process_tf_graph
except (ImportError, ModuleNotFoundError) as e:
    tf2onnx = None
    k2o_logger().warning(
        "Can't import tf2onnx module, so the conversion on a model with any custom/lambda layer will fail!")


def process_begin_end(new_begin, new_end, stride):
    if stride >= 0:
        new_begin.append(0)
        new_end.append(sys.maxsize)
    else:
        new_begin.append(-1)
        new_end.append(-sys.maxsize)


def _prepare_StridedSlice(ctx, node, target_opset):
    max_size = sys.maxsize
    begin = node.inputs[1].get_tensor_value() if node.inputs[1].is_const() else [0] * node.inputs[1].output_shapes[0][0]
    end = node.inputs[2].get_tensor_value() if node.inputs[2].is_const() else [max_size] * \
                                                                              node.inputs[2].output_shapes[0][0]
    strides = node.inputs[3].get_tensor_value() if node.inputs[3].is_const() else [1] * node.inputs[3].output_shapes[0][
        0]
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
    extra_mask = new_axis_mask or shrink_axis_mask or ellipsis_mask
    new_begin = []
    new_end = []
    axes = []
    steps = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    ellipsis_gap = 0
    data_input_shape = node.inputs[0].output_shapes[0]
    for idx, begin_item in enumerate(begin):
        if target_opset < 10 and strides[idx] != 1:
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
        steps.append(strides[idx])

        if (begin_mask >> idx) & 1 != 0 and (end_mask >> idx) & 1 != 0:
            process_begin_end(new_begin, new_end, strides[idx])
            continue

        if begin_item == 0 and end_item == 0:
            process_begin_end(new_begin, new_end, strides[idx])
            continue

        shrink_mask = (shrink_axis_mask >> idx) & 1
        if shrink_mask != 0:
            shrink_begin = begin_item + data_input_shape[idx] if begin_item < 0 else begin_item
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


def on_StridedSlice(ctx, node, name, args):
    target_opset = 10
    new_begin, new_end, axes, steps, needs_squeeze, begin_mask, end_mask, extra_mask, new_axis_mask = _prepare_StridedSlice(
        ctx, node, target_opset)
    cast_node_begin = None
    if extra_mask or begin_mask:
        start_name = tf2onnx.utils.make_name(node.name)
        start_node = ctx.make_const(start_name, np.array(new_begin, dtype=np.int64))
        node.input[1] = start_node.output[0]
    else:
        cast_node_begin = ctx.insert_new_node_on_input(node, "Cast", node.input[1])
        cast_node_begin.set_attr("to", 7)

    cast_node_end = None
    if extra_mask or end_mask:
        end_name = tf2onnx.utils.make_name(node.name)
        end_node = ctx.make_const(end_name, np.array(new_end, dtype=np.int64))
        node.input[2] = end_node.output[0]
    else:
        cast_node_end = ctx.insert_new_node_on_input(node, "Cast", node.input[2])
        cast_node_end.set_attr("to", 7)

    axes_name = tf2onnx.utils.make_name(node.name)
    axes_node = ctx.make_const(axes_name, np.array(axes, dtype=np.int64))
    step_name = tf2onnx.utils.make_name(node.name)
    step_node = ctx.make_const(step_name, np.array(steps, dtype=np.int64))

    node.input[3] = axes_node.output[0]
    node.input.append(step_node.output[0])
    node.type = "Slice"
    nodes = [node]

    if cast_node_begin:
        nodes.append(cast_node_begin)
    if cast_node_end:
        nodes.append(cast_node_end)

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
    target_opset = 9
    new_begin, new_end, axes, steps, needs_squeeze, begin_mask, end_mask, extra_mask, new_axis_mask = _prepare_StridedSlice(
        ctx, node, target_opset)
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


def on_Round_10(ctx, node, name, args):
    const_name = tf2onnx.utils.make_name(node.name)
    const_node = ctx.make_const(const_name, (-0.5 * np.ones((), dtype=np.float32)))
    cast_name = tf2onnx.utils.make_name(node.name)
    cast_node = ctx.insert_new_node_on_output("Cast", const_node.output[0], cast_name)
    cast_node.set_attr("to", onnx_proto.TensorProto.FLOAT)
    ctx.set_dtype(cast_node.output[0], onnx_proto.TensorProto.FLOAT)
    add_output_name = tf2onnx.utils.make_name(node.name) + ':0'
    add_node = ctx.make_node("Add", [node.input[0], cast_node.output[0]], shapes=[node.output_shapes[0]],
                             dtypes=[node.output_dtypes], outputs=[add_output_name])
    node.input[0] = add_output_name
    node.type = "Ceil"
    return [const_node, add_node, node]


def on_Round(ctx, node, name, args):
    node.type = "Round"


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
        'Round': (on_Round_10 if opset <= 10 else on_Round, []),
        'StridedSlice': (on_StridedSlice_9 if opset <= 9 else on_StridedSlice, []),
        'TopKV2': (on_TopKV2, []),
        'All': (on_AllAny, []),
        'Any': (on_AllAny, []),
    }


def tf2onnx_wrap(topo, graph, outputs, target_opset):
    """
    A wrapper function to invoke the basic node conversion from tf2onnx
    """
    if tf2onnx is None:
        raise RuntimeError('tf2onnx loading failed with the tensorflow package!')
    custom_op_handlers = tf2onnx_builtin_conversion(target_opset)
    custom_op_handlers.update(topo.custom_op_dict)
    try:
        g = process_tf_graph(graph,
                             continue_on_error=topo.debug_mode,
                             opset=target_opset,
                             custom_op_handlers=custom_op_handlers,
                             output_names=outputs)
        return g

    except Exception as e:
        k2o_logger().warning("Exception on this tf.graph\n" +
                             '\n'.join(op_.name for op_ in graph.get_operations()))
        raise e


def update_container(varset, op, container):
    onnx_op = op.op
    op_inputs = [varset.get_local_variable_or_declare_one(n_).full_name.encode('utf-8') for n_ in onnx_op.input]
    op_outputs = [varset.get_local_variable_or_declare_one(n_).full_name.encode('utf-8') for n_ in onnx_op.output]
    onnx_op.name = varset.get_unique_operator_name(onnx_op.name).encode('utf-8')
    onnx_op.input[:] = op_inputs
    onnx_op.output[:] = op_outputs
    container.add_onnx_node(onnx_op, op_version=container.target_opset)


def tfnode_convert(varset, operator, container):
    """
    merge the output node from tf2onnx into the final graph.
    """
    g = operator.tf2onnx_graph
    g.delete_unused_nodes(g.outputs)
    g.topological_sort(g.get_nodes())
    g.update_proto()

    # update attributes
    all_inputs = set()
    if StrictVersion(tf2onnx.__version__) <= StrictVersion('0.3.2'):
        for op in g.get_nodes():
            all_inputs |= set(op.input)
            update_container(varset, op, container)

        # create input_tensor_values, initializers
        # if initilizer is not used as input by any node, then it will be ignored
        initializers = [i for i in list(g.initializers.values()) if i.name in all_inputs]
    else:
        # create initializers for constant nodes
        initializers = []
        for op in g.get_nodes():
            all_inputs |= set(op.input)
            if op.is_const():
                const_val = op.get_tensor_value(as_list=False)
                tensor = numpy_helper.from_array(const_val, op.output[0])
                initializers.append(tensor)
                continue
            elif op.is_graph_input():
                continue
            else:
                update_container(varset, op, container)

    for init_tensor_ in initializers:
        init_tensor_.name = varset.get_local_variable_or_declare_one(init_tensor_.name).full_name.encode('utf-8')
        container.add_initializer_from_tensor(init_tensor_)


TFNODES = 'TFNodes'
set_converter(TFNODES, tfnode_convert)
