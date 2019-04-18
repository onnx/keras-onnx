###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six
import tensorflow as tf
from six.moves import queue
from .proto import keras
from .common import k2o_logger
from .ke2onnx import extract_inbound_nodes, build_opdict_from_keras
from .common.data_types import Int32TensorType, Int64TensorType, FloatTensorType, DoubleTensorType, BooleanTensorType
from .topology import Topology
from .subgraph import is_placeholder_node, tsname_to_node, create_subgraph
from .funcbook import get_converter
from .wrapper import tf2onnx_wrap, TFNODES


DEFAULT_BATCH_SIZE = 1


def _infer_variable_type(tensor, default_batch_size=DEFAULT_BATCH_SIZE):
    if tensor.shape == tf.TensorShape(None):
        tensor_shape = []
    elif tensor.shape == tf.TensorShape([]):
        tensor_shape = []
    else:
        tensor_shape = [d.value if d.value is not None else 'None' for d in tensor.shape]
        # Adjust batch size if needed
        if tensor_shape[0] == 'None':
            tensor_shape[0] = default_batch_size

    # Determine the tensor's element type
    tensor_type = tensor.dtype
    if tensor_type in [tf.int8, tf.int16, tf.int32]:
        return Int32TensorType(shape=tensor_shape)
    elif tensor_type == tf.int64:
        return Int64TensorType(shape=tensor_shape)
    elif tensor_type in [tf.float16, tf.float32]:
        return FloatTensorType(shape=tensor_shape)
    elif tensor_type == tf.float64:
        return DoubleTensorType(shape=tensor_shape)
    elif tensor_type == tf.bool:
        return BooleanTensorType(shape=tensor_shape)
    else:
        raise ValueError('Unable to find out a correct type for tensor %s' % tensor)


def _find_node(nodes, name):
    try:
        opname = tsname_to_node(name)
        return next(n_ for n_ in nodes if n_.name == opname)
    except StopIteration:
        return None


def _locate_inputs_by_node(node_list, varset):
    inputs = {}
    for n_ in node_list:
        assert n_ in node_list

        for i_ in n_.inputs:
            op = i_.op
            if (not is_placeholder_node(op)) and op in node_list:
                continue

            if i_ not in inputs:
                v0 = varset.get_local_variable_or_declare_one(i_.name, _infer_variable_type(i_))
                inputs[i_] = v0

    return list(inputs.values()), list(inputs.keys())


def _locate_outputs(node_list, varset):
    var_output = []
    nodes = []
    for n_ in varset.variable_name_mapping.keys():
        node = _find_node(node_list, n_)
        if node is not None and (not is_placeholder_node(node)):
            nodes.append(node)

    assert nodes
    for n0_ in nodes:
        for n_ in n0_.outputs:
            var_output.append(varset.get_local_variable_or_declare_one(n_.name, _infer_variable_type(n_)))

    return var_output


def _is_relevant_keras_node(model, node):
    # type: (keras.Model, object) -> bool
    if not hasattr(model, '_nodes_by_depth'):
        return True  # 'Sequential' object has no attribute '_nodes_by_depth' in the legacy keras version.

    for v in model._nodes_by_depth.values():
        if node in v:
            return True
    return False


def _get_tensor_safe(graph, name):
    try:
        ts = graph.get_tensor_by_name(name)
    except KeyError:
        ts = None

    return ts


# This conversion supports timedistributed wrapper partially where the layer itself can be converted by onnx.
def _convert_keras_timedistributed(graph, node_list, layer, model, varset):
    inputs = []
    ishapes = []
    outputs = []
    oshapes = []
    num_relevant_keras_node = 0
    for nb_ in extract_inbound_nodes(layer):
        if _is_relevant_keras_node(model, nb_):
            inputs += nb_.input_tensors
            ishapes += nb_.input_shapes
            outputs += nb_.output_tensors
            oshapes += nb_.output_shapes
            num_relevant_keras_node = num_relevant_keras_node + 1

    assert num_relevant_keras_node == 1

    i_ = inputs[0]
    iname = i_.name
    k2o_logger().debug('input: ' + iname)
    i0 = varset.get_local_variable_or_declare_one(iname, _infer_variable_type(i_))
    i0_reshape_name = i_.op.name + '_reshape_0:0'
    i0_reshape = varset.declare_local_variable(i0_reshape_name, _infer_variable_type(i_))
    i0_reshape_shape = (-1,) + ishapes[0][2:]
    ishapes0 = [-1 if s_ is None else s_ for s_ in ishapes[0]]
    model_reshape_0 = keras.layers.Reshape(i0_reshape_shape, input_shape=ishapes0)
    operator_reshape_0 = varset.declare_local_operator('reshape_timedistributed', raw_model=model_reshape_0,
                                                       op_name=layer.name + '_reshape_0')
    operator_reshape_0.add_input(i0)
    operator_reshape_0.add_output(i0_reshape)

    o_ = outputs[0]
    oname = o_.name
    k2o_logger().debug('output: ' + oname)
    o1 = varset.get_local_variable_or_declare_one(oname, _infer_variable_type(o_))
    o1_reshape_shape = (-1,) + oshapes[0][2:]
    oshapes1 = [-1 if s_ is None else s_ for s_ in oshapes[0]]
    model_reshape_1 = keras.layers.Reshape(oshapes1, input_shape=oshapes1)
    operator_reshape_1 = varset.declare_local_operator('reshape_timedistributed', raw_model=model_reshape_1,
                                                       op_name=layer.name + '_reshape_1')
    operator_reshape_1.add_output(o1)
    o1_reshape_name = o_.op.name + '_reshape_1:0'
    o1_reshape = varset.declare_local_variable(o1_reshape_name, _infer_variable_type(o_))
    operator_reshape_1.add_input(o1_reshape)

    inner_layer = layer.layer
    setattr(inner_layer, '_input_shape', i0_reshape_shape)
    setattr(inner_layer, '_output_shape', o1_reshape_shape)
    setattr(layer, 'layer', inner_layer)

    operator = varset.declare_local_operator(type(layer.layer), raw_model=layer.layer, op_name=layer.name)
    operator.nodelist = node_list
    operator.add_input(i0_reshape)
    operator.add_output(o1_reshape)

    cvt = get_converter(type(layer.layer))
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator


def _convert_keras_scope(graph, node_list, layer, model, varset):
    operator = varset.declare_local_operator(type(layer), raw_model=layer, op_name=layer.name)
    operator.nodelist = node_list

    inputs = []
    outputs = []
    oshapes = []
    for nb_ in extract_inbound_nodes(layer):
        if _is_relevant_keras_node(model, nb_):
            inputs += nb_.input_tensors
            outputs += nb_.output_tensors
            oshapes += nb_.output_shapes

    for i_ in inputs:
        iname = i_.name
        k2o_logger().debug('input: ' + iname)
        i0 = varset.get_local_variable_or_declare_one(iname, _infer_variable_type(i_))
        operator.add_input(i0)

    for n_, o_ in enumerate(outputs):
        oname = o_.name
        k2o_logger().debug('output: ' + oname)
        o1 = varset.get_local_variable_or_declare_one(oname, _infer_variable_type(o_))
        o1.type.shape = ['None' if s_ is None else s_ for s_ in oshapes[n_]]
        operator.add_output(o1)

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator


def _convert_general_scope(node_list, varset):
    operator = varset.declare_local_operator(TFNODES, raw_model=node_list)
    operator.nodelist = node_list

    sess = keras.backend.get_session()
    subgraph, replacement = create_subgraph(sess.graph, node_list, sess, operator.full_name)
    setattr(operator, 'subgraph', subgraph)
    vars_, ts = _locate_inputs_by_node(node_list, varset)

    for n_, var_ in enumerate(vars_):
        ph_ = ts[n_]
        ph_name = replacement.get(ts[n_].op.name) + ':0'
        assert ph_ is not None

        # ph_.name -> identity -> ph_name -> ...
        oop = varset.declare_local_operator('identity')
        oop.add_input(var_)
        ov = varset.declare_local_variable(ph_name, _infer_variable_type(ph_))
        oop.add_output(ov)
        operator.add_input(ov)

    k2o_logger().debug("input: " + ','.join(operator.input_full_names))


def _finalize_tf2onnx_op(topo, operator, varset):
    subgraph = operator.subgraph  # type: tf.Graph
    node_list = operator.nodelist

    nodes = {}
    for n_ in varset.variable_name_mapping.keys():
        node = _find_node(node_list, n_)
        if node is not None and (not is_placeholder_node(node)):
            if node in nodes:
                nodes[node].append(n_)
            else:
                nodes[node] = [n_, ]

    outputs = []
    with subgraph.as_default():
        for n0_ in nodes:
            for i_, n_ in enumerate(n0_.outputs):
                idf_ = tf.identity(subgraph.get_tensor_by_name(operator.full_name + '/' + n_.name), operator.full_name + '_identity')
                outputs.append(idf_.name)
                iv = varset.get_local_variable_or_declare_one(idf_.name, _infer_variable_type(n_))
                ov = varset.get_local_variable_or_declare_one(nodes[n0_][i_], _infer_variable_type(n_))
                operator.add_output(iv)
                oop = varset.declare_local_operator('identity')
                oop.add_input(iv)
                oop.add_output(ov)

        # need more tests before this tensorflow graph optimization.
        # graph_def = tf_optimize({}, outputs, subgraph.as_graph_def(), True)
        # with tf.Graph().as_default() as sub_tf_graph:
        #     tf.import_graph_def(graph_def)
        #     g = tf2onnx_wrap(sub_tf_graph.get_operations(), outputs, varset.target_opset)
        #     setattr(operator, 'custom_op', g)
        g = tf2onnx_wrap(topo, subgraph, outputs, varset.target_opset)
        assert g
        operator.tf2onnx_graph = g

    return operator


def _finalize_const_graph(topology, top_level, varset):
    # this is const sub-graph list, which will be not traveled.
    const_iop = [op_ for op_ in varset.operators.values() if not op_.input_full_names]
    for op_ in const_iop:
        _finalize_tf2onnx_op(topology, op_, varset)


def _infer_graph_shape(topology, top_level, varset):
    raw_model_container = topology.raw_model
    var_queue = queue.Queue()
    for i_ in raw_model_container.input_names:
        var_queue.put_nowait(top_level.get_local_variable_or_declare_one(i_))

    visited = set()
    while not var_queue.empty():
        var = var_queue.get_nowait()
        k2o_logger().debug("var: " + var.full_name)
        for oop in var.op_to:
            if oop in visited:
                continue

            visited.add(oop)
            if oop.type == TFNODES:
                _finalize_tf2onnx_op(topology, oop, varset)
            else:
                if isinstance(oop.raw_operator, keras.layers.Layer):
                    assert oop.outputs
                elif oop.raw_operator:
                    oop.outputs = _locate_outputs(oop.raw_operator, varset)
                else:
                    assert oop.outputs
                for o_ in oop.outputs:
                    o_.op_from = oop

                si = oop.shape_infer
                if si is not None:
                    # let operator to build its own shape if it can't be deduced from the tf.graph.
                    si(oop)

            for o_ in oop.outputs:
                var_queue.put_nowait(o_)


def _create_link_node(var, top_level, varset, reversed_io=False):
    ty_ = _infer_variable_type(var)
    var0 = top_level.get_local_variable_or_declare_one(var.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(var.name, ty_)
    op = varset.declare_local_operator('identity')
    if reversed_io:
        var0, var1 = var1, var0
    op.add_input(var1)
    op.add_output(var0)
    return op


def _build_inference_nodeset(graph, outputs):
    nodes_to_keep = set()
    node_inputs = outputs[:]
    while node_inputs:
        nd_ = node_inputs[0]
        del node_inputs[0]
        if nd_ in nodes_to_keep:
            continue

        nodes_to_keep.add(nd_)
        node_inputs.extend(in_.op for in_ in nd_.inputs)

    return nodes_to_keep


def _create_keras_nodelist(layer, inference_nodeset, out_node=None):
    newly = set()
    ts_end = set()  # the input tensor set of the whole layer/model.
    for node_ in extract_inbound_nodes(layer):
        if out_node is not None and out_node.name not in\
                [tsname_to_node(ts_.name) for ts_ in node_.output_tensors]:
            continue  # this layer could be reused several times in the whole graph.
        if any(ts_.op not in inference_nodeset for ts_ in node_.output_tensors):
            continue
        newly |= set([ts_.op for ts_ in node_.output_tensors])
        ts_end |= set(node_.input_tensors)

    visited = set()
    while newly:
        visited |= newly
        newly.clear()
        for n_ in visited:
            for i_ in n_.inputs:
                if i_ in ts_end or i_.op in visited or i_.op not in inference_nodeset:
                    continue
                if isinstance(layer, keras.Model) and not i_.name.startswith(layer.name):
                    continue  # ugly fixing for the shared layer.
                newly.add(i_.op)

    return list(visited)


def _general_nodelist_closure(node, nodeset, keras_nodeset):
    nodes = set()
    visited = set()

    def is_stop_node(nd): return is_placeholder_node(nd) or nd in keras_nodeset

    node_added = [node]
    updated = True
    while updated:
        updated = False
        while node_added:
            nd_ = node_added[0]
            del node_added[0]
            if nd_ not in visited:
                visited.add(nd_)
                if not is_stop_node(nd_) and nd_ not in nodes:
                    nodes.add(nd_)
                    updated = True
                node_added.extend(in_.op for in_ in nd_.inputs if not is_stop_node(in_.op))

        node_added = []
        for nd_ in nodeset:
            if any(in_.op in nodes for in_ in nd_.inputs):
                node_added.append(nd_)

    return nodes


def _build_keras_nodeset(inference_nodeset, keras_node_dict):
    nodes = set()
    for layer_, _ in keras_node_dict.values():
        nodes.update(_create_keras_nodelist(layer_, inference_nodeset))
    return nodes


def _get_output_nodes(node_list, layer, node):
    if layer:
        for nd_ in extract_inbound_nodes(layer):
            name_set = set(tsname_to_node(ts_.name) for ts_ in nd_.output_tensors)
            if node.name in name_set:
                return set(n_ for n_ in node_list if n_.name in name_set)
    else:
        nodes_has_children = set()
        for node in node_list:
            if node:
                for input_tensor in node.inputs:
                    nodes_has_children.add(input_tensor.op)
        return set(node_list) - nodes_has_children


def _parse_graph_scope(graph, keras_node_dict, topology, top_scope, output_names):
    input_nodes = set()
    raw_model_container = topology.raw_model

    # build the node in the working scope.
    varset = topology.declare_scope('curr_', top_scope)
    for name in output_names:
        raw_model_container.add_output_name(name)

    model_outputs = []
    for name in output_names:
        var = graph.get_operation_by_name(tsname_to_node(name)).outputs[0]
        _create_link_node(var, top_scope, varset)
        model_outputs.append(var.op)

    # starting from the output node.
    q_overall = queue.Queue()
    for n_ in model_outputs:
        q_overall.put_nowait(n_)

    visited = set()  # since the output could be shared among the successor nodes.

    def advance_by_input(cur_node, layer_nodes, subgraph, inputs):
        for input_ in cur_node.inputs:
            predecessor = input_.op
            if is_placeholder_node(predecessor):
                input_nodes.add(predecessor)
            if predecessor in layer_nodes or len(layer_nodes) == 0:
                subgraph.put_nowait(predecessor)
            else:
                inputs.add(predecessor)
                q_overall.put_nowait(predecessor)

    inference_nodeset = _build_inference_nodeset(graph, model_outputs)
    keras_nodeset = _build_keras_nodeset(inference_nodeset, keras_node_dict)
    while not q_overall.empty():
        node = q_overall.get_nowait()
        if node in input_nodes or node in visited:
            continue

        type_k, model_ = (None, None)
        if node.name in keras_node_dict:
            type_k, model_ = keras_node_dict[node.name]
            activated_keras_nodes = _create_keras_nodelist(type_k, inference_nodeset, node)
        else:
            activated_keras_nodes = _general_nodelist_closure(node, inference_nodeset, keras_nodeset)
        q_subgraph = queue.Queue()
        i_subgraph = set()
        nodes = []
        for ot_ in (_get_output_nodes(activated_keras_nodes, type_k, node
                                      ) if activated_keras_nodes else [node]):
            if ot_ not in nodes:
                visited.add(ot_)
                nodes.append(ot_)
                advance_by_input(ot_, activated_keras_nodes, q_subgraph, i_subgraph)

        while not q_subgraph.empty():
            int_node = q_subgraph.get_nowait()
            if int_node in input_nodes or int_node in visited or int_node.name in keras_node_dict:
                continue

            visited.add(int_node)
            nodes.append(int_node)
            advance_by_input(int_node, activated_keras_nodes, q_subgraph, i_subgraph)

        k2o_logger().debug('Processed a keras layer - (%s: %s)' % (type_k.name, type(type_k)) if
                           type_k else (nodes[0].name, "Custom_Layer"))
        if isinstance(type_k, keras.layers.TimeDistributed):
            _convert_keras_timedistributed(graph, nodes, type_k, model_, varset)
        elif type_k is None or get_converter(type(type_k)) is None:
            _convert_general_scope(nodes, varset)
        else:
            _convert_keras_scope(graph, nodes, type_k, model_, varset)

    for nd_ in input_nodes:
        var = nd_.outputs[0]  # since it's placeholder node, safely claim there is only one output.
        raw_model_container.add_input_name(var.name)
        _create_link_node(var, top_scope, varset, True)

    _finalize_const_graph(topology, top_scope, varset)
    _infer_graph_shape(topology, top_scope, varset)
    topology.root_names = [variable.onnx_name for variable in top_scope.variables.values()]
    return topology


def parse_graph(topo, graph, target_opset, output_names):
    # type: (Topology, tf.Graph, int, []) -> Topology
    keras_op_table = None
    if topo.raw_model.model is not None:
        keras_op_table = \
            {tsname_to_node(nm_): x for (nm_, x) in
             six.iteritems(build_opdict_from_keras(topo.raw_model.model))}

    top_level = topo.declare_scope('__root')
    return _parse_graph_scope(graph, keras_op_table, topo, top_level, output_names)
