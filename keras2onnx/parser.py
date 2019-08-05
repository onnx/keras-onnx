###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six
import tensorflow as tf
from six.moves import queue
from collections.abc import Iterable
from .proto import keras
from .common import k2o_logger, get_default_batch_size
from .ke2onnx import extract_inbound_nodes, list_input_tensors, list_output_tensors, list_input_shapes, list_output_shapes, build_opdict_from_keras
from .common.data_types import Int32TensorType, Int64TensorType, FloatTensorType, DoubleTensorType, BooleanTensorType
from .topology import Topology
from .subgraph import is_placeholder_node, tsname_to_node, create_subgraph
from .funcbook import get_converter
from .wrapper import tf2onnx_wrap, TFNODES


def _infer_variable_type(tensor):
    if tensor.shape == tf.TensorShape(None):
        tensor_shape = []
    elif tensor.shape == tf.TensorShape([]):
        tensor_shape = []
    else:
        tensor_shape = [d.value for d in tensor.shape]

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


def _convert_keras_timedistributed(graph, node_list, layer, model, varset):
    """
        This conversion supports timedistributed wrapper partially where the layer itself can be converted by onnx.
    """
    inputs = []
    ishapes = []
    outputs = []
    oshapes = []
    num_relevant_keras_node = 0
    for nb_ in extract_inbound_nodes(layer):
        if _is_relevant_keras_node(model, nb_):
            inputs += list_input_tensors(nb_)
            ishapes += list_input_shapes(nb_)
            outputs += list_output_tensors(nb_)
            oshapes += list_output_shapes(nb_)
            num_relevant_keras_node = num_relevant_keras_node + 1

    assert num_relevant_keras_node == 1

    i_ = inputs[0]
    iname = i_.name
    k2o_logger().debug('td_layer input: ' + iname)
    i0 = varset.get_local_variable_or_declare_one(iname, _infer_variable_type(i_))
    i0_reshape_name = i_.op.name + '_reshape_0:0'
    i0_reshape = varset.declare_local_variable(i0_reshape_name, _infer_variable_type(i_))
    i0_reshape_shape = (-1,) + ishapes[0][2:]
    operator_reshape_0 = varset.declare_local_operator('reshape_timedistributed',
                                                       op_name=layer.name + '_reshape_0', target_shape=i0_reshape_shape)
    operator_reshape_0.add_input(i0)
    operator_reshape_0.add_output(i0_reshape)

    o_ = outputs[0]
    oname = o_.name
    k2o_logger().debug('td_layer output: ' + oname)
    o1 = varset.get_local_variable_or_declare_one(oname, _infer_variable_type(o_))
    o1_reshape_shape = (-1,) + oshapes[0][2:]
    oshapes1 = [-1 if s_ is None else s_ for s_ in oshapes[0]]
    operator_reshape_1 = varset.declare_local_operator('reshape_timedistributed',
                                                       op_name=layer.name + '_reshape_1', target_shape=oshapes1)
    operator_reshape_1.add_output(o1)
    o1_reshape_name = o_.op.name + '_reshape_1:0'
    o1_reshape = varset.declare_local_variable(o1_reshape_name, _infer_variable_type(o_))
    operator_reshape_1.add_input(o1_reshape)

    inner_layer = layer.layer
    setattr(inner_layer, '_input_shape', i0_reshape_shape)
    setattr(inner_layer, '_output_shape', o1_reshape_shape)
    setattr(layer, 'layer', inner_layer)

    if isinstance(layer.layer, keras.Model):
        kenode = extract_inbound_nodes(layer.layer)[0]
        intop = varset.declare_local_operator('identity')
        intop.add_input(i0_reshape)
        intop.add_output(varset.get_local_variable_or_declare_one(list_input_tensors(kenode)[0].name))
        _convert_keras_sub_model(layer.layer, graph, kenode, varset)
        intop = varset.declare_local_operator('identity')
        intop.add_input(varset.get_local_variable_or_declare_one(list_output_tensors(kenode)[0].name))
        intop.add_output(o1_reshape)
    else:
        operator = varset.declare_local_operator(type(layer.layer), raw_model=layer.layer, op_name=layer.name)
        operator.nodelist = node_list
        operator.add_input(i0_reshape)
        operator.add_output(o1_reshape)
        cvt = get_converter(type(layer.layer))
        if cvt is not None and hasattr(cvt, 'shape_infer'):
            operator.shape_infer = cvt.shape_infer


def _adjust_input_batch_size(var_type):
    if len(var_type.shape) > 0 and var_type.shape[0] is None:
        var_type.shape = [get_default_batch_size()] + var_type.shape[1:]
    return var_type


def _convert_keras_scope(graph, node_list, layer, model, varset, prefix=None):
    operator = varset.declare_local_operator(type(layer), raw_model=layer, op_name=layer.name)
    operator.nodelist = node_list

    inputs = []
    outputs = []
    oshapes = []
    for nb_ in extract_inbound_nodes(layer):
        if _is_relevant_keras_node(model, nb_):
            if not node_list or \
                (node_list and node_list[0] in [ts_.op for ts_ in list_output_tensors(nb_)]):
                    inputs += list_input_tensors(nb_)
                    outputs += list_output_tensors(nb_)
                    oshapes += list_output_shapes(nb_)
                    operator.inbound_node = nb_

    # This layer will be visited because its output is other layer's input
    # The output only need be in one of the layer inbound nodes
    assert len(node_list) == 0 or node_list[0] in [ts_.op for ts_ in outputs]

    if prefix is None:  # prefix is designed for the distinguish among the shared model instances.
        prefix = ''

    for i_ in inputs:
        iname = prefix + i_.name
        k2o_logger().debug('input : ' + iname)
        var_type = _adjust_input_batch_size(_infer_variable_type(i_))
        i0 = varset.get_local_variable_or_declare_one(iname, var_type)
        operator.add_input(i0)

    if hasattr(layer, 'input_mask') and layer.input_mask is not None:
        in_mask = layer.input_mask if isinstance(layer.input_mask, Iterable) else [layer.input_mask]
        for im_ in [m_ for m_ in in_mask if m_]:
            mts_name = im_.name  # input mask in a shared model is not supported yet, who need it?
            k2o_logger().debug('input mask: ' + mts_name)
            mts_var = varset.get_local_variable_or_declare_one(mts_name, _infer_variable_type(im_))
            operator.add_input_mask(mts_var)

    for n_, o_ in enumerate(outputs):
        oname = prefix + o_.name
        k2o_logger().debug('output: ' + oname)
        o1 = varset.get_local_variable_or_declare_one(oname, _infer_variable_type(o_))
        o1.type.shape = oshapes[n_]
        operator.add_output(o1)

    if hasattr(layer, 'output_mask') and layer.output_mask is not None:
        out_mask = layer.output_mask if isinstance(layer.output_mask, Iterable) else [layer.output_mask]
        for om_ in [m_ for m_ in out_mask if m_]:
            mts_name = prefix + om_.name
            k2o_logger().debug('output mask: ' + mts_name)
            mts_var = varset.get_local_variable_or_declare_one(mts_name, _infer_variable_type(om_))
            operator.add_output_mask(mts_var)

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator


def _check_layer_converter_availability(sub_model):
    for l_ in sub_model.layers:
        if isinstance(l_, keras.Model):
            exist = _check_layer_converter_availability(l_)
        else:
            layer_type = type(l_)
            exist = layer_type is keras.layers.InputLayer or get_converter(layer_type)
        if not exist:
            break
    else:
        return True

    return False


def _create_model_input_mapping_operators(ts_from, ts_to, prefix, varset):
    ty_ = _infer_variable_type(ts_from)
    assert type(_infer_variable_type(ts_to)) is type(ty_)
    var0 = varset.get_local_variable_or_declare_one(ts_from.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(prefix + ts_to.name, ty_)
    op = varset.declare_local_operator('identity', op_name=prefix + ts_to.name)
    op.add_input(var0)
    op.add_output(var1)
    k2o_logger().debug("mapping:  %s -> %s" % (ts_from.name, ts_to.name))
    return op


def _find_kenode_by_output_tensor(inbound_nodes, output_name):
    def find_ts_name(tensors, name): return next((ts_ for ts_ in tensors if ts_.name.find(name) == 0), None)
    return next((n_ for n_ in inbound_nodes if find_ts_name(list_output_tensors(n_), output_name) is not None), None)


def _convert_keras_sub_model(sub_model, graph, target_kenode, varset, top_kenode=None, upper_prefix=None):
    ts_inputs = []
    ts_outputs = []
    upper_prefix = upper_prefix if upper_prefix else ''
    prefix = ''
    # mapping input/output nodes for the sub_model.
    inbound_nodes = extract_inbound_nodes(sub_model)
    if len(inbound_nodes) > 1 and inbound_nodes[0] is not target_kenode:
        # Assumption: the first node in the inbound node list is always the one used in the keras layers.
        base_node = inbound_nodes[0]
        curr_node = target_kenode
        assert curr_node is not None
        for idx_, out_ in enumerate(list_output_tensors(curr_node)):
            base_ts = list_output_tensors(base_node)[idx_]
            if not prefix:
                prefix = out_.name[0:out_.name.find(base_ts.name)]
            else:
                assert prefix == out_.name[0:out_.name.find(base_ts.name)]
            ts_outputs.append(out_)
        if top_kenode is None:
            top_kenode = curr_node

        # the input node needs to be mapped to the outmost inbound keras node.
        for idx_, in_ in enumerate(list_input_tensors(top_kenode)):
            _create_model_input_mapping_operators(in_, list_input_tensors(base_node)[idx_], upper_prefix+prefix, varset)
            ts_inputs.append(in_)

    k2o_logger().debug("prefix : %s" % prefix)
    for nodes_ in sub_model._nodes_by_depth.values():
        for n_ in nodes_:
            layer = n_.outbound_layer
            if isinstance(layer, keras.layers.InputLayer):
                continue
            elif isinstance(layer, keras.Model):
                k2o_logger().debug("Processing a keras sub model - %s" % layer.name)
                _convert_keras_sub_model(layer, graph, n_, varset, top_kenode, upper_prefix + prefix)
            else:
                _convert_keras_scope(graph, [], layer, sub_model, varset, upper_prefix+prefix)

    k2o_logger().debug("end prefix - %s" % prefix)
    return ts_inputs, ts_outputs


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


def _create_link_node(var_ts, top_level, varset, reversed_io=False):
    ty_ = _infer_variable_type(var_ts)
    var0 = top_level.get_local_variable_or_declare_one(var_ts.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(var_ts.name, ty_)
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
                [tsname_to_node(ts_.name) for ts_ in list_output_tensors(node_)]:
            continue  # this layer could be reused several times in the whole graph.
        if any(ts_.op not in inference_nodeset for ts_ in list_output_tensors(node_)):
            continue
        newly |= set([ts_.op for ts_ in list_output_tensors(node_)])
        ts_end |= set(list_input_tensors(node_))

    visited = set()
    while newly:
        visited |= newly
        newly.clear()
        for n_ in visited:
            for i_ in n_.inputs:
                if i_ in ts_end or i_.op in visited or i_.op not in inference_nodeset:
                    continue
                if isinstance(layer, keras.Model):  # ugly fixing for the shared layer.
                    if i_.name.startswith(layer.name):
                        pass
                    elif i_.name.startswith('^' + layer.name):
                        pass
                    else:
                        continue

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
            name_set = set(tsname_to_node(ts_.name) for ts_ in list_output_tensors(nd_))
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
    """
    travel the tensor Graph and build the corresponding intermediate operation objects.
    :param graph: the tensorflow session graph of the Keras mode.
    :param keras_node_dict: the mapping of operation node to keras layer output.
    :param topology: The whole topology of the intermediate objects.
    :param top_scope: The top varset
    :param output_names: the output names of the TF graph
    :return: The whole topology of the intermediate objects.
    """
    input_nodes = set()
    raw_model_container = topology.raw_model

    # build the node in the working scope.
    varset = topology.declare_scope('curr_', top_scope)
    for name in output_names:
        raw_model_container.add_output_name(name)

    model_outputs = []
    for name in output_names:
        var_ts = graph.get_operation_by_name(tsname_to_node(name)).outputs[0]
        _create_link_node(var_ts, top_scope, varset)
        model_outputs.append(var_ts.op)

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

        layer_key_, model_ = (None, None)
        if node.name in keras_node_dict:
            layer_key_, model_ = keras_node_dict[node.name]
            if isinstance(layer_key_, keras.Model) and \
                    _check_layer_converter_availability(layer_key_):
                k2o_logger().debug("Processing a keras sub model - %s" % layer_key_.name)
                kenode = _find_kenode_by_output_tensor(extract_inbound_nodes(layer_key_), node.name)
                ts_in, ts_out = _convert_keras_sub_model(layer_key_, graph, kenode, varset)
                for ts_ in ts_in:
                    if is_placeholder_node(ts_.op):
                        input_nodes.add(ts_.op)
                    else:
                        q_overall.put_nowait(ts_.op)

                visited.update(ts_.op for ts_ in ts_out)
                continue

            activated_keras_nodes = _create_keras_nodelist(layer_key_, inference_nodeset, node)
        else:
            activated_keras_nodes = _general_nodelist_closure(node, inference_nodeset, keras_nodeset)
        q_subgraph = queue.Queue()
        i_subgraph = set()
        nodes = []
        for ot_ in (_get_output_nodes(activated_keras_nodes, layer_key_, node
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

        k2o_logger().debug('Processing a keras layer - (%s: %s)' % (layer_key_.name, type(layer_key_)) if
                           layer_key_ else (nodes[0].name, "Custom_Layer"))
        if isinstance(layer_key_, keras.layers.TimeDistributed):
            _convert_keras_timedistributed(graph, nodes, layer_key_, model_, varset)
        elif layer_key_ is None or get_converter(type(layer_key_)) is None:
            _convert_general_scope(nodes, varset)
        else:
            _convert_keras_scope(graph, nodes, layer_key_, model_, varset)

    for nd_ in input_nodes:
        var_ts = nd_.outputs[0]  # since it's placeholder node, safely claim there is only one output.
        _create_link_node(var_ts, top_scope, varset, True)

    _finalize_const_graph(topology, top_scope, varset)
    _infer_graph_shape(topology, top_scope, varset)
    topology.root_names = [variable.onnx_name for variable in top_scope.variables.values()]
    return topology


def parse_graph(topo, graph, target_opset, output_names):
    # type: (Topology, tf.Graph, int, []) -> Topology
    """
    Build the node-layer mapper and parse the whole TF graph of Keras Model.
    """
    keras_layer_ts_map = {}
    if topo.raw_model.model is not None:
        keras_layer_ts_map = \
            {tsname_to_node(nm_): x for (nm_, x) in
             six.iteritems(build_opdict_from_keras(topo.raw_model.model))}

    top_level = topo.declare_scope('__root')

    # Create the onnx model input name before parsing to keep
    # the model input names are identical to the original Keras model.
    for idx_, ts_ in enumerate(topo.raw_model.model.inputs):
        op = top_level.declare_local_operator('identity')
        input_ts = topo.raw_model.model.inputs[idx_]
        var_type = _adjust_input_batch_size(_infer_variable_type(input_ts))
        str_value = input_ts.name
        var0 = None
        if hasattr(topo.raw_model.model, 'input_names'):
            str_value = topo.raw_model.model.input_names[idx_]
        elif topo.raw_model.model.inputs[idx_].name.endswith(':0'):
            str_value = topo.raw_model.model.inputs[idx_].name[:-2]
        else:
            # if there is no difference between input tensor name and model input name
            # skip it.
            var0 = top_level.get_local_variable_or_declare_one(str_value, var_type)
        if not var0:
            var0 = top_level.get_local_variable_or_declare_one(str_value, var_type)
            var1 = top_level.get_local_variable_or_declare_one(topo.raw_model.model.inputs[idx_].name, var_type)
            op.add_input(var0)
            op.add_output(var1)
        topo.raw_model.add_input_name(str_value)

    return _parse_graph_scope(graph, keras_layer_ts_map, topo, top_level, output_names)
