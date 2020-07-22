###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import re
import queue

from .proto import keras, is_tf_keras
from .proto.tfcompat import tensorflow as tf
from .proto.tfcompat import is_tf2
from .common import k2o_logger
from .topology import Topology
from .funcbook import get_converter, set_converter
from ._consts import TYPES
from ._tf_ops import pass_thru_converter
from ._parser_tf import (infer_variable_type, LayerInfo, is_placeholder_node,
                         tsname_to_node, on_parsing_keras_layer_v2, adjust_input_batch_size as _adjust_input_batch_size,
                         adjust_input_output_size as _adjust_input_output_size)
from ._parser_1x import (extract_inbound_nodes,
                         list_input_tensors, list_input_mask, list_output_mask,
                         list_output_tensors, list_input_shapes, list_output_shapes, on_parsing_keras_layer)


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
            if op.name[0] == '^':
                continue
            if (not is_placeholder_node(op)) and op in node_list:
                continue

            if i_ not in inputs:
                v0 = varset.get_local_variable_or_declare_one(i_.name, infer_variable_type(i_, varset.target_opset))
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
            var_output.append(
                varset.get_local_variable_or_declare_one(n_.name, infer_variable_type(n_, varset.target_opset)))

    return var_output


def _is_relevant_keras_node(model, node):
    # type: (keras.Model, object) -> bool
    if not hasattr(model, '_nodes_by_depth'):
        return True  # 'Sequential' object has no attribute '_nodes_by_depth' in the legacy keras version.

    for v in model._nodes_by_depth.values():
        if node in v:
            return True
    return False


def _on_parsing_time_distributed_layer(graph, node_list, layer, model, varset, prefix=None):
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

    prefix = prefix or ''
    i_ = inputs[0]
    iname = prefix + i_.name
    k2o_logger().debug('td_layer input: ' + iname)
    i0 = varset.get_local_variable_or_declare_one(iname, infer_variable_type(i_, varset.target_opset))
    o1_reshape_shape = (-1,) + oshapes[0][2:]
    i0_reshape_name = i_.op.name + '_reshape_0:0'
    i0_reshape = varset.declare_local_variable(i0_reshape_name, infer_variable_type(i_, varset.target_opset))
    i0_reshape_shape = (-1,) + ishapes[0][2:]
    i0_reshape.type.shape = i0_reshape_shape
    operator_reshape_0 = varset.declare_local_operator(TYPES.TD_Reshape,
                                                       op_name=layer.name + '_reshape_0', target_shape=i0_reshape_shape)
    operator_reshape_0.add_input(i0)
    operator_reshape_0.add_output(i0_reshape)

    o_ = outputs[0]
    oname = prefix + o_.name
    k2o_logger().debug('td_layer output: ' + oname)
    o1 = varset.get_local_variable_or_declare_one(oname, infer_variable_type(o_, varset.target_opset))
    oshapes1 = [-1 if s_ is None else s_ for s_ in oshapes[0]]
    operator_reshape_1 = varset.declare_local_operator(TYPES.TD_Reshape,
                                                       op_name=layer.name + '_reshape_1', target_shape=oshapes1)
    operator_reshape_1.add_output(o1)
    o1_reshape_name = o_.op.name + '_reshape_1:0'
    o1_reshape = varset.declare_local_variable(o1_reshape_name, infer_variable_type(o_, varset.target_opset))
    o1_reshape.type.shape = o1_reshape_shape
    operator_reshape_1.add_input(o1_reshape)

    if isinstance(layer.layer, keras.Model):
        kenode = extract_inbound_nodes(layer.layer)[0]
        intop = varset.declare_local_operator(TYPES.Identity)
        intop.add_input(i0_reshape)
        intop.add_output(varset.get_local_variable_or_declare_one(list_input_tensors(kenode)[0].name))
        _on_parsing_model_layer(layer.layer, graph, kenode, varset)
        intop = varset.declare_local_operator(TYPES.Identity)
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


def _check_layer_converter_availability(sub_model):
    for l_ in sub_model.layers:
        if isinstance(l_, keras.Model):
            exist = _check_layer_converter_availability(l_)
        else:
            layer_type = type(l_)
            exist = get_converter(layer_type) or \
                layer_type in [keras.layers.InputLayer, keras.layers.wrappers.TimeDistributed]

        if not exist:
            k2o_logger().info("The layer {} doesn't have a specific converter, fall back.".format(str(l_)))
            break
    else:
        return True

    return False


def _create_identity(ts_from, ts_to, varset):
    ty_ = infer_variable_type(ts_from, varset.target_opset)
    var0 = varset.get_local_variable_or_declare_one(ts_from.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(ts_to.name, ty_)
    op = varset.declare_local_operator(TYPES.Identity, op_name=ts_to.name)
    op.add_input(var0)
    op.add_output(var1)
    return op


def _create_model_input_mapping_operators(ts_from, ts_to, prefix, subprefix, varset):
    ty_ = infer_variable_type(ts_from, varset.target_opset)
    # type(_infer_variable_type(ts_to, varset.target_opset) and ...
    # ... type(ty_) can be different which is resolved by implicit cast.
    var0 = varset.get_local_variable_or_declare_one(subprefix + ts_from.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(prefix + ts_to.name, ty_)
    op = varset.declare_local_operator(TYPES.Identity, op_name=prefix + ts_to.name)
    op.add_input(var0)
    op.add_output(var1)
    k2o_logger().debug(
        "mapping:  %s -> %s (%s -> %s)" % (ts_from.name, ts_to.name, subprefix + ts_from.name, prefix + ts_to.name))
    return op


def _find_kenode_by_output_tensor(inbound_nodes, output_name):
    def find_ts_name(tensors, name):
        return next((ts_ for ts_ in tensors if ts_.name.find(name) == 0), None)

    return next((n_ for n_ in inbound_nodes if find_ts_name(list_output_tensors(n_), output_name) is not None), None)


def _is_template_tensors(tensors, templ_tensors):
    for t_, tt_ in zip(tensors, templ_tensors):
        # t_.shape and tt_.shape can be different if the input shape is different.
        if t_.name.find(tt_.name) < 0:
            return False

    return True


def _on_parsing_model_layer(sub_model, graph, target_kenode, varset, top_kenode=None, upper_prefix=None):
    ts_inputs = []
    ts_outputs = []
    upper_prefix = upper_prefix if upper_prefix else ''
    prefix = ''
    # mapping input/output nodes for the sub_model.
    inbound_nodes = extract_inbound_nodes(sub_model)

    sub_model_node_idx = 0
    if len(inbound_nodes) > 1 and inbound_nodes[0] is not target_kenode:
        # Assumption: the first node in the inbound node list is always the one used in the keras layers.
        curr_node = target_kenode
        assert curr_node is not None
        found = False
        base_node = inbound_nodes[0]
        for nodes_ in sub_model._nodes_by_depth.values():
            for nd_ in nodes_:
                if _is_template_tensors(list_output_tensors(curr_node), list_output_tensors(nd_)):
                    found = True
                    base_node = nd_
                    break
            else:
                sub_model_node_idx += 1
            if found:
                break
        else:
            assert False, "Cannot find the node for the model layer {}".format(sub_model.name)

        bn_name_list = [bn_.name for bn_ in list_output_tensors(base_node)]
        prefix_found = False
        for idx_, out_ in enumerate(list_output_tensors(curr_node)):
            if not prefix_found:
                name_match_len = -1
                for bn_name_ in bn_name_list:
                    cur_match_len = out_.name.find(bn_name_)
                    if cur_match_len > -1:
                        name_match_len = cur_match_len
                        break
                assert name_match_len > 0
                prefix = out_.name[0:name_match_len]
                prefix_found = True
            ts_outputs.append(out_)

        if top_kenode is None:
            top_kenode = curr_node

        # the input node needs to be mapped to the outmost inbound keras node.
        for idx_, in_ in enumerate(list_input_tensors(top_kenode)):
            _create_model_input_mapping_operators(in_, list_input_tensors(inbound_nodes[0])[idx_],
                                                  upper_prefix + prefix, upper_prefix,
                                                  varset)
            ts_inputs.append(in_)

    k2o_logger().debug("prefix_beg: %s" % prefix)
    for i_ in range(sub_model_node_idx, len(sub_model._nodes_by_depth)):
        nodes_ = sub_model._nodes_by_depth[i_]
        for n_ in nodes_:
            layer = n_.outbound_layer
            if isinstance(layer, keras.layers.InputLayer):
                continue
            elif isinstance(layer, keras.layers.wrappers.TimeDistributed):
                _on_parsing_time_distributed_layer(graph, [], layer, sub_model, varset, upper_prefix + prefix)
            elif isinstance(layer, keras.Model):
                k2o_logger().debug("Processing a keras sub model - %s" % layer.name)
                cur_kenode = _find_kenode_by_output_tensor(extract_inbound_nodes(layer), sub_model.outputs[0].name)
                _on_parsing_model_layer(layer, graph, n_, varset, cur_kenode, upper_prefix + prefix)
            else:
                on_parsing_keras_layer(graph, [], layer, n_, sub_model, varset, upper_prefix + prefix)

    k2o_logger().debug("prefix_end: - %s" % prefix)
    return ts_inputs, ts_outputs


def _check_tfnode_converter_availability(graph, node):
    var_assign_map = {'VarHandleOp': 'AssignVariableOp', 'VariableV2': 'Assign'}
    if node.type in var_assign_map:
        if is_tf2:
            v_output = node.outputs[0].name
            for graph_node_name in graph._nodes_by_name:
                graph_op = graph._nodes_by_name[graph_node_name]
                if graph_op.type == var_assign_map[node.type] and len(graph_op.inputs) > 1 and v_output == \
                        graph_op.inputs[0].name:
                    cur_i = graph_op.inputs[1].op
                    if cur_i.type == 'Const' and cur_i.get_attr('value').tensor_content != b'':
                        return True
            return False
        else:
            return True
    else:
        cvt = get_converter(node.type)
        return cvt is not None


def _check_tfnodes_converter_availability(graph, nodelist, debug_mode):
    status = True
    for n_ in nodelist:
        if not _check_tfnode_converter_availability(graph, n_):
            k2o_logger().warning(
                "WARN: No corresponding ONNX op matches the tf.op node {} of type {}".format(n_.name, n_.type) +
                "\n      The generated ONNX model needs run with the custom op supports.")
            status = False

    return status


def _on_parsing_tf_nodes(graph, nodelist, varset, debug_mode):
    _check_tfnodes_converter_availability(graph, nodelist, debug_mode)
    for node_ in nodelist:
        k2o_logger().debug("Processing a tf node - %s" % node_.name)
        operator = varset.declare_local_operator(node_.type, raw_model=node_, op_name=node_.name)

        for o_ in node_.outputs:
            oname = o_.name
            k2o_logger().debug('\toutput: ' + oname)
            out0 = varset.get_local_variable_or_declare_one(oname, infer_variable_type(o_, varset.target_opset))
            operator.add_output(out0)

        for i_ in node_.inputs:
            k2o_logger().debug('\tinput : ' + i_.name)
            var_type = infer_variable_type(i_, varset.target_opset)
            i0 = varset.get_local_variable_or_declare_one(i_.name, var_type)
            operator.add_input(i0)

        cvt = get_converter(operator.type)
        if cvt is None:
            assert isinstance(operator.type, str), \
                "Only tf-op can be pass_thru conversion, type: {}".format(type(operator.type))
            set_converter(operator.type, pass_thru_converter)
        elif hasattr(cvt, 'shape_infer'):
            operator.shape_infer = cvt.shape_infer


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
            if isinstance(oop.raw_operator, (keras.layers.Layer, tf.Operation)):
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


def _create_link_node(var_ts, top_level, varset, reversed_io=False, adjust_batch_size=False):
    if adjust_batch_size:
        ty_ = _adjust_input_batch_size(infer_variable_type(var_ts, varset.target_opset))
    else:
        ty_ = infer_variable_type(var_ts, varset.target_opset)
    var0 = top_level.get_local_variable_or_declare_one(var_ts.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(var_ts.name, ty_)
    op = varset.declare_local_operator(TYPES.Identity)
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
    newly = list()
    ts_end = set()  # the input tensor set of the whole layer/model.
    for node_ in extract_inbound_nodes(layer):
        if out_node is not None and out_node.name not in \
                [tsname_to_node(ts_.name) for ts_ in list_output_tensors(node_)]:
            continue  # this layer could be reused several times in the whole graph.
        for ts_ in list_output_tensors(node_):
            if ts_.op in inference_nodeset:
                newly.extend([ts_.op for ts_ in list_output_tensors(node_)])
        ts_end |= set(list_input_tensors(node_))

    for ts_ in list_input_mask(layer):
        ts_end.add(ts_)

    for ts_ in list_output_mask(layer):
        newly.append(ts_.op)

    visited = set()
    nodelist = list()  # keep the node list order.
    while newly:
        visited.update(newly)
        nodelist.extend(newly)
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

                newly.append(i_.op)

    return nodelist


def _general_nodelist_closure(node, nodeset, keras_nodeset):
    nodes = set()
    visited = set()

    def is_stop_node(nd):
        return is_placeholder_node(nd) or nd in keras_nodeset

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


def _get_output_nodes(node_list, node):
    nodes_has_children = set()
    for node in node_list:
        if node:
            for input_tensor in node.inputs:
                nodes_has_children.add(input_tensor.op)
    return [n_ for n_ in node_list if n_ not in nodes_has_children]  # need to keep the order.


def _filter_out_input(node_name):
    # tf.keras BN layer sometimes create a placeholder node 'scale' in tf 2.x.
    # It creates 'cond/input' since tf 2.2.
    # Given bn layer will be converted in a whole layer, it's fine to just filter this node out.
    filter_patterns = [
        r"batch_normalization_\d+\/scale$",
        r"batch_normalization_\d+\/cond/input",
        # inception_resnet_v2 has a name "conv_7b_bn" for a BN layer, just fixes by filtering.
        r"conv_\d+b_bn/cond/input"
    ]
    filter_out = False
    for pattern_ in filter_patterns:
        filter_out = filter_out or re.match(pattern_, node_name)
    return filter_out


def _advance_by_input(cur_node, layer_nodes, subgraph, inputs, graph_inputs, q_overall):
    for input_ in cur_node.inputs:
        predecessor = input_.op
        if is_placeholder_node(predecessor) and not _filter_out_input(predecessor.name):
            inputs.add(predecessor)
            graph_inputs.add(predecessor)
            continue
        if predecessor in layer_nodes or len(layer_nodes) == 0:
            subgraph.append(predecessor)
        else:
            inputs.add(predecessor)
            q_overall.put_nowait(predecessor)


def _visit_nodelist(activated_keras_nodes, input_nodes, layer_key,
                    keras_node_dict, node, nodelist, q_overall, visited):
    subgraph = list()
    i_subgraph = set()
    for ot_ in (_get_output_nodes(activated_keras_nodes, node) if activated_keras_nodes else [node]):
        if ot_ not in nodelist:
            visited.add(ot_)
            nodelist.append(ot_)
            _advance_by_input(ot_, activated_keras_nodes, subgraph, i_subgraph, input_nodes, q_overall)
    while subgraph:
        int_node = subgraph.pop(0)
        if int_node in input_nodes or int_node in visited or int_node in keras_node_dict:
            continue

        visited.add(int_node)
        nodelist.append(int_node)
        _advance_by_input(int_node, activated_keras_nodes, subgraph, i_subgraph, input_nodes, q_overall)

    return i_subgraph


def _parse_nodes(graph, inference_nodeset, graph_inputs, keras_node_dict, keras_nodeset, node, nodelist, varset,
                 visited, q_overall):
    layer_key_, model_ = (None, None)
    if node.name in keras_node_dict:
        layer_key_, model_ = keras_node_dict[node.name]
        if isinstance(layer_key_, keras.Model) and \
                _check_layer_converter_availability(layer_key_):
            k2o_logger().debug("Processing a keras sub model - %s" % layer_key_.name)
            kenode = _find_kenode_by_output_tensor(extract_inbound_nodes(layer_key_), node.name)
            ts_in, ts_out = _on_parsing_model_layer(layer_key_, graph, kenode, varset)
            for ts_ in ts_in:
                if is_placeholder_node(ts_.op):
                    graph_inputs.add(ts_.op)
                else:
                    q_overall.put_nowait(ts_.op)

            visited.update(ts_.op for ts_ in ts_out)
            return layer_key_, model_

        activated_keras_nodes = _create_keras_nodelist(layer_key_, inference_nodeset, node)
    else:
        activated_keras_nodes = _general_nodelist_closure(node, inference_nodeset, keras_nodeset)

    _visit_nodelist(activated_keras_nodes, graph_inputs, layer_key_,
                    keras_node_dict, node, nodelist, q_overall, visited)

    return layer_key_, model_


def _parse_graph_core(graph, keras_node_dict, topology, top_scope, output_names):
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

    # build the node in the working scope.
    varset = topology.declare_scope('curr_', top_scope)

    model_outputs = []
    for name in output_names:
        var_ts = graph.get_operation_by_name(tsname_to_node(name)).outputs[0]
        _create_link_node(var_ts, top_scope, varset, adjust_batch_size=True)
        model_outputs.append(var_ts.op)

    # starting from the output node.
    q_overall = queue.Queue()
    for n_ in model_outputs:
        q_overall.put_nowait(n_)

    visited = set()  # since the output could be shared among the successor nodes.
    inference_nodeset = _build_inference_nodeset(graph, model_outputs)
    keras_nodeset = _build_keras_nodeset(inference_nodeset, keras_node_dict)
    while not q_overall.empty():
        node = q_overall.get_nowait()
        if node in input_nodes or node in visited or node not in inference_nodeset:
            continue

        nodes = []
        layer_key_, model_ = _parse_nodes(graph, inference_nodeset, input_nodes, keras_node_dict, keras_nodeset,
                                          node, nodes, varset, visited, q_overall)

        if not nodes:  # already processed by the _parse_nodes
            continue

        k2o_logger().debug('Processing a keras layer - (%s: %s)' % (layer_key_.name, type(layer_key_)) if
                           layer_key_ else (nodes[0].name, "Custom_Layer"))
        if isinstance(layer_key_, keras.layers.TimeDistributed):
            _on_parsing_time_distributed_layer(graph, nodes, layer_key_, model_, varset)
        elif layer_key_ is None or get_converter(type(layer_key_)) is None:
            _on_parsing_tf_nodes(graph, nodes, varset, topology.debug_mode)
        else:
            kenode = _find_kenode_by_output_tensor(extract_inbound_nodes(layer_key_), nodes[0].name)
            on_parsing_keras_layer(graph, nodes, layer_key_, kenode, model_, varset)

    for nd_ in input_nodes:
        var_ts = nd_.outputs[0]  # since it's placeholder node, safely claim there is only one output.
        _create_link_node(var_ts, top_scope, varset, True)

    _infer_graph_shape(topology, top_scope, varset)
    topology.root_names = [variable.onnx_name for variable in top_scope.variables.values()]
    return topology


def _sorted_inputs(nodelist, outputs, inputs_set):
    inputs = []
    node_set = frozenset(nodelist)
    visited = set()

    def travel(node):
        for in_ts_ in node.inputs:
            op_node = in_ts_.op
            if op_node in visited:
                continue
            visited.add(op_node)
            if (op_node in inputs_set) and (op_node not in inputs):
                inputs.append(op_node)
            elif op_node in node_set:
                travel(op_node)

    for ts_ in outputs:
        travel(ts_.op)

    return inputs


def _parse_nodes_v2(graph, inference_nodeset, graph_inputs, keras_node_dict, node, varset, visited, q_overall):
    layer_key, model_ = (None, None)
    current_layer_outputs = {}
    if node.name in keras_node_dict:
        layer_key, model_ = keras_node_dict[node.name]
    else:
        ts_out = node.outputs[0]
        kh_ = getattr(ts_out, '_keras_history', None)
        if kh_ is not None:
            layer_key = kh_.layer
            kenode = extract_inbound_nodes(layer_key)[kh_.node_index]
            current_layer_outputs.update({ts_.op.name: (layer_key, None) for ts_ in list_output_tensors(kenode)})

    if layer_key is None:
        layer_info = LayerInfo.create_single_node(node, visited)
    else:
        if isinstance(layer_key, keras.Model):
            k2o_logger().debug("Processing a keras model layer - %s" % layer_key.name)
            kenode = _find_kenode_by_output_tensor(extract_inbound_nodes(layer_key), node.outputs[0].name)
            for ts_ in list_output_tensors(kenode):
                _create_identity(ts_.op.inputs[0], ts_, varset)
                visited.add(ts_.op)
                _advance_by_input(ts_.op, [ts_.op], list(), set(), graph_inputs, q_overall)
            return None, model_
        else:
            layer_info = LayerInfo.create(node, layer_key,
                                          {**keras_node_dict, **current_layer_outputs}, inference_nodeset)

    nodelist = []
    layer_inputs = _visit_nodelist(layer_info.nodelist, graph_inputs, None, keras_node_dict, node, nodelist,
                                   q_overall, visited)
    sorted_inputs = _sorted_inputs(layer_info.nodelist, layer_info.outputs, layer_inputs)
    for input_ in sorted_inputs:
        layer_info.inputs.extend(input_.outputs)

    layer_info.nodelist = [n_ for n_ in layer_info.nodelist if not is_placeholder_node(n_)]
    return layer_info, model_


def _parse_graph_core_v2(graph, keras_node_dict, topology, top_scope, output_names):
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

    # build the node in the working scope.
    varset = topology.declare_scope('curr_', top_scope)

    model_outputs = []
    for name in output_names:
        var_ts = graph.get_operation_by_name(tsname_to_node(name)).outputs[0]
        _create_link_node(var_ts, top_scope, varset, adjust_batch_size=True)
        model_outputs.append(var_ts.op)

    # starting from the output node.
    q_overall = queue.Queue()
    for n_ in model_outputs:
        q_overall.put_nowait(n_)

    visited = set()  # since the output could be shared among the successor nodes.
    # Some complicated layer may have some nodes which cannot be visited from the graph output...
    # ..., so the layer outputs are added into visit graph to avoid missing nodes.
    layer_outputs = [graph.get_operation_by_name(nm_) for nm_ in keras_node_dict]
    inference_nodeset = _build_inference_nodeset(graph, model_outputs + layer_outputs)
    while not q_overall.empty():
        node = q_overall.get_nowait()
        if node in input_nodes or node in visited or node not in inference_nodeset:
            continue

        layer_info, model_ = _parse_nodes_v2(graph, inference_nodeset, input_nodes, keras_node_dict, node,
                                             varset, visited, q_overall)
        if not layer_info:  # already processed by the _parse_nodes_v2
            continue

        k2o_logger().debug('Processing a keras layer - (%s: %s)' % (layer_info.layer.name, type(layer_info.layer)) if
                           layer_info.layer else (layer_info.nodelist[0].name, "Custom_Layer"))
        if layer_info.layer and isinstance(layer_info.layer, keras.layers.TimeDistributed):
            _on_parsing_time_distributed_layer(graph, layer_info.nodelist, layer_info.layer, model_, varset)
        elif layer_info.layer and get_converter(type(layer_info.layer)):
            on_parsing_keras_layer_v2(graph, layer_info, varset)
        else:
            _on_parsing_tf_nodes(graph, layer_info.nodelist, varset, topology.debug_mode)

    for nd_ in input_nodes:
        var_ts = nd_.outputs[0]  # since it's placeholder node, safely claim there is only one output.
        _create_link_node(var_ts, top_scope, varset, True)

    _infer_graph_shape(topology, top_scope, varset)
    topology.root_names = [variable.onnx_name for variable in top_scope.variables.values()]
    return topology


def parse_graph_modeless(topo, graph, target_opset, input_names, output_names, keras_node_dict):
    top_level = topo.declare_scope('__root')
    input_tensors = [graph.get_tensor_by_name(n_) for n_ in input_names]
    output_tensors = [graph.get_tensor_by_name(n_) for n_ in output_names]

    c_ = 0
    for ts_i_ in input_tensors:
        var_type = _adjust_input_batch_size(infer_variable_type(ts_i_, target_opset))
        if topo.initial_types is not None:
            if isinstance(topo.initial_types[c_], str):
                c_ += 1
            var_type = topo.initial_types[c_]
            c_ += 1
        if ts_i_.name.endswith(':0'):
            str_value = ts_i_.name[:-2]
            op = top_level.declare_local_operator(TYPES.Identity)
            var0 = top_level.get_local_variable_or_declare_one(str_value, var_type)
            var1 = top_level.get_local_variable_or_declare_one(ts_i_.name, var_type)
            op.add_input(var0)
            op.add_output(var1)
        else:
            str_value = ts_i_.name
        top_level.get_local_variable_or_declare_one(str_value, var_type)
        topo.raw_model.add_input_name(str_value)

    for ts_o_ in output_tensors:
        var_type = _adjust_input_batch_size(infer_variable_type(ts_o_, target_opset))
        # if the input types was overridden, the output shape has to be undefined.
        if topo.initial_types is not None:
            var_type.shape = []
        str_value = ts_o_.name
        top_level.get_local_variable_or_declare_one(str_value, var_type)
        topo.raw_model.add_output_name(str_value)

    return _parse_graph_core_v2(
        graph, keras_node_dict, topo, top_level, output_names
    )


def parse_graph(topo, graph, target_opset, output_names, keras_node_dict):
    # type: (Topology, tf.Graph, int, [], []) -> Topology
    """
    Build the node-layer mapper and parse the whole TF graph of Keras Model.
    """
    top_level = topo.declare_scope('__root')

    dim_variable_counter = 0
    # Create the onnx model input name before parsing to keep ...
    # ... the model input names are identical to the original Keras model.
    for idx_ in range(len(topo.raw_model.model.inputs)):
        op = top_level.declare_local_operator(TYPES.Identity)
        idx_key = idx_
        if isinstance(topo.raw_model.model.inputs, dict):
            idx_key = list(topo.raw_model.model.inputs.keys())[idx_]
        input_ts = topo.raw_model.model.inputs[idx_key]
        var_type = _adjust_input_batch_size(infer_variable_type(input_ts, target_opset))
        dim_variable_counter = _adjust_input_output_size(var_type, dim_variable_counter)
        str_value = input_ts.name
        var0 = None
        if hasattr(topo.raw_model.model, 'input_names'):
            str_value = topo.raw_model.model.input_names[idx_]
        elif input_ts.name.endswith(':0'):
            str_value = input_ts.name[:-2]
        else:
            # if there is no difference between input tensor name and model input name,
            # skip it.
            var0 = top_level.get_local_variable_or_declare_one(str_value, var_type)
        if not var0:
            var0 = top_level.get_local_variable_or_declare_one(str_value, var_type)
            var1 = top_level.get_local_variable_or_declare_one(input_ts.name, var_type)
            op.add_input(var0)
            op.add_output(var1)
        topo.raw_model.add_input_name(str_value)

    output_name_dict = {}
    output_tensors = topo.raw_model.model.outputs
    if output_names:
        output_tensors = [graph.get_tensor_by_name(n_) for n_ in output_names]
    for idx_, ts_ in enumerate(output_tensors):
        op = top_level.declare_local_operator(TYPES.Identity)
        var_type = _adjust_input_batch_size(infer_variable_type(ts_, target_opset))
        dim_variable_counter = _adjust_input_output_size(var_type, dim_variable_counter)
        str_value = ts_.name
        use_ts_name = False
        if hasattr(topo.raw_model.model, 'output_names'):
            str_value = topo.raw_model.model.output_names[idx_]
        elif ts_.name.endswith(':0'):
            str_value = tsname_to_node(ts_.name)
        else:
            # if there is no difference between output tensor name and model output name
            # skip it.
            use_ts_name = True

        if str_value in output_name_dict:
            cur_count = output_name_dict[str_value]
            output_name_dict[str_value] = cur_count + 1
            str_value = str_value + ':' + str(cur_count)
        else:
            output_name_dict[str_value] = 1

        if not use_ts_name:
            var0 = top_level.get_local_variable_or_declare_one(str_value, var_type)
            var1 = top_level.get_local_variable_or_declare_one(ts_.name, var_type)
            op.add_input(var1)
            op.add_output(var0)

        topo.raw_model.add_output_name(str_value)

    return _parse_graph_core_v2(
        graph, keras_node_dict, topo, top_level, output_names
    ) if is_tf2 and is_tf_keras else _parse_graph_core(
        graph, keras_node_dict, topo, top_level, output_names)
