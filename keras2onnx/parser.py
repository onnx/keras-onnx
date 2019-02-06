###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six
import keras
import tensorflow as tf
from six.moves import queue
from .common import keras2onnx_logger
from .common.utils import GRAPH_OUTMOST_NAME
from .ke2onnx import extract_inbound_nodes
from .common.data_types import *
from .topology import Topology
from .subgraph import get_node_by_name, is_placeholder_node, opname_to_node, create_subgraph
from .funcbook import get_converter, create_pattern_dict, fb_id, fb_key, fb_additional
from .wrapper import tf2onnx_wrap, TFNODES


DEFAULT_BATCH_SIZE = 1


class TfModelContainer(object):
    def __init__(self, graph):
        self._input_raw_names = list()
        self._output_raw_names = list()
        self.tf_graph = graph

    @property
    def raw_model(self):
        return self.tf_graph

    def add_input_name(self, name):
        # The order of adding strings matters. The final model's input names are sequentially added as this list
        if name not in self._input_raw_names:
            self._input_raw_names.append(name)

    def add_output_name(self, name):
        # The order of adding strings matters. The final model's output names are sequentially added as this list
        if name not in self._output_raw_names:
            self._output_raw_names.append(name)

    @property
    def input_names(self):
        return [name for name in self._input_raw_names]

    @property
    def output_names(self):
        return [name for name in self._output_raw_names]


# This dictionary will be updated on each conversion, since
# the user can customize their own subgraph conversion.
_SCOPE_TO_CONVERTER = {}


def _get_predefined_scope(node_name):
    for scope_re, val in six.iteritems(_SCOPE_TO_CONVERTER):
        match = scope_re.match(node_name)
        if match:
            return val[fb_key], match.group(1)
        else:  # try the other scope pattern as well
            for nre in val[fb_additional]:  # val[0] is oop type.
                match = nre.match(node_name)
                if match:
                    return val[fb_key] + '_', match.group(1)
    return None, None


def _get_scope_name(node_name):
    key, matched_name = _get_predefined_scope(node_name)
    if not matched_name:
        index = node_name.rfind('/')
        matched_name = node_name if index == -1 else node_name[0:index]

    return key, matched_name


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
        return BoolTensorType(shape=tensor_shape)
    else:
        raise ValueError('Unable to find out a correct type for tensor %s' % tensor)


def _locate_inputs_by_node(node_list, varset):
    graph = node_list[0].graph  # type: tf.Graph
    assert graph is not None

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
        node = get_node_by_name(node_list, opname_to_node(n_))
        if node is not None and (not is_placeholder_node(node)):
            nodes.append(node)

    assert nodes
    for n0_ in nodes:
        for n_ in n0_.outputs:
            var_output.append(varset.get_local_variable_or_declare_one(n_.name, _infer_variable_type(n_)))

    return var_output


def _convert_keras_scope(node_list, layer, scope_name, varset):
    operator = varset.declare_local_operator(type(layer), raw_model=layer, op_name=layer.name)
    operator.nodelist = node_list

    inputs = layer.input if isinstance(layer.input, list) else [layer.input]
    for i_ in inputs:
        iname = GRAPH_OUTMOST_NAME + '/' + i_.name
        i0 = varset.get_local_variable_or_declare_one(iname, _infer_variable_type(i_))
        operator.add_input(i0)

    if isinstance(layer.output, list):
        outputs = layer.output
        oshapes = layer.output_shape
    else:
        outputs = [layer.output]
        oshapes = [layer.output_shape]

    for n_, o_ in enumerate(outputs):
        oname = GRAPH_OUTMOST_NAME + '/' + o_.name
        o1 = varset.get_local_variable_or_declare_one(oname, _infer_variable_type(o_))
        o1.type.shape = ['None' if s_ is None else s_ for s_ in oshapes[n_]]
        operator.add_output(o1)

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator


def _convert_predefined_scope(node_list, front_nodes, scope_name, varset, kname):
    operator = varset.declare_local_operator(kname, raw_model=node_list)
    operator.nodelist = node_list

    operator.inputs = _locate_inputs_by_node(node_list, varset)[0]
    for i_ in operator.inputs:
        i_.op_to.append(operator)
    operator.outputs = _locate_outputs(node_list, varset)
    for o_ in operator.outputs:
        o_.op_from = operator
    cvt = get_converter(kname)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    for var_ in operator.inputs:
        var_.op_to.append(operator)

    return operator


def _convert_general_scope(node_list, varset):
    operator = varset.declare_local_operator(TFNODES, raw_model=node_list)
    operator.nodelist = node_list

    basename = _get_scope_name(node_list[0].name)[1]
    sgv, replacement = create_subgraph(node_list, basename)  # type: tf.Graph
    subgraph = sgv.graph
    setattr(operator, 'basename', basename)
    setattr(operator, 'subgraph', subgraph)
    vars, ts = _locate_inputs_by_node(node_list, varset)

    for n_, var_ in enumerate(vars):
        # oop = ONNX operator
        ph_ = replacement.get(ts[n_])
        assert ph_ is not None
        oop = varset.declare_local_operator('identity')
        oop.add_input(var_)
        ov = varset.declare_local_variable(ph_.name, _infer_variable_type(ph_))
        oop.add_output(ov)
        operator.add_input(ov)


def _finalize_tf2onnx_op(operator, varset):
    subgraph = operator.subgraph  # type: tf.Graph
    basename = operator.basename
    node_list = subgraph.get_operations()

    nodes = {}
    for n_ in varset.variable_name_mapping.keys():
        node = get_node_by_name(node_list, opname_to_node(n_))
        if node is not None and (not is_placeholder_node(node)):
            if node in nodes:
                nodes[node].append(n_)
            else:
                nodes[node] = [n_, ]

    outputs = []
    with subgraph.as_default():
        for n0_ in nodes:
            for i_, n_ in enumerate(n0_.outputs):
                idf_ = tf.identity(n_, basename + '_identity')
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
        g = tf2onnx_wrap(subgraph.get_operations(), outputs, varset.target_opset)
        setattr(operator, 'custom_op', g)

    return operator


def _infer_graph_shape(topology, top_level, varset):
    raw_model_container = topology.raw_model
    var_queue = queue.Queue()
    for i_ in raw_model_container.input_names:
        var_queue.put_nowait(top_level.get_local_variable_or_declare_one(i_))

    visited = set()
    while not var_queue.empty():
        var = var_queue.get_nowait()
        for oop in var.op_to:
            if oop in visited:
                continue

            visited.add(oop)
            if oop.type == TFNODES:
                _finalize_tf2onnx_op(oop, varset)
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


def _create_link_node(var, top_level, varset, reversed=False):
    ty_ = _infer_variable_type(var)
    var0 = top_level.get_local_variable_or_declare_one(var.name, ty_)
    var1 = varset.get_local_variable_or_declare_one(var.name, ty_)
    op = varset.declare_local_operator('identity')
    if reversed:
        var0, var1 = var1, var0
    op.add_input(var1)
    op.add_output(var0)
    return op


def _is_same_subgraph(node, predecessor, key, scope_name):
    """
    Test if two nodes are in the same graph
    scope_name is the node's effective scope name, the parameter 'node' is reserved.
    """
    # if the predecessor is in the same sub-graph.
    my_key, my_scope = _get_scope_name(predecessor.name)
    return my_scope == scope_name or (isinstance(key, str) and my_key == key + '_')


def _create_keras_nodelist(layer, node_list):
    newly = set()
    ts_end = set()
    for node_ in extract_inbound_nodes(layer):
        newly |= set([ts_.op for ts_ in node_.output_tensors])
        ts_end |= set(node_.input_tensors)

    visited = set()
    while newly:
        visited |= newly
        newly.clear()
        for n_ in visited:
            for i_ in n_.inputs:
                if i_ not in ts_end and i_.op not in visited:
                    newly.add(i_.op)

    return [get_node_by_name(node_list, GRAPH_OUTMOST_NAME + '/' + n_.name, exact_match=True) for n_ in visited]


def _parse_graph_scope(graph, keras_op_table, topology, top_scope, target_opset, output_names):
    node_list = graph.get_operations()
    input_nodes = []
    raw_model_container = topology.raw_model

    # build the node in the working scope.
    varset = topology.declare_scope('curr_', top_scope)
    varset.prefix = GRAPH_OUTMOST_NAME + '/'
    for node in node_list:
        if is_placeholder_node(node):
            var = node.outputs[0]
            raw_model_container.add_input_name(var.name)
            input_nodes.append(node)
            _create_link_node(var, top_scope, varset, True)

    for name in output_names:
        raw_model_container.add_output_name(name)

    model_outputs = []
    for name in output_names:
        var = graph.get_operation_by_name(opname_to_node(name)).outputs[0]
        _create_link_node(var, top_scope, varset)
        model_outputs.append(var.op)

    # starting from the output node.
    q_overall = queue.Queue()
    for n_ in model_outputs:
        q_overall.put_nowait(n_)

    visited = set()  # since the output could be shared among the successor nodes.
    keras_layer_visited = set()

    def advance_by_input(cur_node, t_key, workingset, scope_name, overall, subgraph, inputs, fronts):
        is_front = False
        for input_ in cur_node.inputs:
            predecessor = input_.op
            if predecessor in workingset:
                subgraph.put_nowait(predecessor)
            elif _is_same_subgraph(cur_node, predecessor, t_key, scope_name):
                subgraph.put_nowait(predecessor)
            else:
                is_front = True
                inputs.add(predecessor)
                overall.put_nowait(predecessor)
        if is_front:
            fronts.append(cur_node)

    while not q_overall.empty():
        node = q_overall.get_nowait()
        if node in input_nodes or node in visited:
            continue

        # begin a new scope
        nodes = [node]
        visited.add(node)
        type_k, curr_scope_name = \
            (keras_op_table[node.name], GRAPH_OUTMOST_NAME + '/' + keras_op_table[node.name].name) \
            if node.name in keras_op_table else _get_scope_name(node.name)
        if type_k in keras_layer_visited:
            continue

        activated_keras_nodes = set()
        if isinstance(type_k, keras.layers.Layer):
            activated_keras_nodes = _create_keras_nodelist(type_k, node_list)
        q_subgraph = queue.Queue()
        i_subgraph = set()
        bound_nodes = []
        advance_by_input(node, type_k, activated_keras_nodes, curr_scope_name, q_overall, q_subgraph, i_subgraph, bound_nodes)

        scope_processed = False
        while not scope_processed:
            while not q_subgraph.empty():
                int_node = q_subgraph.get_nowait()
                if int_node in input_nodes or int_node in visited or int_node.name in keras_op_table:
                    continue

                visited.add(int_node)
                nodes.append(int_node)
                advance_by_input(int_node, type_k, activated_keras_nodes, curr_scope_name, q_overall, q_subgraph, i_subgraph, bound_nodes)

            if isinstance(type_k, keras.layers.Layer):
                keras2onnx_logger().info('Processed a keras scope - (%s: %s)' % (type_k.name, type(type_k)))
                if get_converter(type(type_k)) is None:
                    _convert_general_scope(nodes, varset)
                else:
                    _convert_keras_scope(nodes, type_k, curr_scope_name, varset)
                scope_processed = True
                keras_layer_visited.add(type_k)
            elif type_k is not None:
                keras2onnx_logger().info('Processed a predefined scope - (%s)' % type_k)
                _ = _convert_predefined_scope(nodes, bound_nodes, curr_scope_name, varset, type_k)
                scope_processed = True
            else:
                if len(i_subgraph) == 0:
                    break

                # try to expand the scope with other non predefined scopes.
                while len(i_subgraph) > 0:
                    int_node = i_subgraph.pop()
                    if int_node in input_nodes or int_node in visited or int_node.name in keras_op_table:
                        continue

                    type_k, curr_scope_name = _get_scope_name(int_node.name)
                    if type_k is None:
                        q_subgraph.put_nowait(int_node)
                        break

        if not scope_processed:
            _convert_general_scope(nodes, varset)

    _infer_graph_shape(topology, top_scope, varset)
    topology.root_names = [variable.onnx_name for variable in top_scope.variables.values()]
    return topology


def parse_graph(graph, keras_op_table, target_opset, output_names):
    # type: (tf.Graph, {}, int, []) -> Topology
    global _SCOPE_TO_CONVERTER
    _SCOPE_TO_CONVERTER = create_pattern_dict()

    raw_model_container = TfModelContainer(graph)
    topology = Topology(raw_model_container, default_batch_size=DEFAULT_BATCH_SIZE, target_opset=target_opset)

    top_level = topology.declare_scope(GRAPH_OUTMOST_NAME)
    return _parse_graph_scope(graph, keras_op_table, topology, top_level, target_opset, output_names)
