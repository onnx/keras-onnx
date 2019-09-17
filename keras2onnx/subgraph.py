###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import six
import copy
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util


def is_placeholder_node(node):
    return len(node.inputs) == 0 and node.type in ['Placeholder', "PlaceholderV2", 'PlaceholderWithDefault']


def tsname_to_node(name):
    return name.split(':')[0]


def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


def _copy_node_wo_dep(node):
    newnode = copy.deepcopy(node)
    idx = []
    for ix_, inp_ in enumerate(newnode.input):
        if inp_.startswith('^'):
            idx.append(ix_)

    for ix_ in idx[::-1]:
        del newnode.input[ix_]
    return newnode


def _extract_sub_graph(graph_def, dest_nodes, stop_nodes):
    if not isinstance(graph_def, graph_pb2.GraphDef):
        raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

    if isinstance(dest_nodes, six.string_types):
        raise TypeError("dest_nodes must be a list.")

    name_to_node = {_node_name(n_.name): _copy_node_wo_dep(n_) for n_ in graph_def.node}

    nodes_to_keep = dest_nodes[:]

    # Now construct the output GraphDef
    out = graph_pb2.GraphDef()
    for n_ in nodes_to_keep:
        out.node.extend([name_to_node[n_]])
    out.library.CopyFrom(graph_def.library)
    out.versions.CopyFrom(graph_def.versions)

    return out


def create_subgraph(tf_graph, node_list, sess, dst_scope=None):
    """
    Create a tf subgraph from the node list.
    :param tf_graph:
    :param node_list:
    :param sess:
    :param dst_scope:
    :return:
    """
    variable_dict_names = []
    variable_names = []
    tensor_op_names = []
    for n_ in node_list:  # type: tf.Operation
        tensor_op_names.extend([ts_.op.name for ts_ in n_.inputs])
        if n_.type in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = n_.name
            variable_dict_names.append(variable_name)

            if n_.type == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))
    all_op_names = set([n_.name for n_ in node_list])
    missing_ops = set(tensor_op_names) - all_op_names

    replacement = {}
    tf_graph_def = tf_graph.as_graph_def()
    subgraph_def = _extract_sub_graph(tf_graph_def, [n_.name for n_ in node_list], missing_ops)

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in subgraph_def.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]
            output_node.attr["dtype"].CopyFrom(dtype)
            output_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(
                        data, dtype=dtype.type, shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (
                input_node.input[0] in found_variables):
            # The preceding branch converts all VarHandleOps of ResourceVariables to
            # constants, so we need to convert the associated ReadVariableOps to
            # Identity ops.
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        elif input_node.name not in missing_ops:
            output_node.CopyFrom(input_node)
        else:
            output_node = None
        if output_node is not None:
            output_graph_def.node.extend([output_node])

    for input_node in tf_graph_def.node:
        if input_node.name in missing_ops:
            output_node = node_def_pb2.NodeDef()
            output_node.op = "Placeholder"
            output_node.name = input_node.name
            replacement[input_node.name] = input_node.name
            if str(input_node.attr["dtype"]):
                output_node.attr["dtype"].CopyFrom(input_node.attr["dtype"])
            elif str(input_node.attr["T"]):
                output_node.attr["dtype"].CopyFrom(input_node.attr["T"])
            else:
                if input_node.op in ['All', 'Any']:
                    output_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type="DT_BOOL"))
                elif input_node.op == 'Cast':
                    output_node.attr["dtype"].CopyFrom(input_node.attr["DstT"])
                else:
                    raise RuntimeError("Can't get the node data type for %s" % input_node.name)
            ts_shape = tf.graph_util.tensor_shape_from_node_def_name(tf_graph, input_node.name)
            output_node.attr["shape"].CopyFrom(
                attr_value_pb2.AttrValue(shape=ts_shape.as_proto()))
            output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(subgraph_def.library)
    with tf.Graph().as_default() as sub_graph:
        im_scope = "" if dst_scope is None else dst_scope
        tf.import_graph_def(output_graph_def, name=im_scope)
        if im_scope:
            replacement = {k_: im_scope + '/' + k_ for k_ in replacement}

    return sub_graph, replacement
