###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .common import keras2onnx_logger
from tf2onnx.tfonnx import *
from .funcbook import set_converter


def tf2onnx_wrap(topo, node_list, outputs, target_opset):
    """
    A wrapper function to invoke the basic node conversion from tf2onnx
    """
    try:
        onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = tflist_to_onnx(node_list, {})

        g = Graph(onnx_nodes, output_shapes, dtypes, opset=target_opset, output_names=outputs)
        ops = g.get_nodes()
        g.topological_sort(ops)

        _ = tensorflow_onnx_mapping(g, topo.debug_mode, topo.custom_op_dict)
        g.topological_sort(g.get_nodes())
        g.update_proto()
        return g

    except Exception as e:
        for node_ in node_list:
            keras2onnx_logger().debug("tfnode: {}".format(node_.name))
        raise e


def tfnode_convert(varset, operator, container):
    """
    merge the output node from tf2onnx into the final graph.
    """
    g = operator.tf2onnx_graph

    # update attributes
    all_inputs = set()
    for op in g.get_nodes():
        all_inputs |= set(op.input)
        onnx_op = op.op
        op_inputs = [varset.get_local_variable_or_declare_one(n_).full_name.encode('utf-8') for n_ in onnx_op.input]
        op_outputs = [varset.get_local_variable_or_declare_one(n_).full_name.encode('utf-8') for n_ in onnx_op.output]
        onnx_op.name = varset.get_unique_operator_name(onnx_op.name).encode('utf-8')
        onnx_op.input[:] = op_inputs
        onnx_op.output[:] = op_outputs
        container.add_onnx_node(onnx_op, op_version=container.target_opset)

    # create input_tensor_values, initializers
    # if initilizer is not used as input by any node, then it will be ignored
    initializers = [i for i in list(g.initializers.values()) if i.name in all_inputs]
    for init_tensor_ in initializers:
        init_tensor_.name = varset.get_local_variable_or_declare_one(init_tensor_.name).full_name.encode('utf-8')
        container.add_initializer_from_tensor(init_tensor_)


TFNODES = 'TFNodes'
set_converter(TFNODES, tfnode_convert)
