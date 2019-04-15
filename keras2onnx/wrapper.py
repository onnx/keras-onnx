###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import tf2onnx
from distutils.version import StrictVersion
from tf2onnx.tfonnx import process_tf_graph
from onnx import numpy_helper
from .common import k2o_logger
from .funcbook import set_converter


def tf2onnx_wrap(topo, graph, outputs, target_opset):
    """
    A wrapper function to invoke the basic node conversion from *tf2onnx*.
    """
    try:
        g = process_tf_graph(graph,
                         continue_on_error=topo.debug_mode,
                         opset=target_opset,
                         custom_op_handlers=topo.custom_op_dict,
                         output_names=outputs)
        return g

    except Exception as e:
        k2o_logger().warning("Exception on this tf.graph\n" +
                             '\n'.join(op_.name for op_ in graph.get_operations()))
        raise e


def _update_container(varset, op, container):
    onnx_op = op.op
    op_inputs = [varset.get_local_variable_or_declare_one(n_).full_name.encode('utf-8') for n_ in onnx_op.input]
    op_outputs = [varset.get_local_variable_or_declare_one(n_).full_name.encode('utf-8') for n_ in onnx_op.output]
    onnx_op.name = varset.get_unique_operator_name(onnx_op.name).encode('utf-8')
    onnx_op.input[:] = op_inputs
    onnx_op.output[:] = op_outputs
    container.add_onnx_node(onnx_op, op_version=container.target_opset)


def tfnode_convert(varset, operator, container):
    """
    Merge the output node from *tf2onnx* into the final graph.
    """
    g = operator.tf2onnx_graph
    g.update_proto()

    # update attributes
    all_inputs = set()
    if StrictVersion(tf2onnx.__version__) <= StrictVersion('0.3.2'):
        for op in g.get_nodes():
            all_inputs |= set(op.input)
            _update_container(varset, op, container)

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
                _update_container(varset, op, container)

    for init_tensor_ in initializers:
        init_tensor_.name = varset.get_local_variable_or_declare_one(init_tensor_.name).full_name.encode('utf-8')
        container.add_initializer_from_tensor(init_tensor_)


TFNODES = 'TFNodes'
set_converter(TFNODES, tfnode_convert)
