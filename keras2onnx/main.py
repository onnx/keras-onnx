###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import logging
from .proto import keras, is_tf_keras
from .proto.tfcompat import tensorflow as tf
from .proto.tfcompat import is_tf2, dump_graph_into_tensorboard
from .proto import onnx, get_opset_number_from_onnx
from .topology import convert_topology
from .ke2onnx import static_set_ke2onnx_converters
from .parser import parse_graph
from .topology import Topology
from .common.utils import set_logger_level
from .funcbook import set_converter
from ._parse_tf import is_placeholder_node, tsname_to_node, build_layer_output_from_model
from ._parser_1x import build_opdict_from_keras


def convert_keras(model, name=None, doc_string='', target_opset=None,
                  channel_first_inputs=None, debug_mode=False, custom_op_conversions=None):
    # type: (keras.Model, str, str, int, [], bool, {}) -> onnx.ModelProto
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string: doc string
    :param target_opset: the targeted onnx model opset
    :param channel_first_inputs: A list of channel first input
    :param debug_mode: will enable the log and try to convert as much as possible on conversion
    :param custom_op_conversions: the handler for custom operator conversion
    :return an ONNX ModelProto
    """
    if isinstance(model, tf.keras.Model) and not is_tf_keras:
        raise Exception("This is a tensorflow keras model, but keras standalone converter is used." +
                        " Please set environment variable TF_KERAS = 1.")

    set_logger_level(logging.DEBUG if debug_mode else logging.INFO)
    if debug_mode:
        print(model.summary())

    name = name or model.name
    target_opset = target_opset or get_opset_number_from_onnx()

    output_names = []
    output_dict = {}
    if is_tf2 and is_tf_keras:
        tf_graph = build_layer_output_from_model(model, output_dict, output_names)
    else:
        tf_graph = model.outputs[0].graph if is_tf2 else keras.backend.get_session().graph
        output_dict = build_opdict_from_keras(model)
        output_names = [n.name for n in model.outputs]

    static_set_ke2onnx_converters(set_converter)
    dump_graph_into_tensorboard(tf_graph)
    topology = Topology(model, tf_graph,
                        target_opset=target_opset,
                        custom_op_dict=custom_op_conversions)
    topology.debug_mode = debug_mode
    parse_graph(topology, tf_graph, target_opset, output_names, output_dict)
    topology.compile()

    return convert_topology(topology, name, doc_string, target_opset, channel_first_inputs)


def build_io_names_tf2onnx(model):
    return {
        'input_names': [n_.name for n_ in model.inputs],
        'output_names': [n_.name for n_ in model.outputs]
    }


def _freeze_graph(session, keep_var_names=None, output_names=None):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph_def


def export_tf_frozen_graph(model, keep_var_names=None, output_names=None):
    """
    Freezes internal tensorflow graph for the specified keras model.
    :return The frozen graph object.
    """
    output_names = output_names or \
                   [tsname_to_node(n_) for n_ in build_io_names_tf2onnx(model)['output_names']]
    return _freeze_graph(keras.backend.get_session(), keep_var_names, output_names)


def _collect_input_nodes(graph, outputs):
    nodes_to_keep = set()
    input_nodes = set()
    node_inputs = [graph.get_tensor_by_name(ts_).op for ts_ in outputs]
    while node_inputs:
        nd_ = node_inputs[0]
        del node_inputs[0]
        if is_placeholder_node(nd_):
            input_nodes.add(nd_)
        if nd_ in nodes_to_keep:
            continue

        nodes_to_keep.add(nd_)
        node_inputs.extend(in_.op for in_ in nd_.inputs)

    return input_nodes, nodes_to_keep


def convert_tensorflow(frozen_graph_def,
                       name=None, input_names=None, output_names=None,
                       doc_string='',
                       target_opset=None,
                       channel_first_inputs=None,
                       debug_mode=False, custom_op_conversions=None):
    """
    convert a tensorflow graph def into a ONNX model proto, just like how keras does.
    :param graph_def: the frozen tensorflow graph
    :param name: the converted onnx model internal name
    :param input_names: the inputs name list of the model
    :param output_names: the output name list of the model
    :param doc_string: doc string
    :param target_opset: the targeted onnx model opset
    :param channel_first_inputs: A list of channel first input (not supported yet)
    :param debug_mode: will enable the log and try to convert as much as possible on conversion
    :return an ONNX ModelProto
    """
    set_logger_level(logging.DEBUG if debug_mode else logging.INFO)
    from .wrapper import tf2onnx

    if target_opset is None:
        target_opset = get_opset_number_from_onnx()

    if not doc_string:
        doc_string = "converted from {}".format(name)

    tf_graph_def = tf2onnx.tfonnx.tf_optimize(input_names, output_names, frozen_graph_def, True)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tf_graph_def, name='')

    with tf.Session(graph=tf_graph):
        if not input_names:
            input_nodes = list(_collect_input_nodes(tf_graph, output_names)[0])
            input_names = [nd_.outputs[0].name for nd_ in input_nodes]
        g = tf2onnx.tfonnx.process_tf_graph(tf_graph,
                                            continue_on_error=debug_mode,
                                            opset=target_opset,
                                            custom_op_handlers=custom_op_conversions,
                                            inputs_as_nchw=channel_first_inputs,
                                            output_names=output_names,
                                            input_names=input_names)

        onnx_graph = tf2onnx.optimizer.optimize_graph(g)
        model_proto = onnx_graph.make_model(doc_string)

    return model_proto
