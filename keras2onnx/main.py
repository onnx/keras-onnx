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
from .parser import parse_graph, parse_graph_modeless
from .topology import Topology
from .common.utils import set_logger_level, k2o_logger
from .funcbook import set_converter
from ._parse_tf import tsname_to_node, build_layer_output_from_model
from ._parser_1x import build_opdict_from_keras


def _get_maximum_opset_supported():
    default_max_opset = 11
    try:
        from onnxconverter_common.topology import DEFAULT_OPSET_NUMBER
        default_max_opset = DEFAULT_OPSET_NUMBER
    except:  # noqa
        pass
    return min(default_max_opset, onnx.defs.onnx_opset_version())


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
    if is_tf2:
        from tensorflow.python.eager import context
        k2o_logger().info("tf executing eager_mode: {}".format(context.executing_eagerly()))
        if hasattr(model, 'run_eagerly'):
            k2o_logger().info("tf.keras model eager_mode: {}".format(model.run_eagerly))
    if debug_mode:
        print(model.summary())

    name = name or model.name
    target_opset = target_opset or _get_maximum_opset_supported()
    output_names = [n.name for n in model.outputs]

    input_names = []
    output_names = []
    output_dict = {}
    if is_tf2 and is_tf_keras:
        tf_graph = build_layer_output_from_model(model, output_dict, input_names, output_names)
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
    if (not model.inputs) or (not model.outputs):
        # Since Tensorflow 2.2, For the subclassed tf.keras model, there is no inputs/outputs info ...
        # ... stored in model object any more.
        parse_graph_modeless(topology, tf_graph, target_opset, input_names, output_names, output_dict)
    else:
        parse_graph(topology, tf_graph, target_opset, output_names, output_dict)
    topology.compile()

    return convert_topology(topology, name, doc_string, target_opset, channel_first_inputs)


def build_io_names_tf2onnx(model):
    return {
        'input_names': [n_.name for n_ in model.inputs],
        'output_names': [n_.name for n_ in model.outputs]
    }


def _freeze_graph(session, keep_var_names=None, output_names=None):
    graph = tf.get_default_graph()
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
    if is_tf2:
        raise RuntimeError("Only Tensorflow 1.x supported.")
    session = keras.backend.get_session()
    graph = model.outputs[0].graph if is_tf2 else session.graph
    with graph.as_default():
        output_names = output_names or \
                       [tsname_to_node(n_) for n_ in build_io_names_tf2onnx(model)['output_names']]
        return _freeze_graph(session, keep_var_names, output_names)
