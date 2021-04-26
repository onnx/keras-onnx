# SPDX-License-Identifier: Apache-2.0

import logging
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from .proto import keras, is_tf_keras
from .proto.tfcompat import tensorflow as tf
from .proto.tfcompat import is_tf2, dump_graph_into_tensorboard
from .proto import onnx
from .topology import convert_topology
from .ke2onnx import static_set_ke2onnx_converters
from .parser import parse_graph, parse_graph_modeless
from .topology import Topology
from .common.utils import set_logger_level, k2o_logger
from .funcbook import set_converter
from ._tf_utils import tsname_to_node, to_tf_tensor_spec
from ._builtin import register_direct_tf_ops
from ._parser_1x import build_opdict_from_keras
from ._parser_tf import build_layer_output_from_model


def _process_initial_types(initial_types):
    if initial_types is None:
        return None

    input_specs = []
    c_ = 0
    while c_ < len(initial_types):
        name = None
        type_idx = c_
        if isinstance(initial_types[c_], str):
            name = initial_types[c_]
            type_idx = c_ + 1
        ts_spec = to_tf_tensor_spec(initial_types[type_idx], name)
        input_specs.append(ts_spec)
        c_ += 1 if name is None else 2

    return input_specs


def convert_keras_tf2onnx(model, name=None, doc_string='', target_opset=None, initial_types=None,
                          channel_first_inputs=None, debug_mode=False, custom_op_conversions=None):
    if target_opset is None:
        target_opset = 13
    input_signature = _process_initial_types(initial_types)

    import tf2onnx
    model, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature, opset=target_opset)

    return model


def convert_keras(*args, **kwargs):
    return convert_keras_tf2onnx(*args, **kwargs)


def convert_keras_old(model, name=None, doc_string='', target_opset=None, initial_types=None,
                      channel_first_inputs=None, debug_mode=False, custom_op_conversions=None):
    # type: (keras.Model, str, str, int, [], [], bool, {}) -> onnx.ModelProto
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string: doc string
    :param target_opset: the targeted onnx model opset
    :param initial_types: the overridden input type for the target ONNX model.
    :param channel_first_inputs: A list of channel first input
    :param debug_mode: will enable the log and try to convert as much as possible on conversion
    :param custom_op_conversions: the handler for custom operator conversion
    :return an ONNX ModelProto
    """
    print("Hello! I'm a test")
    if isinstance(model, tf.keras.Model) and not is_tf_keras:
        raise Exception("This is a tensorflow keras model, but keras standalone converter is used." +
                        " Please set environment variable TF_KERAS = 1 before importing keras2onnx.")

    set_logger_level(logging.DEBUG if debug_mode else logging.INFO)
    if is_tf2:
        from tensorflow.python.eager import context
        k2o_logger().info("tf executing eager_mode: {}".format(context.executing_eagerly()))
        if hasattr(model, 'run_eagerly'):
            k2o_logger().info("tf.keras model eager_mode: {}".format(model.run_eagerly))
    if debug_mode:
        print(model.summary())

    name = name or model.name
    cvt_default_opset = get_maximum_opset_supported()
    if target_opset is None:
        target_opset = cvt_default_opset
    elif target_opset > cvt_default_opset:
        raise RuntimeError(
            "The opset {} conversion not support yet, the current maximum opset version supported is {}.".format(
                target_opset, cvt_default_opset))
    input_names = []
    output_names = []
    output_dict = {}
    if is_tf2 and is_tf_keras:
        tf_graph = build_layer_output_from_model(model, output_dict, input_names,
                                                 output_names, _process_initial_types(initial_types))
    else:
        tf_graph = model.outputs[0].graph if is_tf2 else keras.backend.get_session().graph
        output_dict = build_opdict_from_keras(model)
        output_names = [n.name for n in model.outputs]

    static_set_ke2onnx_converters(set_converter)
    register_direct_tf_ops()
    dump_graph_into_tensorboard(tf_graph)
    topology = Topology(model, tf_graph,
                        target_opset=target_opset,
                        initial_types=initial_types,
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
