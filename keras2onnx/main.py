###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import logging
import tf2onnx
import tensorflow as tf
from .proto import keras, is_tf_keras
from .proto import onnx, get_opset_number_from_onnx
from .topology import convert_topology
from .common import with_variable, k2o_logger
from .ke2onnx import static_set_ke2onnx_converters
from .parser import parse_graph, DEFAULT_BATCH_SIZE, tsname_to_node
from .topology import Topology
from .common.utils import set_logger_level
from ._builtin import set_converter, tf2onnx_builtin_conversion


class KerasTfModelContainer(object):
    def __init__(self, graph, model=None):
        self._input_raw_names = list()
        self._output_raw_names = list()
        self.tf_graph = graph
        self.model = model

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


@with_variable('pb_visual_writer')
def get_tensorboard_writer():
    pb_visual_writer = None
    _tb_log_dir = os.environ.get('TB_LOG_DIR')
    if _tb_log_dir:
        from tensorflow.python.summary import summary
        pb_visual_writer = summary.FileWriter(_tb_log_dir)
    setattr(get_tensorboard_writer, 'pb_visual_writer', pb_visual_writer)
    return pb_visual_writer


def convert_keras(model, name=None, doc_string='', target_opset=None, channel_first_inputs=None, debug_mode=False,
                  custom_op_conversions=None):
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
    set_logger_level(logging.DEBUG if debug_mode else logging.INFO)
    tf2onnx.logging.set_level(logging.DEBUG if debug_mode else logging.INFO)

    if isinstance(model, tf.keras.Model) and not is_tf_keras:
        raise Exception("This is a tensorflow keras model, but keras standalone converter is used." +
                        " Please set environment variable TF_KERAS = 1.")

    if name is None:
        name = model.name

    if target_opset is None:
        target_opset = get_opset_number_from_onnx()

    output_names = [n.name for n in model.outputs]

    static_set_ke2onnx_converters(set_converter)

    sess = keras.backend.get_session()
    if get_tensorboard_writer() is not None:
        get_tensorboard_writer().add_graph(sess.graph)
    raw_model_container = KerasTfModelContainer(sess.graph, model)
    topology = Topology(raw_model_container,
                        default_batch_size=DEFAULT_BATCH_SIZE,
                        target_opset=target_opset,
                        custom_op_dict=custom_op_conversions)
    topology.debug_mode = debug_mode
    parse_graph(topology, sess.graph, target_opset, output_names)
    topology.compile()

    return convert_topology(topology, name, doc_string, target_opset, channel_first_inputs)


def build_io_names_tf2onnx(model):
    return {
        'input_names': [n_.name for n_ in model.inputs],
        'output_names': [n_.name for n_ in model.outputs]
    }


def export_tf_frozen_graph(model, keep_var_names=None, output_names=None):
    """
    Freezes internal tensorflow graph for the specified keras model.

    :return The frozen graph object.
    """
    session = keras.backend.get_session()
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or \
                       [tsname_to_node(n_) for n_ in build_io_names_tf2onnx(model)['output_names']]
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph_def


def convert_tensorflow(frozen_graph_def,
                       name=None, input_names=None, output_names=None,
                       doc_string='',
                       target_opset=None,
                       channel_first_inputs=None,
                       debug_mode=False, custom_op_conversions=None):
    """
    convert a frozen tensorflow graph def into a ONNX model proto, just like how keras does.
    :param frozen_graph_def: the frozen tensorflow graph
    :param name: the converted onnx model internal name
    :param input_names: the inputs name list of the model
    :param output_names: the output name list of the model
    :param doc_string: doc string
    :param target_opset: the targeted onnx model opset
    :param channel_first_inputs: A list of channel first input (not supported yet)
    :param debug_mode: will enable the log and try to convert as much as possible on conversion
    :return an ONNX ModelProto
    """
    from uuid import uuid4
    set_logger_level(logging.DEBUG if debug_mode else logging.INFO)

    if name is None:
        name = str(uuid4())

    if target_opset is None:
        target_opset = get_opset_number_from_onnx()

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_graph_def, name='')
        if get_tensorboard_writer() is not None:
            get_tensorboard_writer().add_graph(tf_graph)

    custom_op_handlers = tf2onnx_builtin_conversion(target_opset)
    if custom_op_conversions:
        custom_op_handlers += custom_op_conversions
    with tf.Session(graph=tf_graph):
        g = tf2onnx.tfonnx.process_tf_graph(tf_graph,
                                            continue_on_error=debug_mode,
                                            opset=target_opset,
                                            custom_op_handlers=custom_op_handlers,
                                            inputs_as_nchw=channel_first_inputs,
                                            output_names=output_names,
                                            input_names=input_names)

        model_proto = g.make_model(doc_string, graph_name=name)
        model_proto = tf2onnx.graph.GraphUtil.optimize_model_proto(model_proto)
        return model_proto
