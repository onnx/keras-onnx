###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import logging
import tf2onnx
from .proto import keras
from .proto import onnx, get_opset_number_from_onnx
from .topology import convert_topology
from .common import with_variable
from .ke2onnx import static_set_ke2onnx_converters
from .parser import parse_graph, DEFAULT_BATCH_SIZE
from .topology import Topology
from .common.utils import set_logger_level
from ._builtin import set_converter


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
    :return:
    """
    set_logger_level(logging.DEBUG if debug_mode else logging.INFO)
    tf2onnx.logging.set_level(logging.DEBUG if debug_mode else logging.INFO)

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
    topology = Topology(raw_model_container, default_batch_size=DEFAULT_BATCH_SIZE, target_opset=target_opset,
                        custom_op_dict=custom_op_conversions)
    topology.debug_mode = debug_mode
    parse_graph(topology, sess.graph, target_opset, output_names)
    topology.compile()

    return convert_topology(topology, name, doc_string, target_opset, channel_first_inputs)
