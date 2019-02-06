###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
from .proto import onnx, get_opset_number_from_onnx
from .topology import convert_topology
from .common import with_variable
from .ke2onnx import *
from .parser import *
from ._builtin import *

_TF_SESSION = None


@with_variable('pb_visual_writer')
def get_tensorboard_writer():
    pb_visual_writer = None
    _tb_log_dir = os.environ.get('TB_LOG_DIR')
    if _tb_log_dir:
        from tensorflow.python.summary import summary
        pb_visual_writer = summary.FileWriter(_tb_log_dir)
    setattr(get_tensorboard_writer, 'pb_visual_writer', pb_visual_writer)
    return pb_visual_writer


def _build_opmap_from_keras(model):
    # type: (keras.Model) -> []

    static_set_ke2onnx_converters(set_converter)
    output_dict = {}
    for l_ in model.layers:
        if hasattr(l_, 'layers'):
            dict = _build_opmap_from_keras(l_)
            output_dict.update(dict)
            continue

        for node_ in extract_inbound_nodes(l_):
            for ts_ in node_.output_tensors:
                output_dict[GRAPH_OUTMOST_NAME + '/' + ts_.op.name] = l_

    return output_dict


def _convert_tf(name, tf_graph_def, keras_op_table, output_names, target_opset, doc_string, channel_first_inputs,
                debug_mode, custom_op_conversions):
    # type: (str, tf.GraphDef, {}, [], int, str, [], bool, {}) -> onnx.ModelProto
    if target_opset is None:
        target_opset = get_opset_number_from_onnx()

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tf_graph_def, name=GRAPH_OUTMOST_NAME)
        if get_tensorboard_writer() is not None:
            get_tensorboard_writer().add_graph(tf_graph)

        output_names = [GRAPH_OUTMOST_NAME + '/' + name for name in output_names]

        raw_model_container = TfModelContainer(tf_graph)
        topology = Topology(raw_model_container, default_batch_size=DEFAULT_BATCH_SIZE, target_opset=target_opset,
                            custom_op_dict=custom_op_conversions)
        topology.debug_mode = debug_mode
        parse_graph(topology, tf_graph, keras_op_table, target_opset, output_names)
        topology.compile()

        return convert_topology(topology, name, doc_string, target_opset, channel_first_inputs)


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
    from keras import backend as K

    if name is None:
        name = model.name

    op_dict = _build_opmap_from_keras(model)
    output_names = [n.name for n in model.outputs]

    sess = K.get_session()
    out_node = [n_.replace(':0', '') for n_ in output_names]
    tf_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=out_node)
    return _convert_tf(name, tf_graph_def, op_dict, output_names, target_opset, doc_string, channel_first_inputs,
                       debug_mode, custom_op_conversions)


def convert_keras_tf(name, output_names, doc_string='', target_opset=None, channel_first_inputs=None):
    # type: (str, [], str, int, []) -> onnx.ModelProto
    """
    Convert the frozen tensorflow model originally defined by Keras
    :param name:
    :param lstm_scope_name:
    :return:
    """
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(name, 'rb') as f:
        graph_def.ParseFromString(f.read())

        return _convert_tf(name, graph_def, None, output_names, target_opset, doc_string, channel_first_inputs, False, None)
