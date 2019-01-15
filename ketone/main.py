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
        # if get_converter(type(l_)) is None:
        #     continue
        #
        for node_ in extract_inbound_nodes(l_):
            for ts_ in node_.output_tensors:
                output_dict[GRAPH_OUTMOST_NAME + '/' + ts_.op.name] = l_

    return output_dict


def _convert_tf(name, tf_graph_def, keras_op_table, output_names, target_opset, doc_string):
    # type: (str, tf.GraphDef, {}, [], int, str) -> onnx.ModelProto
    if target_opset is None:
        target_opset = get_opset_number_from_onnx()

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tf_graph_def, name=GRAPH_OUTMOST_NAME)
        if get_tensorboard_writer() is not None:
            get_tensorboard_writer().add_graph(tf_graph)

        output_names = [GRAPH_OUTMOST_NAME + '/' + name for name in output_names]

        topology = parse_graph(tf_graph, keras_op_table, target_opset, output_names)
        topology.compile()

        return convert_topology(topology, name, doc_string, target_opset)


def convert_keras(model, name=None, doc_string='', target_opset=None):
    # type: (keras.Model, str, str, int) -> onnx.ModelProto
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string:
    :param target_opset:
    :return:
    """
    from keras import backend as K

    if name is None:
        name = model.name

    op_dict = _build_opmap_from_keras(model)
    output_names = [n.name for n in model.outputs]

    global _TF_SESSION
    if _TF_SESSION is not None:
        _TF_SESSION.close()

    _TF_SESSION = tf.Session(graph=tf.get_default_graph())
    sess = _TF_SESSION
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    out_node = [n_.replace(':0', '') for n_ in output_names]
    tf_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=out_node)
    return _convert_tf(name, tf_graph_def, op_dict, output_names, target_opset, doc_string)


def convert_keras_tf(name, output_names, doc_string='', target_opset=None):
    # type: (str, [], str, int) -> onnx.ModelProto
    """
    Convert the frozen tensorflow model originally defined by Keras
    :param name:
    :param lstm_scope_name:
    :return:
    """
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(name, 'rb') as f:
        graph_def.ParseFromString(f.read())

        return _convert_tf(name, graph_def, None, output_names, target_opset, doc_string)
