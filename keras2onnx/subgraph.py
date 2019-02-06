###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import re
import tensorflow as tf
from .common import keras2onnx_logger

ge = tf.contrib.graph_editor


class InputCopyHandler(object):
    def __init__(self):
        self.replacement = {}

    def __call__(self, info, t):
        if t in self.replacement:
            return self.replacement[t]
        ts_new = ge.transform.keep_t_if_possible_handler(info, t)
        self.replacement[t] = ts_new
        keras2onnx_logger().debug("{} -> {}".format(t, ts_new))
        return ts_new


def create_subgraph(node_list, dst_scope=""):
    """
    Create a tf subgraph from the node list.
    :param node_list:
    :param dst_scope:
    :return:
    """
    sgv = ge.sgv(node_list)
    copier = ge.Transformer()
    handler = InputCopyHandler()
    copier.transform_external_input_handler = handler
    copied_sgv, info = copier(sgv, tf.Graph(), dst_scope)
    return copied_sgv, handler.replacement


def get_node_by_name(node_list, name, exact_match=False):
    """
    select the node by its name, without comparing the scope name.
    :param node_list:
    :param name:
    :param exact_match:
    :return:
    """
    try:
        if exact_match:
            return next(node for node in node_list if node.name == name)
        return next(node for node in node_list if node.name.endswith(name))
    except StopIteration:
        return None


def is_placeholder_node(node):
    return node.type == 'Placeholder'


def opname_to_node(name):
    if not hasattr(opname_to_node, '_OPNAME_PATTERN'):
        setattr(opname_to_node, '_OPNAME_PATTERN', re.compile(r'(.*):\d+$'))
    return opname_to_node._OPNAME_PATTERN.match(name).group(1)
