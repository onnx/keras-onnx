# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import tensorflow
import numpy as np
from typing import Union
from onnx import mapping


def is_placeholder_node(node):
    return len(node.inputs) == 0 and node.type in ['Placeholder', "PlaceholderV2", 'PlaceholderWithDefault'] and \
           node.outputs[0].dtype.name != 'resource'


def tsname_to_node(name):
    return name.split(':')[0]


def is_nhwc(node):
    return node.get_attr('data_format') == b'NHWC' or node.get_attr('data_format') == b'NDHWC'


_MAX_FOLDING_NODE_NUMBER = 15


def _count_input_nodes(tensor):  # type: (tensorflow.Tensor)->int
    nodes_to_keep = set()
    node_inputs = [tensor.op]
    while node_inputs:
        nd_ = node_inputs[0]
        del node_inputs[0]
        if nd_ in nodes_to_keep:
            continue

        if is_placeholder_node(nd_):
            return -1
        nodes_to_keep.add(nd_)
        if len(nodes_to_keep) >= _MAX_FOLDING_NODE_NUMBER:
            return -1

        node_inputs.extend(in_.op for in_ in nd_.inputs)

    return len(nodes_to_keep)


def cal_tensor_value(tensor):  # type: (tensorflow.Tensor)->Union[np.ndarray, None]
    if _count_input_nodes(tensor) < 0:
        return None

    node = tensor.op
    if node.type in ["Const", "ConstV2"]:
        make_ndarray = tensorflow.make_ndarray
        np_arr = make_ndarray(node.get_attr("value"))
        return np_arr
    else:
        try:
            cls_sess = tensorflow.Session if hasattr(tensorflow, 'Session') else tensorflow.compat.v1.Session
            with cls_sess(graph=node.graph) as sess:
                np_arr = sess.run(tensor)
                return np_arr
        except (ValueError, tensorflow.errors.InvalidArgumentError, tensorflow.errors.OpError):
            return None


def cal_tensor_shape(tensor):
    if len(tensor.shape) > 0 and hasattr(tensor.shape[0], 'value'):
        return [x.value for x in tensor.shape]
    else:
        return list(tensor.shape)


def to_onnx_type(dt_type):
    # TensorFlow data types integrate seamlessly with numpy
    return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dt_type.as_numpy_dtype)]


def tf_attrs_to_onnx(node):
    attrs = {}
    for s_ in node.node_def.attr:
        if s_.startswith('T'):  # all T starts attr is TF internal.
            continue
        v = node.get_attr(s_)
        if hasattr(tensorflow.dtypes, 'DType') and isinstance(v, tensorflow.dtypes.DType):
            v = to_onnx_type(v)
        attrs[s_] = v
    return attrs


def to_tf_tensor_spec(onnx_type, name=None):
    shp = [1 if isinstance(n_, str) else n_ for n_ in onnx_type.shape]
    return tensorflow.TensorSpec(shp,
                                 mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.to_onnx_type().tensor_type.elem_type],
                                 name=name)
