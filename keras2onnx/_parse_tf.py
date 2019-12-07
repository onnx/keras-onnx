###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import tensorflow as tf
from .common import k2o_logger, get_default_batch_size
from .funcbook import get_converter

from .proto import keras
from .proto.tfcompat import normalize_tensor_shape
from onnxconverter_common import Int32TensorType, Int64TensorType, FloatTensorType, DoubleTensorType, BooleanTensorType


def is_placeholder_node(node):
    return len(node.inputs) == 0 and node.type in ['Placeholder', "PlaceholderV2", 'PlaceholderWithDefault'] and \
           node.outputs[0].dtype.name != 'resource'


def tsname_to_node(name):
    return name.split(':')[0]


def infer_variable_type(tensor, opset):
    tensor_shape = []
    if tensor.shape not in (tf.TensorShape(None), tf.TensorShape([])):
        if opset > 8:
            tensor_shape = normalize_tensor_shape(tensor.shape)
        else:  # most inference engine has problem with unset dim param if they released around opset 8 publish
            tensor_shape = ['None' if d is None else d for d in normalize_tensor_shape(tensor.shape)]

    # Determine the tensor's element type
    tensor_type = tensor.dtype
    if tensor.dtype == 'resource':
        node_attr = tensor.op.node_def.attr
        tensor_type = node_attr['dtype'].type
        tensor_shape = ['None' if d.size is None else d.size for d in node_attr['shape'].shape.dim]
    if tensor_type in [tf.int8, tf.int16, tf.int32]:
        return Int32TensorType(shape=tensor_shape)
    elif tensor_type == tf.int64:
        return Int64TensorType(shape=tensor_shape)
    elif tensor_type in [tf.float16, tf.float32]:
        return FloatTensorType(shape=tensor_shape)
    elif tensor_type == tf.float64:
        return DoubleTensorType(shape=tensor_shape)
    elif tensor_type == tf.bool:
        return BooleanTensorType(shape=tensor_shape)
    else:
        raise ValueError(
            "Unable to find out a correct type for tensor type = {} of {}".format(tensor_type, tensor.name))


def adjust_input_batch_size(var_type):
    if len(var_type.shape) > 0 and var_type.shape[0] is None:
        var_type.shape = [get_default_batch_size()] + var_type.shape[1:]
    return var_type


def _get_layer_name(ts_or_op):
    return ts_or_op.rsplit('/', 1)[0]


class LayerInfo(object):
    def __init__(self, _ly):
        self.layer = _ly
        self.inputs = []
        self.outputs = []
        self.nodelist = []

    @staticmethod
    def create(node, layer, outputs_map, inference_nodeset):
        graph = node.graph
        layer_info = LayerInfo(layer)
        # find the output
        visited = set()
        for nn_, layer_info_ in outputs_map.items():
            if layer_info_[0] == layer and _get_layer_name(nn_) == _get_layer_name(node.name):
                op_node = graph.get_operation_by_name(tsname_to_node(nn_))
                visited.add(op_node)
                layer_info.outputs.extend(op_node.outputs)

        next_itr = set(visited)
        while next_itr:
            visited |= next_itr
            next_itr.clear()
            for n_ in visited:
                for i_ in n_.inputs:
                    if i_.op in visited or i_.op not in inference_nodeset:
                        continue
                    if (not is_placeholder_node(i_.op)) and i_ in outputs_map:
                        continue
                    next_itr.add(i_.op)

        layer_info.nodelist = list(visited)
        return layer_info


def is_subclassing(model):
    return not (model._is_graph_network or  # pylint:disable=protected-access
                isinstance(model, keras.engine.sequential.Sequential))


def layer_name_dict(layer, prefix, parent=None):
    output_dict = {}
    if hasattr(layer, 'layers'):
        for l_ in layer.layers:
            prefix_l = "{}/{}".format(prefix, l_.name)
            submodel_dict = layer_name_dict(l_, prefix_l, layer)
            output_dict.update(submodel_dict)

    output_dict[prefix] = (layer, parent)
    return output_dict


def _is_input_output_tensor(ts):
    return ts.name.find('/') == -1 or ts.op.type == 'ReadVariableOp'


def build_layer_outputs(model, graph, outputs):
    # type: (keras.Model, tf.Graph, []) -> {}

    output_dict = {}
    layer_dict = layer_name_dict(model, model.name)

    for ou_ in outputs:
        if _is_input_output_tensor(ou_):
            continue
        layer_name = _get_layer_name(ou_.name)
        if layer_name not in layer_dict:
            layer_name = layer_name.rsplit('_', 1)[0]

        assert layer_name in layer_dict, "Cannot find the Keras layer of the output tensor({}).".format(ou_.name)
        output_dict[tsname_to_node(ou_.name)] = layer_dict[layer_name]

    return output_dict


def extract_outputs_from_subclassing_model(model, output_dict, output_names):
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.python.keras.saving import saving_utils as _saving_utils
    from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
    from tensorflow.python.framework import convert_to_constants as _convert_to_constants

    function = _saving_utils.trace_model_call(model)
    concrete_func = function.get_concrete_function()
    output_names.extend([ts_.name for ts_ in concrete_func.outputs])
    tf_graph = concrete_func._first_order_tape_functions.forward.graph
    output_dict.update(build_layer_outputs(model, tf_graph, concrete_func.outputs))
    frozen_func = _convert_to_constants.convert_variables_to_constants_v2(
        concrete_func, lower_control_flow=True)
    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != tf.dtypes.resource
    ]
    output_tensors = frozen_func.outputs
    graph_def = frozen_func.graph.as_graph_def()
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.constant_folding = rewrite_options.ON
    graph_def = _run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=config,
        graph=frozen_func.graph)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')

    return tf_graph


def outputs_to_dict(graph, outputs):
    t2l_dict = {}
    for k_, v_ in outputs.items():
        op = graph.get_operation_by_name(tsname_to_node(k_))
        assert op is not None, "Cannot find the {} in the graph".format(k_)
        t2l_dict.update({ts_k_: v_ for ts_k_ in op.outputs})

    return t2l_dict


def on_parsing_keras_layer_v2(graph, layer_info, varset, prefix=None):
    layer = layer_info.layer
    node_list = layer_info.nodelist
    operator = varset.declare_local_operator(type(layer), raw_model=layer, op_name=layer.name)
    operator.nodelist = node_list

    inputs = layer_info.inputs
    outputs = layer_info.outputs

    if prefix is None:  # prefix is designed for the distinguish among the shared model instances.
        prefix = ''

    for n_, o_ in enumerate(outputs):
        oname = prefix + o_.name
        k2o_logger().debug('output: ' + oname)
        o1 = varset.get_local_variable_or_declare_one(oname, infer_variable_type(o_, varset.target_opset))
        operator.add_output(o1)

    for i_ in inputs:
        iname = prefix + i_.name
        k2o_logger().debug('input : ' + iname)
        var_type = adjust_input_batch_size(infer_variable_type(i_, varset.target_opset))
        i0 = varset.get_local_variable_or_declare_one(iname, var_type)
        operator.add_input(i0)

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator
