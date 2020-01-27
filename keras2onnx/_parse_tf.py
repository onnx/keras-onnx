###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import tensorflow as tf
from onnxconverter_common import Int32TensorType, Int64TensorType, FloatTensorType, DoubleTensorType, BooleanTensorType
from .common import k2o_logger, get_default_batch_size
from .funcbook import get_converter

from .proto import keras
from .proto.tfcompat import normalize_tensor_shape
from .ke2onnx import keras_layer_spec
from ._builtin import TYPES, is_placeholder_node, tsname_to_node


def infer_variable_type(tensor, opset, inbound_node_shape=None):
    tensor_shape = []
    if inbound_node_shape is None:
        if tensor.shape not in (tf.TensorShape(None), tf.TensorShape([])):
            if opset > 8:
                tensor_shape = normalize_tensor_shape(tensor.shape)
            else:  # most inference engine has problem with unset dim param if they released around opset 8 publish
                tensor_shape = ['None' if d is None else d for d in normalize_tensor_shape(tensor.shape)]
    else:
        tensor_shape = list(inbound_node_shape)

    # Determine the tensor's element type
    tensor_type = tensor.dtype.base_dtype
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
        next_itr = set()
        if node.type == TYPES.Identity:  # the case on not subclassing model
            fstr_list, func_c = (None, None)
        else:
            fstr_list, func_c = keras_layer_spec(type(layer))
        layer_name = _get_layer_name(node.name)
        if fstr_list is not None:
            layer_name = func_c(fstr_list, node.name)
        for nn_, layer_info_ in outputs_map.items():
            if layer_info_[0] == layer and _get_layer_name(nn_) == layer_name:
                op_node = graph.get_operation_by_name(tsname_to_node(nn_))
                next_itr.add(op_node)
                layer_info.outputs.extend(op_node.outputs)

        visited = set()
        while next_itr:
            visited |= next_itr
            next_itr.clear()
            for n_ in visited:
                for i_ in n_.inputs:
                    if fstr_list is not None and i_.op.name.find(layer_name) == -1:
                        continue
                    if i_.op in visited or i_.op not in inference_nodeset:
                        continue
                    if (not is_placeholder_node(i_.op)) and i_.op.name in outputs_map:
                        continue
                    next_itr.add(i_.op)

        layer_info.nodelist = list(visited)
        return layer_info


def is_subclassing(model):
    return not (model._is_graph_network or  # pylint:disable=protected-access
                isinstance(model, keras.engine.sequential.Sequential))


def _get_layers(tf_utils, layer):
    if hasattr(layer, '_layers'):
        sub_layers = layer._layers
        if len(sub_layers) == 0:
            return None
        return sub_layers[0].layers if isinstance(sub_layers[0], tf_utils.ListWrapper) else sub_layers
    if hasattr(layer, 'layers'):
        return layer.layers
    return None


def layer_name_dict(tf_utils, layer, prefix, parent=None):
    output_dict = {}
    sub_layers = layer if isinstance(layer, list) else _get_layers(tf_utils, layer)

    if sub_layers is not None:
        for l_ in sub_layers:
            if isinstance(l_, list):
                submodel_dict = layer_name_dict(tf_utils, l_, prefix, layer)
            else:
                prefix_l = "{}/{}".format(prefix, l_.name)
                submodel_dict = layer_name_dict(tf_utils, l_, prefix_l, layer)
            output_dict.update(submodel_dict)

    output_dict[prefix] = (layer, parent)
    return output_dict


def _is_input_output_tensor(ts):
    return ts.name.find('/') == -1 or ts.op.type == 'ReadVariableOp'


def build_layer_outputs(model, graph, outputs):
    # type: (keras.Model, tf.Graph, []) -> {}

    from tensorflow.python.training.tracking import data_structures as tf_utils
    output_dict = {}
    layer_dict = layer_name_dict(tf_utils, model, model.name)

    for op_ in graph.get_operations():
        if op_.name in output_dict:
            continue
        orig_layer_name = _get_layer_name(op_.name)
        layer_name = orig_layer_name
        if layer_name not in layer_dict:
            layer_name = layer_name.rsplit('_', 1)[0]

        # assert layer_name in layer_dict, "Cannot find the Keras layer of the output tensor({}).".format(ou_.name)
        if layer_name in layer_dict:
            lobj, _ = layer_dict[layer_name]
            fstr_list, _ = keras_layer_spec(type(lobj))
            if fstr_list is None:
                continue

            for fstr in fstr_list:
                if fstr and fstr.format(orig_layer_name) == op_.name:
                    output_dict[op_.name] = layer_dict[layer_name]

    return output_dict


TF_GRAPH_OPTIMIZATION = True


def extract_outputs_from_subclassing_model(model, output_dict, output_names):
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.python.keras.saving import saving_utils as _saving_utils
    from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
    from tensorflow.python.framework import convert_to_constants as _convert_to_constants

    function = _saving_utils.trace_model_call(model)
    concrete_func = function.get_concrete_function()
    output_names.extend([ts_.name for ts_ in concrete_func.outputs])
    output_dict.update(build_layer_outputs(model, concrete_func.graph, concrete_func.outputs))
    frozen_func = _convert_to_constants.convert_variables_to_constants_v2(
        concrete_func, lower_control_flow=True)
    graph_def = frozen_func.graph.as_graph_def()
    if TF_GRAPH_OPTIMIZATION:
        input_tensors = [
            tensor for tensor in frozen_func.inputs
            if tensor.dtype != tf.dtypes.resource
        ]
        output_tensors = frozen_func.outputs
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


def extract_outputs_from_inbound_nodes(model):
    output_dict = {}
    if hasattr(model, 'layers'):
        for l_ in model.layers:
            output_dict.update(extract_outputs_from_inbound_nodes(l_))

    if hasattr(model, 'inbound_nodes'):
        for nd_ in model.inbound_nodes:
            output_tensors = [nd_.output_tensors] if hasattr(nd_.output_tensors, 'dtype') else \
                nd_.output_tensors
            for ts_ in output_tensors:
                op_name = tsname_to_node(ts_.name)
                if op_name not in output_dict:
                    output_dict[op_name] = (model, None)

    return output_dict


def build_layer_output_from_model(model, output_dict, output_names):
    if is_subclassing(model):
        return extract_outputs_from_subclassing_model(model, output_dict, output_names)
    else:
        graph = model.outputs[0].graph
        output_names.extend([n.name for n in model.outputs])
        output_dict.update(extract_outputs_from_inbound_nodes(model))
        return graph


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
