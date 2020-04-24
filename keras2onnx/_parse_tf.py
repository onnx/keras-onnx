###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import re
import tensorflow as tf
from onnxconverter_common import Int32TensorType, Int64TensorType, FloatTensorType, DoubleTensorType, BooleanTensorType
from .common import k2o_logger, get_default_batch_size
from .funcbook import get_converter

from .proto import keras
from .proto.tfcompat import normalize_tensor_shape, is_subclassed
from .ke2onnx import keras_layer_spec
from ._consts import TYPES
from ._tf_utils import is_placeholder_node, tsname_to_node


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


def _get_layer_name(reserved, ts_or_op):
    return ts_or_op.rsplit('/', 1)[0]


def _get_input_mask(layer):
    # type: (keras.models.Layer) -> []
    if hasattr(layer, 'input_mask') and layer.input_mask is not None:
        return layer.input_mask if isinstance(layer.input_mask, (list, tuple)) else [layer.input_mask]
    return []


def _get_output_mask(layer):
    # type: (keras.models.Layer) -> []
    if hasattr(layer, 'output_mask') and layer.output_mask is not None:
        return layer.output_mask if isinstance(layer.output_mask, (list, tuple)) else [layer.output_mask]
    return []


class LayerInfo(object):
    def __init__(self, _ly):
        self.layer = _ly
        self.inputs = []
        self.outputs = []
        self.nodelist = []

    @staticmethod
    def create_single_node(node, visited):
        info = LayerInfo(None)
        info.inputs = list(node.inputs)
        info.outputs = list(node.outputs)
        info.nodelist = [node]

        # const_nodes = [ts_.op for ts_ in info.inputs if ts_.op.type == "Const" and ts_.op not in visited]
        # info.nodelist.extend(const_nodes)
        # info.inputs = [ts_ for ts_ in info.inputs if ts_.op not in info.nodelist]
        return info

    @staticmethod
    def create(node, layer, outputs_map, inference_nodeset):
        graph = node.graph
        layer_info = LayerInfo(layer)
        # find the output
        next_itr = set()
        if node.type == TYPES.Identity:  # the case on not subclassing model
            fstr_list, fx_list = (None, None)
        else:
            fstr_list, fx_list = keras_layer_spec(type(layer))
        fx_layer_name = _get_layer_name
        if fstr_list is not None:
            fx_layer_name = fx_list[0]
        layer_name = fx_layer_name(fstr_list, node.name)
        for nn_, layer_info_ in outputs_map.items():
            if layer_info_[0] == layer and fx_layer_name(fstr_list, nn_) == layer_name:
                op_node = graph.get_operation_by_name(tsname_to_node(nn_))
                next_itr.add(op_node)
                layer_info.outputs.extend(op_node.outputs)

        visited = set()
        while next_itr:
            visited |= next_itr
            next_itr.clear()
            for n_ in visited:
                for i_ in n_.inputs:
                    # in layer_spec model, the layer name will be checked
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
    subclassed = not (model._is_graph_network or  # pylint:disable=protected-access
                      isinstance(model, keras.engine.sequential.Sequential))
    if subclassed:
        return True

    def subclassed_layer(layer):
        if hasattr(layer, 'layers'):
            if any(is_subclassed(l_) for l_ in layer.layers):
                return True
            for l_ in layer.layers:
                if subclassed_layer(l_):
                    return True
        return False

    return subclassed_layer(model)


def _get_layers(tf_utils, layer):
    if hasattr(layer, 'layers'):
        return layer.layers
    if hasattr(layer, 'submodules'):
        sub_layers = layer.submodules
        if len(sub_layers) == 0:
            return None
        return sub_layers[0].layers if isinstance(sub_layers[0], tf_utils.ListWrapper) else sub_layers
    return None


def _layer_name_dict(tf_utils, layer, prefix, parent=None):
    output_dict = {}
    sub_layers = layer if isinstance(layer, list) else _get_layers(tf_utils, layer)

    if sub_layers is not None:
        for l_ in sub_layers:
            if isinstance(l_, list):
                submodel_dict = _layer_name_dict(tf_utils, l_, prefix, layer)
            else:
                prefix_l = "{}/{}".format(prefix, l_.name)
                submodel_dict = _layer_name_dict(tf_utils, l_, prefix_l, layer)
            output_dict.update(submodel_dict)

    output_dict[prefix] = (layer, parent)
    return output_dict


def _to_tf_ops(layer_name, fstr, ops_table):
    ops = []
    op_name = fstr.format(layer_name) if fstr is not None else None
    if op_name is None:
        return ops

    if re.match(r".+_\d+$", layer_name):  # if layer name already numbered, skipped then.
        return ops

    idx = 0
    while True:
        op_name = fstr.format("%s_%d" % (layer_name, idx + 1))
        if op_name in ops_table:
            ops.append(ops_table[op_name])
        else:
            break
        idx += 1

    return ops


def build_layer_outputs(model, graph, outputs):
    # type: (keras.Model, tf.Graph, []) -> {}

    from tensorflow.python.training.tracking import data_structures as tf_utils
    output_dict = {}
    layer_dict = _layer_name_dict(tf_utils, model, model.name)

    ops_table = {op_.name: op_ for op_ in graph.get_operations()}

    def add_output_node(graph, op, fx_list, layer_name):
        output_node_name = op.name
        if len(fx_list) > 1:  # if there is no output node function.
            # fx_[1] is output node redirect function.
            output_node_name = fx_list[1](lobj, op)
        assert graph.get_operation_by_name(output_node_name) is not None, "Parsing layer({}) failed.".format(lobj)
        if output_node_name not in output_dict:  # if there is already a same kind of layer, not overwrite it.
            output_dict[output_node_name] = layer_dict[layer_name]

    for ln_, layer_info_ in layer_dict.items():
        lobj = layer_info_[0]
        fstr_list, fx_list = keras_layer_spec(type(lobj))
        if fstr_list is None:
            continue

        for fstr_ in fstr_list:
            op_name = fstr_.format(ln_)
            if op_name not in ops_table:
                continue
            add_output_node(graph, ops_table[op_name], fx_list, ln_)

    # now process the case when a layer was re-used several times in one model.
    for ln_, layer_info_ in layer_dict.items():
        lobj = layer_info_[0]
        fstr_list, fx_list = keras_layer_spec(type(lobj))
        if fstr_list is None:
            continue

        for fstr_ in fstr_list:
            for op_ in _to_tf_ops(ln_, fstr_, ops_table):
                add_output_node(graph, op_, fx_list, ln_)

    return output_dict


def extract_outputs_from_subclassing_model(model, output_dict, input_names, output_names):
    from tensorflow.python.keras.saving import saving_utils as _saving_utils
    from tensorflow.python.util import object_identity
    from ._graph_cvt import convert_variables_to_constants_v2 as _convert_to_constants

    function = _saving_utils.trace_model_call(model)
    concrete_func = function.get_concrete_function()
    output_names.extend([ts_.name for ts_ in concrete_func.outputs])
    output_dict.update(build_layer_outputs(model, concrete_func.graph, concrete_func.outputs))
    graph_def, converted_input_indices = _convert_to_constants(
        concrete_func, lower_control_flow=True)
    input_tensors = concrete_func.graph.internal_captures
    converted_inputs = object_identity.ObjectIdentitySet(
        [input_tensors[index] for index in converted_input_indices])
    input_names.extend([
        tensor.name for tensor in concrete_func.inputs if tensor not in converted_inputs])

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

        for ts_ in _get_output_mask(model):
            if ts_ is not None:
                output_dict[ts_.op.name] = (model, model)

    return output_dict


def build_layer_output_from_model(model, output_dict, input_names, output_names):
    if is_subclassing(model):
        tf.compat.v1.enable_tensor_equality()  # re-enable tensor tensor equality for subclassing model.
        return extract_outputs_from_subclassing_model(model, output_dict, input_names, output_names)
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

    if prefix is None:  # prefix is designed for the distinguish among the shared model instances.
        prefix = ''

    input_masks = _get_input_mask(layer)
    output_masks = _get_output_mask(layer)
    for o_ in layer_info.outputs:
        if o_ not in output_masks:  # the layer converter will handle output_mask by itself.
            oname = prefix + o_.name
            k2o_logger().debug('output: ' + oname)
            o1 = varset.get_local_variable_or_declare_one(oname, infer_variable_type(o_, varset.target_opset))
            operator.add_output(o1)

    for i_ in layer_info.inputs:
        if i_ not in input_masks:  # the layer converter will handle input_mask by itself.
            iname = prefix + i_.name
            k2o_logger().debug('input : ' + iname)
            var_type = adjust_input_batch_size(infer_variable_type(i_, varset.target_opset))
            i0 = varset.get_local_variable_or_declare_one(iname, var_type)
            operator.add_input(i0)

    for om_ in [m_ for m_ in output_masks if m_ is not None]:
        mts_name = prefix + om_.name
        k2o_logger().debug('output mask: ' + mts_name)
        mts_var = varset.get_local_variable_or_declare_one(mts_name, infer_variable_type(om_, varset.target_opset))
        operator.add_output_mask(mts_var)

    for im_ in [m_ for m_ in input_masks if m_ is not None]:
        mts_name = im_.name  # input mask in a shared model is not supported yet, why is it needed?
        k2o_logger().debug('input mask: ' + mts_name)
        mts_var = varset.get_local_variable_or_declare_one(mts_name, infer_variable_type(im_, varset.target_opset))
        operator.add_input_mask(mts_var)

    if hasattr(layer, 'mask_value') and layer.mask_value is not None:
        operator.mask_value = layer.mask_value

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    # in some cases, some constants will be used by an operator outside of this layer.
    for nd_ in layer_info.nodelist:
        if nd_.type == 'Const' and nd_.name not in varset.variable_name_mapping:
            operator = varset.declare_local_operator(nd_.type, raw_model=nd_, op_name=nd_.name)
            o1 = varset.get_local_variable_or_declare_one(nd_.name,
                                                          infer_variable_type(nd_.outputs[0], varset.target_opset))
            operator.add_output(o1)

    return operator
