###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from typing import Iterable

from .common import k2o_logger
from .funcbook import get_converter
from ._parser_tf import infer_variable_type, tsname_to_node, adjust_input_batch_size


def extract_inbound_nodes(layer):
    if hasattr(layer, 'inbound_nodes'):
        return layer.inbound_nodes
    elif hasattr(layer, '_inbound_nodes'):
        return layer._inbound_nodes
    else:
        raise ValueError("Failed to find inbound_nodes and _inbound_nodes when parsing %s" % layer.name)


def list_input_tensors(node):
    """
    Since Tensorflow 1.14, sometimes the node.input_tensors may not be a list, though the word is plural.
    """
    return [node.input_tensors] if hasattr(node.input_tensors, 'dtype') else node.input_tensors


def list_output_tensors(node):
    """
    Since Tensorflow 1.14, sometimes the node.output_tensors may not be a list, though the output_tensors is plural.
    """
    return [node.output_tensors] if hasattr(node.output_tensors, 'dtype') else node.output_tensors


def list_input_shapes(node):
    """
    Since Tensorflow 1.14, sometimes the node.input_shapes may not be a list, though the input_shapes is plural.
    """
    return node.input_shapes if isinstance(node.input_shapes[0], Iterable) else [node.input_shapes]


def list_output_shapes(node):
    """
    Since Tensorflow 1.14, sometimes the node.output_shapes may not be a list, though the output_shapes is plural.
    """
    return node.output_shapes if isinstance(node.output_shapes[0], Iterable) else [node.output_shapes]


def list_input_mask(layer):
    if hasattr(layer, 'input_mask'):
        if hasattr(layer.input_mask, 'dtype'):
            return [layer.input_mask]
        if layer.input_mask is not None:
            return [ts_ for ts_ in layer.input_mask if ts_ is not None]

    return []


def list_output_mask(layer):
    if hasattr(layer, 'output_mask'):
        if hasattr(layer.output_mask, 'dtype'):
            return [layer.output_mask]
        if layer.output_mask is not None:
            return [ts_ for ts_ in layer.output_mask if ts_ is not None]

    return []


def on_parsing_keras_layer(graph, node_list, layer, kenode, model, varset, prefix=None):
    operator = varset.declare_local_operator(type(layer), raw_model=layer, op_name=layer.name)
    operator.nodelist = node_list

    inputs = list_input_tensors(kenode)
    outputs = list_output_tensors(kenode)

    # This layer will be visited because its output is other layer's input
    assert len(node_list) == 0 or node_list[0] in [ts_.op for ts_ in outputs]

    if prefix is None:  # prefix is designed for the distinguish among the shared model instances.
        prefix = ''

    kenode_input_shapes = kenode.input_shapes if isinstance(kenode.input_shapes, list) else [kenode.input_shapes]
    for n_, i_ in enumerate(inputs):
        iname = prefix + i_.name
        k2o_logger().debug('input : ' + iname)
        var_type = adjust_input_batch_size(infer_variable_type(i_, varset.target_opset, kenode_input_shapes[n_]))
        i0 = varset.get_local_variable_or_declare_one(iname, var_type)
        operator.add_input(i0)

    if hasattr(layer, 'input_mask') and layer.input_mask is not None:
        in_mask = layer.input_mask if isinstance(layer.input_mask, (list, tuple)) else [layer.input_mask]
        for im_ in [m_ for m_ in in_mask if m_ is not None]:
            mts_name = im_.name  # input mask in a shared model is not supported yet, why is it needed?
            k2o_logger().debug('input mask: ' + mts_name)
            mts_var = varset.get_local_variable_or_declare_one(mts_name, infer_variable_type(im_, varset.target_opset))
            operator.add_input_mask(mts_var)

    kenode_output_shapes = kenode.output_shapes if isinstance(kenode.output_shapes, list) else [kenode.output_shapes]
    for n_, o_ in enumerate(outputs):
        oname = prefix + o_.name
        k2o_logger().debug('output: ' + oname)
        o1 = varset.get_local_variable_or_declare_one(oname, infer_variable_type(o_, varset.target_opset, kenode_output_shapes[n_]))
        operator.add_output(o1)

    if hasattr(layer, 'output_mask') and layer.output_mask is not None:
        out_mask = layer.output_mask if isinstance(layer.output_mask, (list, tuple)) else [layer.output_mask]
        for om_ in [m_ for m_ in out_mask if m_ is not None]:
            mts_name = prefix + om_.name
            k2o_logger().debug('output mask: ' + mts_name)
            mts_var = varset.get_local_variable_or_declare_one(mts_name, infer_variable_type(om_, varset.target_opset))
            operator.add_output_mask(mts_var)

    if hasattr(layer, 'mask_value') and layer.mask_value is not None:
        operator.mask_value = layer.mask_value

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator


def build_opdict_from_keras(model):
    # type: (keras.Model) -> {}

    output_dict = {}
    for l_ in model.layers:
        if hasattr(l_, 'layers'):
            submodel_dict = build_opdict_from_keras(l_)
            shared_layer = False
            for node_ in extract_inbound_nodes(l_):
                shared_layer |= any(
                    ts_.name not in submodel_dict for ts_ in list_output_tensors(node_))
                if shared_layer:
                    break
            if not shared_layer:  # shared layer(model) will be processed as a whole.
                output_dict.update(submodel_dict)
                continue

        for node_ in extract_inbound_nodes(l_):
            for ts_ in list_output_tensors(node_):
                output_dict[ts_.name] = (l_, model)

        for ts_ in list_output_mask(l_):
            output_dict[ts_.name] = (l_, model)

    return {tsname_to_node(n_): v_ for n_, v_ in output_dict.items()}
