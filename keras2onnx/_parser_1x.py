###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .common import k2o_logger, get_default_batch_size
from .funcbook import get_converter
from .ke2onnx import list_input_tensors, list_output_tensors
from ._parse_tf import _infer_variable_type


def adjust_input_batch_size(var_type):
    if len(var_type.shape) > 0 and var_type.shape[0] is None:
        var_type.shape = [get_default_batch_size()] + var_type.shape[1:]
    return var_type


def _on_parsing_keras_layer(graph, node_list, layer, kenode, model, varset, prefix=None):
    operator = varset.declare_local_operator(type(layer), raw_model=layer, op_name=layer.name)
    operator.nodelist = node_list

    inputs = list_input_tensors(kenode)
    outputs = list_output_tensors(kenode)

    # This layer will be visited because its output is other layer's input
    assert len(node_list) == 0 or node_list[0] in [ts_.op for ts_ in outputs]

    if prefix is None:  # prefix is designed for the distinguish among the shared model instances.
        prefix = ''

    for i_ in inputs:
        iname = prefix + i_.name
        k2o_logger().debug('input : ' + iname)
        var_type = adjust_input_batch_size(_infer_variable_type(i_, varset.target_opset))
        i0 = varset.get_local_variable_or_declare_one(iname, var_type)
        operator.add_input(i0)

    if hasattr(layer, 'input_mask') and layer.input_mask is not None:
        in_mask = layer.input_mask if isinstance(layer.input_mask, (list, tuple)) else [layer.input_mask]
        for im_ in [m_ for m_ in in_mask if m_ is not None]:
            mts_name = im_.name  # input mask in a shared model is not supported yet, why is it needed?
            k2o_logger().debug('input mask: ' + mts_name)
            mts_var = varset.get_local_variable_or_declare_one(mts_name, _infer_variable_type(im_, varset.target_opset))
            operator.add_input_mask(mts_var)

    for n_, o_ in enumerate(outputs):
        oname = prefix + o_.name
        k2o_logger().debug('output: ' + oname)
        o1 = varset.get_local_variable_or_declare_one(oname, _infer_variable_type(o_, varset.target_opset))
        operator.add_output(o1)

    if hasattr(layer, 'output_mask') and layer.output_mask is not None:
        out_mask = layer.output_mask if isinstance(layer.output_mask, (list, tuple)) else [layer.output_mask]
        for om_ in [m_ for m_ in out_mask if m_ is not None]:
            mts_name = prefix + om_.name
            k2o_logger().debug('output mask: ' + mts_name)
            mts_var = varset.get_local_variable_or_declare_one(mts_name, _infer_variable_type(om_, varset.target_opset))
            operator.add_output_mask(mts_var)

    cvt = get_converter(operator.type)
    if cvt is not None and hasattr(cvt, 'shape_infer'):
        operator.shape_infer = cvt.shape_infer

    return operator
