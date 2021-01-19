# SPDX-License-Identifier: Apache-2.0

from onnxconverter_common.oopb import OnnxOperatorBuilder
from .funcbook import converter_func
from ._tf_utils import tf_attrs_to_onnx as _to_onnx_attrs
from ._tf_utils import cal_tensor_shape as _cal_tensor_shape


def _random_converter(scope, operator, container):
    tf_op = operator.raw_operator
    op_type = tf_op.type
    if op_type == 'RandomStandardNormal':
        op_type = 'RandomNormalLike'
    else:
        op_type = op_type + 'Like'
    inputs = [var_.full_name for var_ in operator.inputs]

    attrs = {}
    if 'seed2' in _to_onnx_attrs(tf_op):
        attrs['seed'] = float(tf_op.get_attr('seed2'))
    with OnnxOperatorBuilder(container, scope).as_default(tf_op.name) as oopb:
        cast_n = oopb.cast(inputs, to=oopb.int64)
        const_op = oopb.add_node('ConstantOfShape', cast_n, op_version=9)
        oopb.add_node(op_type, const_op,
                      outputs=[var_.full_name for var_ in operator.outputs],
                      op_version=1, **attrs)


@converter_func(
    'RandomNormal',
    'RandomStandardNormal',
    'RandomUniform')
def convert_tf_random_standard_normal(scope, operator, container):
    _random_converter(scope, operator, container)


@converter_func('Select', 'SelectV2')
def convert_tf_select(scope, operator, container):
    tf_op = operator.raw_operator
    shape_i0 = _cal_tensor_shape(tf_op.inputs[0])
    target_shape = _cal_tensor_shape(tf_op.inputs[1])
    if len(target_shape) == 0:
        target_shape = _cal_tensor_shape(tf_op.inputs[2])
    input0 = operator.input_full_names[0]
    with OnnxOperatorBuilder(container, scope).as_default(operator.full_name) as oopb:  # type: OnnxOperatorBuilder
        if len(shape_i0) == 1 and len(target_shape) > 1:
            input0 = oopb.unsqueeze(input0, axes=list(range(len(target_shape)))[1:])
        oopb.add_node("Where", [input0] + operator.input_full_names[1:],
                      outputs=operator.output_full_names,
                      op_version=9)


@converter_func('LogicalNot', 'LogicalAnd', 'LogicalOr')
def convert_tf_logical_ops(scope, operator, container):
    onnx_type = operator.type[len('Logical'):]
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.add_node(onnx_type,
                  operator.input_full_names,
                  name=operator.full_name,
                  outputs=operator.output_full_names,
                  op_version=1)


def pass_thru_converter(scope, operator, container):
    """
    This converter is to copy the original graph node with its def into a ONNX node format.
    """
    tf_op = operator.raw_operator
    attrs = _to_onnx_attrs(tf_op)

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.add_node(operator.type,
                  operator.input_full_names,
                  name=operator.full_name,
                  outputs=operator.output_full_names,
                  op_domain='ai.onnx.contrib',
                  op_version=1,
                  **attrs)
