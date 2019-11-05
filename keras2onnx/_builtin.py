###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import sys
import numbers
import numpy as np
from onnx import numpy_helper, mapping
from .common.onnx_ops import apply_identity, apply_reshape, OnnxOperatorBuilder
from .funcbook import converter_func, set_converters
from .proto import onnx_proto
from .proto.tfcompat import tensorflow


class TYPES:
    # tf-node types:
    Identity = 'Identity'
    Const = 'Const'
    Any = 'Any'
    All = 'All'
    Cast = 'Cast'
    Round = 'Round'
    StridedSlice = 'StridedSlice'
    TopKV2 = 'TopKV2'

    # converter internal types:
    TD_Reshape = '_reshape_timedistributed'


def _is_tensor_const(tensor):
    return tensor.op.type in ["Const", "ConstV2"]


def _cal_tensor_value(tensor):  # type: (tensorflow.Tensor)->np.ndarray
    node = tensor.op
    assert _is_tensor_const(tensor), "{} has to be a constant".format(node.name)
    make_ndarray = tensorflow.make_ndarray
    np_arr = make_ndarray(node.get_attr("value"))
    return np_arr


def _to_onnx_type(dt_type):
    # TensorFlow data types intergrate seamlessly with numpy
    return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dt_type.as_numpy_dtype)]


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


@converter_func(TYPES.Identity)
def convert_tf_identity(scope, operator, container):
    default_convert(scope, operator, container)


@converter_func(TYPES.Const)
def convert_tf_const(scope, operator, container):
    node = operator.raw_operator
    np_arr = _cal_tensor_value(node.outputs[0])
    onnx_tensor = numpy_helper.from_array(np_arr, node.outputs[0].name)
    container.add_initializer_from_tensor(onnx_tensor)


@converter_func(TYPES.TD_Reshape)
def convert_reshape_timedistributed(scope, operator, container):
    target_shape = operator.get_attr('target_shape')
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.full_name, desired_shape=target_shape)


@converter_func(TYPES.All, TYPES.Any)
def convert_tf_any_all(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = _cal_tensor_value(node.inputs[1]).tolist()
    axis = [axis] if np.isscalar(axis) else axis

    # It is fine to have nagative reduce_dim.
    cast_op = oopb.apply_cast(operator.input_full_names[0],
                              to=onnx_proto.TensorProto.FLOAT,
                              name=operator.full_name + '_cast')
    keepdims = node.get_attr("keep_dims")
    op_type = "ReduceMin" if node.type == "All" else "ReduceSum"
    reduce_op = oopb.add_node(op_type, cast_op,
                              axes=axis,
                              keepdims=keepdims,
                              name=operator.full_name + '_reduce')

    oopb.add_node_with_output("Greater",
                              [reduce_op, np.array(0, dtype=np.float32)],
                              operator.output_full_names,
                              name=operator.full_name,
                              op_version=9)


@converter_func(TYPES.Round)
def convert_tf_round(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    if operator.target_opset < 11:
        add_output_name = oopb.add_node('Add',
                                        [operator.inputs[0].full_name,
                                         ('_add', oopb.float, np.array(-0.5, dtype=np.float32))
                                        ],
                                        operator.inputs[0].full_name + '_add')
        cast_0 = oopb.add_node('Cast',
                               add_output_name,
                               operator.inputs[0].full_name + '_0_cast', to=onnx_proto.TensorProto.FLOAT)
        oopb.add_node_with_output("Ceil",
                                  cast_0,
                                  operator.output_full_names,
                                  name=operator.full_name)
    else:
        oopb.add_node_with_output("Round",
                                  operator.input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name)


@converter_func(TYPES.TopKV2)
def convert_tf_topkv2(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    cast_0 = oopb.add_node('Cast',
                           operator.inputs[0].full_name,
                           operator.inputs[0].full_name + '_0_cast', to=onnx_proto.TensorProto.FLOAT)
    cast_1 = oopb.add_node('Cast',
                           operator.inputs[1].full_name,
                           operator.inputs[1].full_name + '_1_cast', to=onnx_proto.TensorProto.INT64)
    unsqueeze = oopb.add_node('Unsqueeze',
                              cast_1,
                              operator.inputs[1].full_name + '_unsqueeze', axes=[0])
    oopb.add_node_with_output("TopK",
                              [cast_0, unsqueeze],
                              operator.output_full_names,
                              name=operator.full_name)


@converter_func(TYPES.Cast)
def convert_tf_case(scope, operator, container):
    node = operator.raw_operator
    to = _to_onnx_type(node.get_attr("DstT"))
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output("apply_cast",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              to=to)


def _process_begin_end(new_begin, new_end, stride):
    if stride >= 0:
        new_begin.append(0)
        new_end.append(sys.maxsize)
    else:
        new_begin.append(-1)
        new_end.append(-sys.maxsize)


def _prepare_StridedSlice(node, target_opset):
    max_size = sys.maxsize
    begin = _cal_tensor_value(node.inputs[1]) if _is_tensor_const(node.inputs[1]) else [0] * node.inputs[1].shape[0]
    end = _cal_tensor_value(node.inputs[2]) if _is_tensor_const(node.inputs[2]) else [max_size] * \
                                                                              node.inputs[2].shape[0]
    strides = _cal_tensor_value(node.inputs[3]) if _is_tensor_const(node.inputs[3]) else [1] * node.inputs[3].shape[0]
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask if end_mask is not None else 0
    new_axis_mask = node.get_attr("new_axis_mask")
    new_axis_mask = new_axis_mask if new_axis_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask if shrink_axis_mask is not None else 0
    ellipsis_mask = node.get_attr("ellipsis_mask")
    ellipsis_mask = ellipsis_mask if ellipsis_mask is not None else 0
    extra_mask = new_axis_mask or shrink_axis_mask or ellipsis_mask
    new_begin = []
    new_end = []
    axes = []
    steps = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    ellipsis_gap = 0
    data_input_shape = node.inputs[0].shape
    for idx, begin_item in enumerate(begin):
        if target_opset < 10 and strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 are supported, current stride =" + str(strides[idx]))

        if (ellipsis_mask >> idx) & 1:
            input_shape = node.inputs[0].shape # ctx.get_shape(node.input[0])
            if input_shape is None:
                raise ValueError("StridedSlice op {} requires the shape of input".format(node.name))
            ellipsis_gap = len(input_shape) - len(begin)
            continue

        end_item = end[idx]
        axes.append(idx + ellipsis_gap)
        steps.append(strides[idx])

        if (begin_mask >> idx) & 1 != 0 and (end_mask >> idx) & 1 != 0:
            _process_begin_end(new_begin, new_end, strides[idx])
            continue

        if begin_item == 0 and end_item == 0:
            _process_begin_end(new_begin, new_end, strides[idx])
            continue

        shrink_mask = (shrink_axis_mask >> idx) & 1
        if shrink_mask != 0:
            shrink_begin = begin_item + data_input_shape[idx].value if begin_item < 0 else begin_item
            new_begin.append(shrink_begin)
            new_end.append(shrink_begin + 1)
            needs_squeeze.append(idx + ellipsis_gap)
            continue

        if (begin_mask >> idx) & 1 != 0:
            new_begin.append(0) if strides[idx] >= 0 else new_begin.append(-1)
            new_end.append(end_item)
            continue

        if (end_mask >> idx) & 1 != 0:
            new_begin.append(begin_item)
            new_end.append(max_size) if strides[idx] >= 0 else new_begin.append(-max_size)
            continue

        new_begin.append(begin_item)
        new_end.append(end_item)

    return new_begin, new_end, axes, steps, needs_squeeze, begin_mask, end_mask, extra_mask, new_axis_mask


@converter_func(TYPES.StridedSlice)
def convert_tf_strided_slice(scope, operator, container):
    node = operator.raw_operator
    new_begin, new_end, axes, steps, needs_squeeze, begin_mask, end_mask, extra_mask, new_axis_mask = _prepare_StridedSlice(
        node, operator.target_opset)
    oopb = OnnxOperatorBuilder(container, scope)

    new_axis_axes = []
    cur_idx = 0
    while new_axis_mask > 0:
        if new_axis_mask & 1:
            new_axis_axes.append(cur_idx)
        new_axis_mask = new_axis_mask >> 1
        cur_idx = cur_idx + 1

    if len(new_axis_axes) > 0:
        new_axis_unsqueeze = oopb.add_node('Unsqueeze',
                                           operator.inputs[0].full_name,
                                           operator.inputs[0].full_name + '_unsqueeze',
                                           axes=new_axis_axes)
    else:
        new_axis_unsqueeze = operator.inputs[0].full_name

    if operator.target_opset < 10:
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
        cropped_tensor_name = oopb.add_node('Slice',
                                            new_axis_unsqueeze,
                                            operator.inputs[0].full_name + '_cropping',
                                            starts=new_begin, ends=new_end, axes=axes)
    else:
        if extra_mask or begin_mask:
            cast_node_begin = True
        else:
            start_cast = oopb.add_node('Cast',
                                       operator.inputs[1].full_name,
                                       operator.inputs[1].full_name + '_start_cast', to=7)
            cast_node_begin = False

        if extra_mask or end_mask:
            cast_node_end = True
        else:
            end_cast = oopb.add_node('Cast',
                                     operator.inputs[2].full_name,
                                     operator.inputs[2].full_name + '_end_cast', to=7)
            cast_node_begin = False

        cropped_tensor_name = oopb.add_node('Slice',
                                            [new_axis_unsqueeze,
                                             ('_start', oopb.int64, np.array(new_begin, dtype=np.int64)) if cast_node_begin else start_cast,
                                             ('_end', oopb.int64, np.array(new_end, dtype=np.int64)) if cast_node_end else end_cast,
                                             ('_axes', oopb.int64, np.array(axes, dtype=np.int64)),
                                             ('_steps', oopb.int64, np.array(steps, dtype=np.int64))
                                             ],
                                            operator.inputs[0].full_name + '_cropping')

    if needs_squeeze:
        oopb.add_node_with_output('Squeeze',
                                   cropped_tensor_name,
                                   operator.output_full_names,
                                   operator.inputs[0].full_name + '_squeeze',
                                   axes=needs_squeeze)
    else:
        oopb.add_node_with_output('Identity',
                                  cropped_tensor_name,
                                  operator.output_full_names,
                                  operator.inputs[0].full_name + '_identity')


direct_ops = {"Abs": ("apply_abs",),
              "Acos": 7,
              "Acosh": 9,
              "Add": ("apply_add",),
              "Asin": 7,
              "Asinh": 9,
              "Atan": 7,
              "Atanh": 9,
              "Ceil": ("apply_ceil",),
              "Cos": 7,
              "Cosh": 9,
              "Div": ("apply_div",),
              "Elu": ("apply_elu",),
              "Exp": ("apply_exp",),
              "Floor": ("apply_floor",),
              "Log": ("apply_log",),
              "Mul": ("apply_mul",),
              "Neg": ("apply_neg",),
              "Pow": ("apply_pow",),
              "Reciprocal": ("apply_reciprocal",),
              "Relu": ("apply_relu",),
              "Sigmoid": ("apply_sigmoid",),
              "Sin": 7,
              "Sinh": 9,
              "Softplus": 1,
              "Softsign": 1,
              "Sqrt": ("apply_sqrt",),
              "Sub": ("apply_sub",),
              "Tan": 7,
              "Tanh": ("apply_tanh",)
              }


def tf_op_convert(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    type = operator.raw_operator.type
    item = direct_ops[type]
    assert item is not None, "Can't find the tf op item."
    if isinstance(item, numbers.Integral):
        oopb.add_node_with_output(type,
                                  [var_.full_name for var_ in operator.inputs],
                                  [var_.full_name for var_ in operator.outputs],
                                  name=operator.raw_operator.name,
                                  op_version=item
                                  )
    else:
        apply_func_name = item[0]
        oopb.apply_op_with_output(apply_func_name,
                                  [var_.full_name for var_ in operator.inputs],
                                  [var_.full_name for var_ in operator.outputs],
                                  name=operator.raw_operator.name,
                                  )


set_converters({k: tf_op_convert for k in direct_ops.keys()})
