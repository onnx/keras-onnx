###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import collections
import numbers
import numpy as np
from ..common import cvtfunc, name_func
from ..common.onnx_ops import apply_transpose, OnnxOperatorBuilder
from ..proto import onnx_proto, keras
from . import simplernn, lstm

LSTM = keras.layers.LSTM
TensorProto = onnx_proto.TensorProto


def _calculate_keras_bidirectional_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], collections.abc.Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None
                                              for i in op.output_shape[0])
        if op.merge_mode is None:
            operator.outputs[1].type.shape = list(i if isinstance(i, numbers.Integral) else None
                                                  for i in op.output_shape[1])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None for i in op.output_shape)


@cvtfunc(shape_infer=_calculate_keras_bidirectional_output_shapes)
def convert_bidirectional(scope, operator, container):
    op = operator.raw_operator
    forward_layer = op.forward_layer

    lstm.check_sequence_lengths(operator, container)

    _name = name_func(scope, operator)

    # Inputs
    lstm_x = _name('X')
    tensor_w, tensor_r, tensor_b = lstm.build_parameters(scope, operator, container, bidirectional=True)
    sequence_lengths = simplernn.build_sequence_lengths(scope, operator, container)
    initial_h, initial_c = lstm.build_initial_states(scope, operator, container, bidirectional=True)

    input_names = [
        lstm_x,
        tensor_w,
        tensor_r,
        tensor_b,
        sequence_lengths,
        initial_h,
        initial_c,
        '',  # P (optional) : No peep hole in Keras.
    ]

    # Attributes
    attrs = lstm.build_attributes(scope, operator, container, bidirectional=True)

    # Outputs
    output_names = [_name('Y'), _name('Y_h'), _name('Y_c')]

    # Reshape Keras input format into ONNX input format
    input_name = operator.inputs[0].full_name
    apply_transpose(scope, input_name, lstm_x, container, perm=[1, 0, 2])

    # Create the major node, ONNX LSTM
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_lstm',
                              input_names,
                              output_names,
                              name=op.name,
                              output_seq=forward_layer.return_sequences,
                              **attrs)

    lstm.build_output(scope, operator, container, output_names, bidirectional=True)
