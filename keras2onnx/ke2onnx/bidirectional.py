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
    lstm.convert_keras_lstm(scope, operator, container, bidirectional=True)
