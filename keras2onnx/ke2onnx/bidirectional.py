###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import collections
import numbers
from ..common import cvtfunc
from ..proto import keras
from . import simplernn, gru, lstm


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
    op_type = type(operator.raw_operator.forward_layer)
    bidirectional = True

    if op_type == keras.layers.LSTM:
        lstm.convert_keras_lstm(scope, operator, container, bidirectional)
    elif op_type == keras.layers.GRU:
        gru.convert_keras_gru(scope, operator, container, bidirectional)
    elif op_type == keras.layers.SimpleRNN:
        simplernn.convert_keras_simple_rnn(scope, operator, container, bidirectional)
    else:
        raise ValueError('Unsupported class for Bidirectional layer: {}'.format(op_type))
