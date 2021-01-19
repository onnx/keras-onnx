# SPDX-License-Identifier: Apache-2.0

import collections
import numbers
from ..common import cvtfunc
from ..proto import keras, is_tf_keras, is_tensorflow_later_than
from . import simplernn, gru, lstm

LSTM_CLASSES = {keras.layers.LSTM}
GRU_CLASSES = {keras.layers.GRU}


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

    if is_tf_keras and is_tensorflow_later_than("1.14.0"):
        # Add the TF v2 compatability layers (available after TF 1.14)
        from tensorflow.python.keras.layers import recurrent_v2
        LSTM_CLASSES.add(recurrent_v2.LSTM)
        GRU_CLASSES.add(recurrent_v2.GRU)

    if op_type in LSTM_CLASSES:
        lstm.convert_keras_lstm(scope, operator, container, bidirectional)
    elif op_type in GRU_CLASSES:
        gru.convert_keras_gru(scope, operator, container, bidirectional)
    elif op_type == keras.layers.SimpleRNN:
        simplernn.convert_keras_simple_rnn(scope, operator, container, bidirectional)
    else:
        raise ValueError('Unsupported class for Bidirectional layer: {}'.format(op_type))
