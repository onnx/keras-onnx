###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras2onnx
import numpy as np
from keras2onnx.proto import keras
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_onnx_runtime

Sequential = keras.models.Sequential
Model = keras.models.Model

import keras.backend as K

class SimpleAttention(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = keras.initializers.get('glorot_uniform')
        self.bias_initializer = keras.initializers.get('zeros')
        self.activation = keras.activations.get('tanh')
        self.units = 1

    def compute_mask(self, inputs, input_mask=None):
        return None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=None,
                                    constraint=None)
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        attention = K.dot(inputs, self.kernel)
        attention = K.bias_add(attention, self.bias, data_format='channels_last')
        attention = self.activation(attention)
        attention = K.squeeze(attention, axis=2)
        if mask is not None:
            attention = K.exp(attention) * K.cast(mask, K.floatx())
        else:
            attention = K.exp(attention)
        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon())
        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)
    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]

from keras2onnx import set_converter
from keras2onnx.proto import onnx_proto
from keras2onnx.common.onnx_ops import apply_identity
from keras2onnx.common.onnx_ops import OnnxOperatorBuilder

def convert_SimpleAttentionLayer(scope, operator, container):
    op = operator.raw_operator
    kernel = op.get_weights()[0]
    bias = op.get_weights()[1]

    kernel_tensor_name = scope.get_unique_variable_name('kernel')
    container.add_initializer(kernel_tensor_name, onnx_proto.TensorProto.FLOAT, kernel.shape, kernel)
    bias_tensor_name = scope.get_unique_variable_name('bias')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, bias.shape, bias)

    oopb = OnnxOperatorBuilder(container, scope)
    reshape_input = oopb.add_node('Reshape',
                                   [operator.inputs[0].full_name,
                                    ('_start', oopb.int64, np.array([-1, kernel.shape[0]], dtype='int64'))],
                                   operator.inputs[0].full_name + '_input_reshape')

    reshape_kernel = oopb.add_node('Reshape',
                                   [kernel_tensor_name, ('_start', oopb.int64, np.array([kernel.shape[0], -1], dtype='int64'))],
                                   operator.inputs[0].full_name + '_kernel_reshape')

    matmul = oopb.add_node('MatMul',
                           [reshape_input,
                            reshape_kernel],
                           operator.inputs[0].full_name + '_matmul')

    input_shape = oopb.add_node('Shape', [operator.inputs[0].full_name], operator.inputs[0].full_name + '_input_shape')
    input_shape_cast = oopb.add_node('Cast', [input_shape], operator.inputs[0].full_name + '_input_shape_cast', to=6)
    input_shape_split = oopb.add_node('Split', [input_shape_cast], operator.inputs[0].full_name + '_input_shape_split', outputs_num=3, axis=0)
    input_shape_squeeze = oopb.add_node('Squeeze', [input_shape_split[0]], operator.inputs[0].full_name + '_input_shape_squeeze',
                                        axes=[0])
    input_shape_unsqueeze = oopb.add_node('Unsqueeze', [input_shape_squeeze],
                                        operator.inputs[0].full_name + '_input_shape_unsqueeze',
                                        axes=[0])

    input_shape_concat = oopb.add_node('Concat',
                                       [input_shape_unsqueeze,
                                        ('_start', oopb.int32, np.array([op.input_shape[1], 1], dtype='int32'))],
                                       operator.inputs[0].full_name + '_input_shape_concat', axis=0)
    input_shape_cocnat_cast = oopb.add_node('Cast', [input_shape_concat], operator.inputs[0].full_name + '_input_shape_concat_cast', to=7)

    matmul_reshape = oopb.add_node('Reshape',
                           [matmul,
                            input_shape_cocnat_cast],
                           operator.inputs[0].full_name + '_matmul_reshape')

    bias_add = oopb.add_node('Add',
                           [matmul_reshape,
                            bias_tensor_name],
                           operator.inputs[0].full_name + '_bias_add')
    tanh = oopb.add_node('Tanh',
                          [bias_add],
                           operator.inputs[0].full_name + '_tanh')
    tanh_squeeze = oopb.add_node('Squeeze',
                                 [tanh],
                                 operator.inputs[0].full_name + '_tanh_squeeze',
                                 axes=[2])
    tanh_exp = oopb.add_node('Exp',
                             [tanh_squeeze],
                             operator.inputs[0].full_name + '_exp')

    exp_reduce_sum = oopb.add_node('ReduceSum',
                             [tanh_exp],
                             operator.inputs[0].full_name + '_exp_reduce_sum', axes=[1], keepdims=1)
    exp_add = oopb.add_node('Add',
                             [exp_reduce_sum,
                              ('_start', oopb.float, np.array([K.epsilon()], dtype='float32'))],
                             operator.inputs[0].full_name + '_exp_add')
    exp_add_div = oopb.add_node('Div',
                             [tanh_exp,
                              exp_add],
                             operator.inputs[0].full_name + '_exp_add_div')
    exp_reshape = oopb.add_node('Reshape',
                                [exp_add_div,
                                 ('_start', oopb.int64, np.array([-1, op.input_shape[1], 1], dtype='int64'))],
                                operator.inputs[0].full_name + '_exp_reshape')
    exp_reshape_mul = oopb.add_node('Mul',
                                [operator.inputs[0].full_name,
                                 exp_reshape],
                                operator.inputs[0].full_name + '_exp_reshape_mul')
    exp_reduce_sum_2 = oopb.add_node('ReduceSum',
                                    [exp_reshape_mul],
                                    operator.inputs[0].full_name + '_exp_reduce_sum_2',
                                   axes=[1],
                                   keepdims=0)
    apply_identity(scope, exp_reduce_sum_2, operator.outputs[0].full_name, container)

set_converter(SimpleAttention, convert_SimpleAttentionLayer)

class TestSimpleAttention(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_SimpleAttention(self):
        keras_model = Sequential()
        keras_model.add(SimpleAttention(input_shape=(10, 20)))
        x = np.random.rand(4, 10, 20).astype(np.float32)
        expected = keras_model.predict(x)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
