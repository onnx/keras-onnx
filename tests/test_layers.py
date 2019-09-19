###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import unittest
import tensorflow as tf
import keras2onnx
import numpy as np
from keras2onnx.proto import keras, is_tf_keras, get_opset_number_from_onnx, is_keras_older_than, is_keras_later_than
from test_utils import run_onnx_runtime

import importlib

importlib.import_module('test_utils')

K = keras.backend
Activation = keras.layers.Activation
Add = keras.layers.Add
advanced_activations = keras.layers.advanced_activations
Average = keras.layers.Average
AveragePooling1D = keras.layers.AveragePooling1D
AveragePooling2D = keras.layers.AveragePooling2D
AveragePooling3D = keras.layers.AveragePooling3D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
Conv3D = keras.layers.Conv3D
Conv3DTranspose = keras.layers.Conv3DTranspose
Cropping1D = keras.layers.Cropping1D
Cropping2D = keras.layers.Cropping2D
Cropping3D = keras.layers.Cropping3D
Dense = keras.layers.Dense
Dot = keras.layers.Dot
dot = keras.layers.dot
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
GRU = keras.layers.GRU
Input = keras.layers.Input
InputLayer = keras.layers.InputLayer
Lambda = keras.layers.Lambda
Layer = keras.layers.Layer
LSTM = keras.layers.LSTM
Maximum = keras.layers.Maximum
MaxPool1D = keras.layers.MaxPool1D
MaxPool3D = keras.layers.MaxPool3D
MaxPooling2D = keras.layers.MaxPooling2D
Model = keras.models.Model
Multiply = keras.layers.Multiply
Reshape = keras.layers.Reshape
SeparableConv1D = keras.layers.SeparableConv1D
SeparableConv2D = keras.layers.SeparableConv2D
Sequential = keras.models.Sequential
SimpleRNN = keras.layers.SimpleRNN
Subtract = keras.layers.Subtract
TimeDistributed = keras.layers.TimeDistributed
UpSampling1D = keras.layers.UpSampling1D
UpSampling2D = keras.layers.UpSampling2D
UpSampling3D = keras.layers.UpSampling3D
ZeroPadding2D = keras.layers.ZeroPadding2D
if not (is_keras_older_than("2.2.4") or is_tf_keras):
    ReLU = keras.layers.ReLU


class TestKerasTF2ONNX(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @staticmethod
    def asarray(*a):
        return np.array([a], dtype='f')

    def test_keras_with_tf2onnx(self):
        model = Sequential()
        model.add(Dense(units=4, input_shape=(10,), activation='relu'))
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
        graph_def = keras2onnx.export_tf_frozen_graph(model)
        onnx_model = keras2onnx.convert_tensorflow(graph_def, **keras2onnx.build_io_names_tf2onnx(model))
        data = np.random.rand(4 * 10).astype(np.float32).reshape(4, 10)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('ktf2onnx_test', onnx_model, data, expected, self.model_files))

    def test_keras_lambda(self):
        model = Sequential()
        model.add(Lambda(lambda x: x ** 2, input_shape=[3, 5]))
        model.add(Lambda(lambda x: tf.round(x), input_shape=[3, 5]))
        model.add(Flatten(data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')

        _custom_op_handlers = {
            'Round': keras2onnx.main.tf2onnx_builtin_conversion(10)['Round']}
        onnx_model = keras2onnx.convert_keras(model, 'test', custom_op_conversions=_custom_op_handlers)
        data = np.random.rand(3 * 5).astype(np.float32).reshape(1, 3, 5)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('onnx_lambda', onnx_model, data, expected, self.model_files))

    def _test_stridedslice_with_version(self, target_opset):
        for v1 in [-1, 1]:
            for v2 in [-1, 2]:
                model = Sequential()
                model.add(
                    Lambda(lambda x: x[:, tf.newaxis, v1:, tf.newaxis, :v2, tf.newaxis, 3], input_shape=[2, 3, 4, 5]))
                onnx_model = keras2onnx.convert_keras(model, 'test', target_opset=target_opset)

                data = np.random.rand(6 * 2 * 3 * 4 * 5).astype(np.float32).reshape(6, 2, 3, 4, 5)
                expected = model.predict(data)
                self.assertTrue(run_onnx_runtime('onnx_stridedslice', onnx_model, data, expected, self.model_files))

    def _test_stridedslice_ellipsis_mask_with_version(self, target_opset):
        model = Sequential()
        model.add(Lambda(lambda x: x[:, :2, ..., 1:], input_shape=[3, 4, 5, 6, 3]))
        onnx_model = keras2onnx.convert_keras(model, 'test', target_opset=target_opset)

        data = np.random.rand(5 * 3 * 4 * 5 * 6 * 3).astype(np.float32).reshape(5, 3, 4, 5, 6, 3)
        expected = model.predict(data)
        self.assertTrue(
            run_onnx_runtime('onnx_stridedslice_ellipsis_mask', onnx_model, data, expected, self.model_files))

    def _test_stridedslice_shrink_mask_with_version(self, target_opset):
        for shrink_value in [-1, 2]:
            model = Sequential()
            model.add(Lambda(lambda x: x[:, shrink_value, :], input_shape=[3, 4, 5]))
            onnx_model = keras2onnx.convert_keras(model, 'test', target_opset=target_opset)
            data = np.random.rand(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)
            expected = model.predict(data)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_stridedslice(self):
        for opset_ in [9, 10]:
            self._test_stridedslice_with_version(opset_)
            self._test_stridedslice_ellipsis_mask_with_version(opset_)
            self._test_stridedslice_shrink_mask_with_version(opset_)

    def test_any_all(self):
        for l_ in [keras.backend.any, keras.backend.all]:
            for axis in [1, -1]:
                keras_model = Sequential()
                keras_model.add(Lambda(lambda x: l_(x, axis=axis), input_shape=[3, 5]))
                onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
                x = np.random.rand(2, 3, 5).astype(np.float32)
                expected = keras_model.predict(x)
                self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_dense(self):
        for bias_value in [True, False]:
            model = keras.Sequential()
            model.add(Dense(5, input_shape=(4,), activation='sigmoid'))
            model.add(Dense(3, input_shape=(5,), use_bias=bias_value))
            model.compile('sgd', 'mse')
            onnx_model = keras2onnx.convert_keras(model, model.name)

            data = self.asarray(1, 0, 0, 1)
            expected = model.predict(data)
            self.assertTrue(run_onnx_runtime('dense', onnx_model, data, expected, self.model_files))

    def test_dense_add(self):
        input1 = Input(shape=(4,))
        x1 = Dense(3, activation='relu')(input1)
        input2 = Input(shape=(5,))
        x2 = Dense(3, activation='sigmoid')(input2)
        input3 = Input(shape=(3,))
        x3 = Dense(3)(input3)
        added = Add()([x1, x2, x3])  # equivalent to added = add([x1, x2])
        model = keras.models.Model(inputs=[input1, input2, input3], outputs=added)
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = [self.asarray(1.2, 2.4, -2, 1), self.asarray(-1, -2, 0, 1, 2), self.asarray(0.5, 1.5, -3.14159)]
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('onnx_dense_add', onnx_model, data, expected, self.model_files))

    def test_dense_softmax(self):
        data = self.asarray(1, 2, 3, 4)
        model = Sequential()
        model.add(Dense(5, input_shape=(4,), activation='softmax'))
        model.add(Dense(3, input_shape=(5,), use_bias=True))
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('dense_softmax_1', onnx_model, data, expected, self.model_files))

        model = Sequential()
        model.add(Dense(5, input_shape=(4,)))
        model.add(Activation('softmax'))
        model.add(Dense(3, input_shape=(5,), use_bias=True))
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('dense_softmax_2', onnx_model, data, expected, self.model_files))

    def mergelayer_helper(self, keras_layer_type, *data):
        data2 = [self.asarray(*d) for d in data]
        inputs = [Input(shape=d.shape[1:]) for d in data2]
        layer = keras_layer_type()(inputs)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data2)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data2, expected, self.model_files))

    def test_add(self):
        self.mergelayer_helper(Add, [1, 2, 3], [4, 5, 6])
        self.mergelayer_helper(Add, [1, 2, 3], [4, 5, 6], [-3, -1, 1.5])

    def test_sub(self):
        self.mergelayer_helper(Subtract, [1, 2, 3], [4, 5, 6])

    def test_mul(self):
        self.mergelayer_helper(Multiply, [1, 2, 3], [4, 5, 6])

    def test_average(self):
        self.mergelayer_helper(Average, [1, -2, 3], [3, 1, 1])

    def test_max(self):
        self.mergelayer_helper(Maximum, [1, -2, 3], [3, 1, 1])

    def test_concat(self):
        self.mergelayer_helper(lambda: Concatenate(), [1, 2, 3], [4, 5, 6, 7])
        self.mergelayer_helper(lambda: Concatenate(), [1, 2, 3], [4, 5, 6, 7])

    def test_concat_2d(self):
        self.mergelayer_helper(lambda: Concatenate(-1), [[1, 2], [3, 4]], [[4, 5], [6, 7]])
        self.mergelayer_helper(lambda: Concatenate(1), [[1, 2], [3, 4]], [[4, 5], [6, 7]])
        self.mergelayer_helper(lambda: Concatenate(2), [[1, 2], [3, 4]], [[4, 5], [6, 7]])

    def _conv_helper(self, layer_type, input_channels, output_channels, kernel_size, strides, input_size, activation,
                     rtol, atol, bias, channels_first=False, padding='valid'):
        model = keras.Sequential()
        input_size_seq = (input_size,) if isinstance(input_size, int) else input_size
        kwargs = {}
        if channels_first:
            input_shape = (input_channels,) + input_size_seq
            if not isinstance(layer_type, Conv1D):
                kwargs['data_format'] = 'channels_first'
        else:
            input_shape = input_size_seq + (input_channels,)

        model.add(layer_type(output_channels, kernel_size, input_shape=input_shape, strides=strides, padding=padding,
                             dilation_rate=1, activation=activation, use_bias=bias, **kwargs))
        data = np.random.uniform(-0.5, 0.5, size=(1,) + input_shape).astype(np.float32)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files, rtol=rtol, atol=atol))

    def _conv1_helper(self, input_channels, output_channels, kernel_size, strides, input_length, activation=None,
                      rtol=1e-4, atol=1e-6, bias=False, padding='valid'):
        self._conv_helper(Conv1D, input_channels, output_channels, kernel_size, strides, input_length,
                          activation, rtol, atol, bias, padding=padding)

    def test_conv1d(self):
        self._conv1_helper(4, 5, 3, 1, 15)
        self._conv1_helper(4, 5, 3, 2, 15)

    def test_conv1d_padding_same(self):
        self._conv1_helper(4, 5, 3, 1, 15, padding='same')
        # Not sure about 'causal'.

    def test_conv1d_activation(self):
        self._conv1_helper(4, 5, 3, 1, 15, activation='sigmoid')

    def test_conv1d_bias(self):
        self._conv1_helper(4, 5, 3, 1, 15, bias=True)

    def _conv2_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                      rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 2)
        self._conv_helper(Conv2D, input_channels, output_channels, kernel_size, strides, inputs_dims,
                          activation, rtol, atol, bias, channels_first, padding)

    def _conv2trans_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                           rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 2)
        self._conv_helper(Conv2DTranspose, input_channels, output_channels, kernel_size, strides,
                          inputs_dims, activation, rtol, atol, bias, channels_first, padding)

    def test_conv2d(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5))

    def test_conv2d_transpose(self):
        self._conv2trans_helper(3, 5, (2, 2), (1, 1), (5, 5))

    def test_conv2d_padding_same(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), padding='same')
        self._conv2_helper(8, 16, (1, 1), (2, 2), (60, 60), padding='same')
        self._conv2_helper(1, 1, (3, 3), (2, 2), (6, 6), padding='same')
        self._conv2_helper(1, 1, (7, 7), (2, 2), (25, 25), padding='same')
        self._conv2_helper(1, 1, (5, 7), (3, 5), (25, 25), padding='same')

    @unittest.skipIf(is_tf_keras, "Generic conv implementation only supports NHWC tensor format in tf_keras")
    def test_conv2d_format(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), channels_first=True)

    def test_conv2d_activation(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), activation='relu')
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), activation='softmax')

    def test_conv2d_bias(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), bias=True)

    def test_conv2d_larger(self):
        self._conv2_helper(3, 5, (7, 9), 1, (30, 20))

    def test_conv2d_uneven_stride(self):
        self._conv2_helper(3, 5, (4, 4), (3, 2), (20, 10))

    def _conv3_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                      rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 3)
        self._conv_helper(Conv3D, input_channels, output_channels, kernel_size, strides, inputs_dims,
                          activation, rtol, atol, bias, channels_first, padding)

    def test_conv3d(self):
        self._conv3_helper(3, 5, (2, 2, 2), (1, 1, 1), (5, 5, 8))

    def _conv3trans_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                           rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 3)
        self._conv_helper(Conv3DTranspose, input_channels, output_channels, kernel_size, strides,
                          inputs_dims, activation, rtol, atol, bias, channels_first, padding)

    @unittest.skip("ONNXRuntime doesn't support 3D ConvTranspose.")
    def test_conv3d_transpose(self):
        self._conv3trans_helper(3, 5, (2, 2, 2), (1, 1, 1), (5, 5, 8))

    def test_flatten(self):
        model = keras.Sequential()
        model.add(keras.layers.core.Flatten(input_shape=(3, 2)))
        model.add(Dense(3))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('flatten', onnx_model, data, expected, self.model_files))

    def test_flatten2(self):
        C = 3
        H = 5
        W = 7
        for data_format in ['channels_first', 'channels_last']:
            model = keras.Sequential()
            model.add(Conv2D(64, (3, 3),
                             input_shape=(C, H, W), padding='same', ))
            model.add(Flatten(data_format=data_format))
            onnx_model = keras2onnx.convert_keras(model, model.name)
            x = np.random.rand(4, C, H, W).astype(np.float32)
            expected = model.predict(x)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_reshape(self):
        model = keras.Sequential()
        model.add(keras.layers.core.Reshape((2, 3), input_shape=(3, 2)))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('reshape', onnx_model, data, expected, self.model_files))

    def test_permute(self):
        model = keras.Sequential()
        model.add(keras.layers.core.Permute((2, 1), input_shape=(3, 2)))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('permute', onnx_model, data, expected, self.model_files))

    def test_repeat_vector(self):
        model = keras.Sequential()
        model.add(keras.layers.core.RepeatVector(3, input_shape=(4,)))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = self.asarray(1, 2, 3, 4)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('repeat_vector', onnx_model, data, expected, self.model_files))

    def _pooling_test_helper(self, layer, ishape, data_format='channels_last'):
        model = keras.Sequential()
        if is_keras_later_than('2.1.6'):
            nlayer = layer(data_format=data_format, input_shape=ishape) if \
                (layer.__name__.startswith("Global")) else layer(2, data_format=data_format, input_shape=ishape)
        else:
            nlayer = layer(input_shape=ishape) if \
                (layer.__name__.startswith("Global")) else layer(2, input_shape=ishape)

        model.add(nlayer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.random.uniform(-0.5, 0.5, size=(1,) + ishape).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_pooling_1d(self):
        self._pooling_test_helper(AveragePooling1D, (4, 6))
        self._pooling_test_helper(MaxPool1D, (4, 6))
        if is_keras_later_than('2.1.6'):
            self._pooling_test_helper(AveragePooling1D, (4, 6), 'channels_first')
            self._pooling_test_helper(MaxPool1D, (4, 6), 'channels_first')

    def test_pooling_2d(self):
        self._pooling_test_helper(AveragePooling2D, (4, 4, 3))

        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        model = Sequential()
        model.add(MaxPooling2D((2, 2), strides=(2, 2), input_shape=(H, W, C), data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime('max_pooling_2d', onnx_model, x, expected, self.model_files))

        # test padding='same'
        model = Sequential()
        model.add(
            MaxPooling2D((2, 2), strides=(2, 2), padding='same', input_shape=(H, W, C), data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime('max_pooling_2d', onnx_model, x, expected, self.model_files))

    def test_pooling_3d(self):
        self._pooling_test_helper(AveragePooling3D, (4, 4, 4, 3))
        self._pooling_test_helper(MaxPool3D, (4, 4, 4, 3))

    def test_pooling_global(self):
        self._pooling_test_helper(GlobalAveragePooling2D, (4, 6, 2))

    def activationlayer_helper(self, layer, data_for_advanced_layer=None, op_version=None):
        if op_version is None:
            op_version = get_opset_number_from_onnx()
        if data_for_advanced_layer is None:
            data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
            layer = Activation(layer, input_shape=(data.size,))
        else:
            data = data_for_advanced_layer

        model = keras.Sequential()
        model.add(layer)
        onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=op_version)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_tanh(self):
        self.activationlayer_helper('tanh')
        self.activationlayer_helper(keras.activations.tanh)

    def test_sigmoid(self):
        self.activationlayer_helper('sigmoid')
        self.activationlayer_helper(keras.activations.sigmoid)

    def test_hard_sigmoid(self):
        self.activationlayer_helper('hard_sigmoid')
        self.activationlayer_helper(keras.activations.hard_sigmoid)

    def test_relu(self):
        self.activationlayer_helper('relu')
        self.activationlayer_helper(keras.activations.relu)

    def test_elu(self):
        self.activationlayer_helper('elu')
        self.activationlayer_helper(keras.activations.elu)

    def test_selu(self):
        self.activationlayer_helper('selu')
        self.activationlayer_helper(keras.activations.selu)
        SIZE = 10
        NB_CLASS = 5
        model = Sequential()
        model.add(Conv2D(32, strides=(2, 2), kernel_size=3, input_shape=(SIZE, SIZE, 1)))
        model.add(Flatten())
        model.add(Dense(32, activation='selu'))
        model.add(Dense(NB_CLASS, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        data = np.random.rand(5, SIZE, SIZE, 1).astype(np.float32)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_softsign(self):
        self.activationlayer_helper('softsign')
        self.activationlayer_helper(keras.activations.softsign)

    def test_softplus(self):
        self.activationlayer_helper('softplus')
        self.activationlayer_helper(keras.activations.softplus)

    def test_softmax(self):
        self.activationlayer_helper('softmax')
        self.activationlayer_helper(keras.activations.softmax)

    def test_linear(self):
        self.activationlayer_helper('linear')
        self.activationlayer_helper(keras.activations.linear)

    def test_LeakyRelu(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = advanced_activations.LeakyReLU(alpha=0.1, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_ThresholdedRelu(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = advanced_activations.ThresholdedReLU(theta=1.0, input_shape=(data.size,))
        self.activationlayer_helper(layer, data, op_version=8)
        layer = advanced_activations.ThresholdedReLU(theta=1.0, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_ELU(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = advanced_activations.ELU(alpha=1.0, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_PReLU(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = advanced_activations.PReLU(alpha_initializer='zeros', input_shape=(data.size,))
        self.activationlayer_helper(layer, data)
        layer = advanced_activations.PReLU(alpha_initializer='ones', input_shape=(data.size,))
        self.activationlayer_helper(layer, data)
        layer = advanced_activations.PReLU(alpha_initializer='RandomNormal', input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_Softmax(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = advanced_activations.Softmax(axis=-1, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_tf_nn_activation(self):
        for activation in [tf.nn.relu, 'relu']:
            model = keras.Sequential([
                Dense(64, activation=activation, input_shape=[10]),
                Dense(64, activation=activation),
                Dense(1)
            ])
            x = np.random.rand(5, 10).astype(np.float32)
            expected = model.predict(x)
            onnx_model = keras2onnx.convert_keras(model, model.name)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def _misc_conv_helper(self, layer, ishape, target_opset=None):
        if target_opset is None:
            target_opset = get_opset_number_from_onnx()
        input = keras.Input(ishape)
        out = layer(input)
        model = keras.models.Model(input, out)
        onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=target_opset)

        data = np.random.uniform(0, 1, size=(1,) + ishape).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_crop(self):
        # It also passes the test for opset 9, we skip here because it uses a legacy experimental op DynamicSlice.
        for opset_ in [10]:
            ishape = (10, 20)
            for crop_v in [2, (1, 2)]:
                layer = Cropping1D(cropping=crop_v)
                self._misc_conv_helper(layer, ishape, opset_)

            for data_format_ in ['channels_last', 'channels_first']:
                ishape = (20, 20, 1)
                for crop_v in [2, (2, 2), ((1, 2), (2, 3))]:
                    layer = Cropping2D(cropping=crop_v, data_format=data_format_)
                    self._misc_conv_helper(layer, ishape, opset_)
                ishape = (20, 20, 20, 1)
                for crop_v in [2, (2, 3, 4), ((1, 2), (2, 3), (3, 5))]:
                    layer = Cropping3D(cropping=crop_v, data_format=data_format_)
                    self._misc_conv_helper(layer, ishape, opset_)

        # TODO handle other cases for opset 8
        ishape = (20, 20, 1)
        layer = Cropping2D(cropping=((1, 2), (2, 3)), data_format='channels_last')
        self._misc_conv_helper(layer, ishape, opset_)

    def test_upsample(self):
        if is_keras_later_than('2.1.6'):
            ishape = (20, 5)
            layer = UpSampling1D(size=2)
            self._misc_conv_helper(layer, ishape)
            if not is_tf_keras:
                ishape = (20,)
                layer = UpSampling1D(size=2)
                self._misc_conv_helper(layer, ishape)
        ishape = (20, 20, 1)
        for size in [2, (2, 3)]:
            layer = UpSampling2D(size=size, data_format='channels_last')
            self._misc_conv_helper(layer, ishape)
            if not is_keras_older_than("2.2.3"):
                layer = UpSampling2D(size=size, data_format='channels_last', interpolation='bilinear')
                self._misc_conv_helper(layer, ishape)
        ishape = (20, 20, 20, 1)
        layer = UpSampling3D(size=(2, 3, 4), data_format='channels_last')
        self._misc_conv_helper(layer, ishape)

    def test_padding(self):
        ishape = (20, 20, 1)
        layer = ZeroPadding2D(padding=((1, 2), (2, 3)), data_format='channels_last')
        self._misc_conv_helper(layer, ishape)

    def test_embedding(self):
        model = keras.Sequential()
        model.add(Embedding(1000, 64, input_length=10))
        input_array = np.random.randint(1000, size=(1, 10)).astype(np.float32)

        model.compile('rmsprop', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(input_array)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, input_array, expected, self.model_files))

    def _dot_helper(self, l2Normalize, input1, input2):
        data = [input1, input2]
        inputs = [Input(shape=d.shape[1:]) for d in data]

        layer = Dot(axes=-1, normalize=l2Normalize)(inputs)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_dot(self):
        self._dot_helper(False, self.asarray(1, 2, 3), self.asarray(4, 5, 6))
        self._dot_helper(True, self.asarray(1, 2, 3), self.asarray(4, 5, 6))

    def test_dot2(self):
        input_1_shapes = [[32, 20, 1], [2, 3, 5], [2, 3, 5], [4, 3, 5], [2, 7], [2, 3, 4, 12, 3], [1, 3]]
        input_2_shapes = [[32, 30, 20], [2, 3, 5], [2, 3, 5], [4, 5], [2, 7, 5], [2, 3, 4, 15, 3], [1, 3]]
        axes_list = [[1, 2], 1, 2, [2, 1], [1, 1], 4, 1]
        for i_ in range(len(input_1_shapes)):
            for normalize in [True, False]:
                drop2_embed_title = Input(batch_shape=tuple(input_1_shapes[i_]), name='input1')
                att_weight = Input(batch_shape=tuple(input_2_shapes[i_]), name='input2')
                doc_vec1 = dot([drop2_embed_title, att_weight], axes=axes_list[i_], normalize=normalize)
                model = keras.models.Model(inputs=[drop2_embed_title, att_weight], outputs=doc_vec1)
                data1 = np.random.rand(*input_1_shapes[i_]).astype(np.float32)
                data2 = np.random.rand(*input_2_shapes[i_]).astype(np.float32)
                expected = model.predict([data1, data2])
                onnx_model = keras2onnx.convert_keras(model, model.name)
                self.assertTrue(
                    run_onnx_runtime(onnx_model.graph.name, onnx_model, [data1, data2], expected, self.model_files))

        drop2_embed_title = Input(batch_shape=(None, 7), name='input1')
        att_weight = Input(batch_shape=(None, 7, 5), name='input2')
        doc_vec1 = dot([drop2_embed_title, att_weight], axes=[1, 1])
        model = keras.models.Model(inputs=[drop2_embed_title, att_weight], outputs=doc_vec1)
        data1 = np.random.rand(2, 7).astype(np.float32)
        data2 = np.random.rand(2, 7, 5).astype(np.float32)
        expected = model.predict([data1, data2])
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [data1, data2], expected, self.model_files))

    def _batch_norm_helper(self, data, gamma, beta, scale, center, axis):
        model = keras.Sequential()
        layer = BatchNormalization(
            axis=axis,
            input_shape=data.shape[1:],
            moving_mean_initializer=keras.initializers.constant(np.mean(data)),
            moving_variance_initializer=keras.initializers.constant(np.var(data)),
            gamma_initializer=gamma,
            beta_initializer=beta,
            center=center,
            scale=scale,
        )
        model.add(layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_batch_normalization(self):
        data = self.asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self._batch_norm_helper(data, 'ones', 'zeros', True, True, 3)
        self._batch_norm_helper(data, 'ones', 'ones', True, True, 3)
        # The CPU implementation of FusedBatchNorm only supports NHWC tensor format in tf keras
        if not is_tf_keras:
            self._batch_norm_helper(data, 'ones', 'zeros', True, True, 1)
            self._batch_norm_helper(data, 'ones', 'ones', True, True, 1)
            self._batch_norm_helper(data, 'ones', 'ones', True, False, 1)
            self._batch_norm_helper(data, 'zeros', 'zeros', False, True, 1)

    def test_batch_normalization_2(self):
        # The CPU implementation of FusedBatchNorm only supports NHWC tensor format in tf keras
        axis_list = [-1] if is_tf_keras else [1, -1]
        for axis in axis_list:
            batch_size = 4
            input_dim_1 = 10
            input_dim_2 = 20
            input_dim_3 = 30

            model = Sequential()
            model.add(InputLayer(input_shape=(input_dim_1,)))
            model.add(BatchNormalization(axis=axis))
            model.add(Dense(5))
            data = np.random.randn(batch_size, input_dim_1).astype(np.float32)
            onnx_model = keras2onnx.convert_keras(model)
            expected = model.predict(data)
            self.assertTrue(
                run_onnx_runtime('test_batch_normalization_2_2d', onnx_model, [data], expected, self.model_files))

            model = Sequential()
            model.add(InputLayer(input_shape=(input_dim_1, input_dim_2)))
            model.add(BatchNormalization(axis=axis))
            model.add(Dense(5))
            data = np.random.randn(batch_size, input_dim_1, input_dim_2).astype(np.float32)
            onnx_model = keras2onnx.convert_keras(model)
            expected = model.predict(data)
            self.assertTrue(
                run_onnx_runtime('test_batch_normalization_2_3d', onnx_model, [data], expected, self.model_files))

            model = Sequential()
            model.add(InputLayer(input_shape=(input_dim_1, input_dim_2, input_dim_3)))
            model.add(BatchNormalization(axis=axis))
            model.add(Dense(5))
            data = np.random.randn(batch_size, input_dim_1, input_dim_2, input_dim_3).astype(np.float32)
            onnx_model = keras2onnx.convert_keras(model)
            expected = model.predict(data)
            self.assertTrue(
                run_onnx_runtime('test_batch_normalization_2_4d', onnx_model, [data], expected, self.model_files))

    def test_simpleRNN(self):
        K.clear_session()
        inputs1 = keras.Input(shape=(3, 1))
        cls = SimpleRNN(2, return_state=False, return_sequences=True)
        oname = cls(inputs1)  # , initial_state=t0)
        model = keras.Model(inputs=inputs1, outputs=[oname])
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

        # with initial state
        inputs2 = keras.Input(shape=(1, 2))
        state = keras.Input(shape=(5,))
        hidden_1 = SimpleRNN(5, activation='relu', return_sequences=True)(inputs2, initial_state=[state])
        output = Dense(2, activation='sigmoid')(hidden_1)
        keras_model = keras.Model(inputs=[inputs2, state], outputs=output)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, debug_mode=True)

        N, H, W, C = 3, 1, 2, 5
        x = np.random.rand(N, H, W).astype(np.float32, copy=False)
        s = np.random.rand(N, C).astype(np.float32, copy=False)
        expected = keras_model.predict([x, s])
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [x, s], expected, self.model_files))

        # with initial state and output state
        input = keras.Input(shape=(1, 2))
        state_in = keras.Input(shape=(10,))
        hidden_1, state_out = SimpleRNN(10, activation='relu', return_sequences=True,
                                        return_state=True)(input, initial_state=[state_in])
        output = Dense(2, activation='linear')(hidden_1)
        keras_model = keras.Model(inputs=[input, state_in], outputs=[output, state_out])
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

        N, H, W, C = 3, 1, 2, 10
        x = np.random.rand(N, H, W).astype(np.float32, copy=False)
        s = np.random.rand(N, C).astype(np.float32, copy=False)
        expected = keras_model.predict([x, s])
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [x, s], expected, self.model_files))

    def test_GRU(self):
        inputs1 = keras.Input(shape=(3, 1))

        cls = GRU(2, return_state=False, return_sequences=False)
        oname = cls(inputs1)
        model = keras.Model(inputs=inputs1, outputs=[oname])
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

        # GRU with initial state
        for return_sequences in [True, False]:
            cls = GRU(2, return_state=False, return_sequences=return_sequences)
            initial_state_input = keras.Input(shape=(2,))
            oname = cls(inputs1, initial_state=initial_state_input)
            model = keras.Model(inputs=[inputs1, initial_state_input], outputs=[oname])
            onnx_model = keras2onnx.convert_keras(model, model.name)

            data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
            init_state = np.array([0.4, 0.5]).astype(np.float32).reshape((1, 2))
            init_state_onnx = np.array([0.4, 0.5]).astype(np.float32).reshape((1, 2))
            expected = model.predict([data, init_state])
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [data, init_state_onnx], expected,
                                             self.model_files))

    def test_LSTM(self):
        inputs1 = keras.Input(shape=(3, 5))
        data = np.random.rand(3, 5).astype(np.float32).reshape((1, 3, 5))
        for use_bias in [True, False]:
            for return_sequences in [True, False]:
                cls = LSTM(units=2, return_state=True, return_sequences=return_sequences, use_bias=use_bias)
                lstm1, state_h, state_c = cls(inputs1)
                model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
                onnx_model = keras2onnx.convert_keras(model, model.name)
                expected = model.predict(data)
                self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_LSTM_with_bias(self):
        inputs1 = keras.Input(shape=(1, 1))
        cls = LSTM(units=1, return_state=True, return_sequences=True)
        lstm1, state_h, state_c = cls(inputs1)
        model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
        # Set weights: kernel, recurrent_kernel and bias
        model.set_weights((np.array([[1, 2, 3, 4]]), np.array([[5, 6, 7, 8]]), np.array([1, 2, 3, 4])))
        data = np.random.rand(1, 1).astype(np.float32).reshape((1, 1, 1))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_LSTM_reshape(self):
        input_dim = 7
        sequence_len = 3
        inputs1 = keras.Input(shape=(sequence_len, input_dim))
        cls = LSTM(units=5, return_state=False, return_sequences=True)
        lstm1 = cls(inputs1)
        output = Reshape((sequence_len, 5))(lstm1)
        model = keras.Model(inputs=inputs1, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        onnx_model = keras2onnx.convert_keras(model, 'test')
        data = np.random.rand(input_dim, sequence_len).astype(np.float32).reshape((1, sequence_len, input_dim))
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('tf_lstm', onnx_model, data, expected, self.model_files))

    def test_LSTM_with_initializer(self):
        # batch_size = N
        # seq_length = H
        # input_size = W
        # hidden_size = C
        N, H, W, C = 3, 1, 2, 5

        # inputs shape: (batch_size, seq_length)
        inputs = keras.Input(shape=(H, W), name='inputs')

        # initial state shape: (hidden_size, 1)
        state_h = keras.Input(shape=(C,), name='state_h')
        state_c = keras.Input(shape=(C,), name='state_c')

        # create keras model
        lstm_layer = LSTM(units=C, activation='relu', return_sequences=True)(inputs,
                                                                             initial_state=[state_h,
                                                                                            state_c])
        outputs = Dense(W, activation='sigmoid')(lstm_layer)
        keras_model = keras.Model(inputs=[inputs, state_h, state_c], outputs=outputs)

        x = np.random.rand(1, H, W).astype(np.float32)
        sh = np.random.rand(1, C).astype(np.float32)
        sc = np.random.rand(1, C).astype(np.float32)
        expected = keras_model.predict([x, sh, sc])
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, {"inputs": x, 'state_h': sh, 'state_c': sc}, expected,
                             self.model_files))

    @unittest.skipIf(get_opset_number_from_onnx() < 9,
                     "None seq_length LSTM is not supported before opset 9.")
    def test_LSTM_seqlen_none(self):
        lstm_dim = 2
        data = np.random.rand(1, 5, 1).astype(np.float32)
        for return_sequences in [True, False]:
            inp = Input(batch_shape=(1, None, 1))
            out = LSTM(lstm_dim, return_sequences=return_sequences, stateful=True)(inp)
            keras_model = keras.Model(inputs=inp, outputs=out)

            onnx_model = keras2onnx.convert_keras(keras_model, target_opset=10)
            expected = keras_model.predict(data)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_Bidirectional(self):
        input_dim = 10
        sequence_len = 5
        op_version = get_opset_number_from_onnx()
        batch_list = [1, 4] if op_version >= 9 else [1]

        for return_sequences in [True, False]:
            model = keras.Sequential()
            model.add(Bidirectional(LSTM(7, return_sequences=return_sequences),
                                    input_shape=(5, 10)))
            model.add(Dense(5))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            onnx_model = keras2onnx.convert_keras(model, 'test', target_opset=op_version)
            for batch in batch_list:
                data = np.random.rand(batch, sequence_len, input_dim).astype(np.float32)
                expected = model.predict(data)
                self.assertTrue(run_onnx_runtime('bidirectional', onnx_model, data, expected, self.model_files))

        for merge_mode in ['concat', None]:
            for return_sequences in [True, False]:
                sub_input1 = Input(shape=(sequence_len, input_dim))
                sub_mapped1 = Bidirectional(LSTM(7, return_sequences=return_sequences),
                                            input_shape=(5, 10), merge_mode=merge_mode)(sub_input1)
                keras_model = keras.Model(inputs=sub_input1, outputs=sub_mapped1)
                onnx_model = keras2onnx.convert_keras(keras_model, 'test_2', target_opset=op_version)
                for batch in batch_list:
                    data = np.random.rand(batch, sequence_len, input_dim).astype(np.float32)
                    expected = keras_model.predict(data)
                    self.assertTrue(run_onnx_runtime('bidirectional', onnx_model, data, expected, self.model_files))

    def test_Bidirectional_with_bias(self):
        model = keras.Sequential()
        model.add(Bidirectional(LSTM(1, return_sequences=False),
                                input_shape=(1, 1)))
        # Set weights(kernel, recurrent_kernel, bias) for forward layer followed by the backward layer
        model.set_weights(
            (np.array([[1, 2, 3, 4]]), np.array([[5, 6, 7, 8]]), np.array([1, 2, 3, 4]),
             np.array([[1, 2, 3, 4]]), np.array([[5, 6, 7, 8]]), np.array([1, 2, 3, 4])))
        onnx_model = keras2onnx.convert_keras(model, 'test')
        data = np.random.rand(1, 1).astype(np.float32).reshape((1, 1, 1))
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime('bidirectional', onnx_model, data, expected, self.model_files))

    # Bidirectional LSTM with seq_length = None
    @unittest.skipIf(get_opset_number_from_onnx() < 9,
                     "None seq_length Bidirectional LSTM is not supported before opset 9.")
    def test_Bidirectional_seqlen_none(self):
        model = Sequential()
        model.add(Embedding(39, 128))
        model.add(Bidirectional(LSTM(256, input_shape=(None, 32), return_sequences=True)))
        model.add(Dense(44))

        onnx_model = keras2onnx.convert_keras(model, model.name)
        for batch in [1, 4]:
            x = np.random.rand(batch, 50).astype(np.float32)
            expected = model.predict(x)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_seq_dynamic_batch_size(self):
        K.clear_session()
        data_dim = 4  # input_size
        timesteps = 3  # seq_length

        # expected input data shape: (batch_size, timesteps, data_dim)
        test_input = np.random.random_sample((100, timesteps, data_dim))
        test_output = np.random.random_sample((100, 128))

        # Number of layer and number of neurons in each layer
        num_neur = [128, 256, 128]
        epochs = 200
        batch_size = 50
        nodeFuncList = [SimpleRNN, GRU, LSTM]

        for nodeFunc in nodeFuncList:
            model = Sequential()
            for i in range(len(num_neur)):  # multi-layer
                if len(num_neur) == 1:
                    model.add(nodeFunc(num_neur[i], input_shape=(timesteps, data_dim), unroll=True))
                else:
                    if i < len(num_neur) - 1:
                        model.add(
                            nodeFunc(num_neur[i], input_shape=(timesteps, data_dim), return_sequences=True,
                                     unroll=True))
                    else:
                        model.add(nodeFunc(num_neur[i], input_shape=(timesteps, data_dim), unroll=True))

            # Compile the neural network
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(test_input, test_output, epochs=epochs, batch_size=batch_size, verbose=0)
            test_input = np.random.random_sample((5, timesteps, data_dim)).astype(np.float32)
            test_output = model.predict(test_input)
            onnx_model = keras2onnx.convert_keras(model, model.name)
            self.assertTrue(
                run_onnx_runtime(onnx_model.graph.name, onnx_model, test_input, test_output, self.model_files))

    def test_separable_convolution(self):
        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)
        model = Sequential()
        model.add(
            SeparableConv2D(filters=10, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                            data_format='channels_last', depth_multiplier=4))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, 'test')
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime('separable_convolution_1', onnx_model, x, expected, self.model_files))

        x = np.random.rand(N, H, C).astype(np.float32, copy=False)
        model = Sequential()
        model.add(SeparableConv1D(filters=10, kernel_size=2, strides=1, padding='valid', input_shape=(H, C),
                                  data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, 'test')
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime('separable_convolution_2', onnx_model, x, expected, self.model_files))

    def test_shared_embed(self):
        max_cont_length = 5
        max_ques_length = 7
        word_dict_len = 10
        word_dim = 6
        h_word_mat = 'aa'
        # Input Embedding Layer
        contw_input_ = Input((max_cont_length,))  # [bs, c_len]
        quesw_input_ = Input((max_ques_length,))  # [bs, q_len]

        # embedding word
        WordEmbedding = Embedding(word_dict_len, word_dim, trainable=False,
                                  name="word_embedding_" + h_word_mat)
        xw_cont = Dropout(0.)(WordEmbedding(contw_input_))  # [bs, c_len, word_dim]
        xw_ques = Dropout(0.)(WordEmbedding(quesw_input_))  # [bs, c_len, word_dim]

        keras_model = keras.models.Model(inputs=[contw_input_, quesw_input_],
                                         outputs=[xw_cont, xw_ques])
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        batch_size = 3
        x = np.random.rand(batch_size, max_cont_length).astype(np.float32)
        y = np.random.rand(batch_size, max_ques_length).astype(np.float32)
        expected = keras_model.predict([x, y])
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [x, y], expected, self.model_files))

    def test_recursive_model(self):
        keras.backend.set_learning_phase(0)

        N, C, D = 2, 3, 3
        x = np.random.rand(N, C).astype(np.float32, copy=False)

        sub_input1 = Input(shape=(C,))
        sub_mapped1 = Dense(D)(sub_input1)
        sub_model1 = keras.Model(inputs=sub_input1, outputs=sub_mapped1)

        sub_input2 = Input(shape=(C,))
        sub_mapped2 = Dense(D)(sub_input2)
        sub_model2 = keras.Model(inputs=sub_input2, outputs=sub_mapped2)

        input1 = Input(shape=(D,))
        input2 = Input(shape=(D,))
        mapped1_2 = sub_model1(input1)
        mapped2_2 = sub_model2(input2)
        sub_sum = Add()([mapped1_2, mapped2_2])
        keras_model = keras.Model(inputs=[input1, input2], outputs=sub_sum)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

        x = [x, 2 * x]
        expected = keras_model.predict(x)
        self.assertTrue(run_onnx_runtime('recursive', onnx_model, x, expected, self.model_files))

    def test_recursive_and_shared_model(self):
        keras.backend.set_learning_phase(0)
        N, C, D = 2, 3, 3
        x = np.random.rand(N, C).astype(np.float32, copy=False)

        sub_input1 = Input(shape=(C,))
        sub_mapped1 = Dense(D)(sub_input1)
        sub_output1 = Activation('sigmoid')(sub_mapped1)
        sub_model1 = keras.Model(inputs=sub_input1, outputs=sub_output1)

        sub_input2 = Input(shape=(C,))
        sub_mapped2 = sub_model1(sub_input2)
        sub_output2 = Activation('tanh')(sub_mapped2)
        sub_model2 = keras.Model(inputs=sub_input2, outputs=sub_output2)

        input1 = Input(shape=(D,))
        input2 = Input(shape=(D,))
        mapped1_1 = Activation('tanh')(input1)
        mapped2_1 = Activation('sigmoid')(input2)
        mapped1_2 = sub_model1(mapped1_1)
        mapped1_3 = sub_model1(mapped1_2)
        mapped2_2 = sub_model2(mapped2_1)
        sub_sum = Add()([mapped1_3, mapped2_2])
        keras_model = keras.Model(inputs=[input1, input2], outputs=sub_sum)
        keras_model.compile('sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, debug_mode=True)

        x = [x, 2 * x]
        expected = keras_model.predict(x)
        self.assertTrue(run_onnx_runtime('recursive_and_shared', onnx_model, x, expected, self.model_files))

    @unittest.skipIf(is_keras_older_than("2.2.4") or is_tf_keras,
                     "Low keras version is not supported.")
    def test_shared_model_2(self):
        K.set_learning_phase(0)

        def _conv_layer(input, filters, kernel_size, strides=1, dilation_rate=1):
            padding = 'same' if strides == 1 else 'valid'
            if strides > 1:
                input = ZeroPadding2D(((0, 1), (0, 1)), data_format=K.image_data_format())(input)
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       padding=padding, use_bias=False, dilation_rate=dilation_rate)(input)
            ch_axis = 1 if K.image_data_format() == 'channels_first' else -1
            x = BatchNormalization(axis=ch_axis)(x)
            return ReLU()(x)

        def _model():
            input = Input(shape=(3, 320, 320), name='input_1')
            x = _conv_layer(input, 16, 3)
            return Model(inputs=input, outputs=x, name='backbone')

        input = Input(shape=(3, 320, 320), name='input')
        backbone = _model()
        x = backbone(input)
        x = _conv_layer(x, 16, 3)
        model = Model(inputs=[input], outputs=[x])

        onnx_model = keras2onnx.convert_keras(model, model.name)
        x = np.random.rand(2, 3, 320, 320).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_masking(self):
        timesteps, features = (3, 5)
        model = Sequential([
            keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)),
            LSTM(8, return_state=False, return_sequences=False)
        ])

        onnx_model = keras2onnx.convert_keras(model, model.name)
        x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_timedistributed(self):
        keras_model = keras.Sequential()
        keras_model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # keras_model.output_shape == (None, 10, 8)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, debug_mode=True)
        x = np.random.rand(32, 10, 16).astype(np.float32)
        expected = keras_model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

        keras_model = keras.Sequential()
        N, D, W, H, C = 5, 10, 15, 15, 3
        keras_model.add(TimeDistributed(Conv2D(64, (3, 3)),
                                        input_shape=(D, W, H, C)))
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, debug_mode=True)
        x = np.random.rand(N, D, W, H, C).astype(np.float32)
        expected = keras_model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_channel_first_input(self):
        N, W, H, C = 2, 5, 6, 3
        inp1 = Input(batch_shape=(N, W, H, C), name='input1')
        inp2 = Input(batch_shape=(N, W, H, C), name='input2')
        output = Add()([inp1, inp2])
        model = keras.models.Model(inputs=[inp1, inp2], outputs=output)
        onnx_model = keras2onnx.convert_keras(model, model.name, channel_first_inputs=['input1'])
        self.assertIsNotNone(onnx_model)

        data1 = np.random.rand(N, W, H, C).astype(np.float32).reshape((N, W, H, C))
        data2 = np.random.rand(N, W, H, C).astype(np.float32).reshape((N, W, H, C))
        data_transpose = np.transpose(data1, (0, 3, 1, 2))
        self.assertTrue(data_transpose.shape == (N, C, W, H))

        expected = model.predict([data1, data2])
        self.assertTrue(
            run_onnx_runtime('channel_first_input', onnx_model, [data_transpose, data2], expected, self.model_files))

    def test_channel_last(self):
        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        model = Sequential()
        model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))

        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, channel_first_inputs=[model.input_names[0]])

        expected = model.predict(x)
        self.assertIsNotNone(expected, self.model_files)
        self.assertIsNotNone(onnx_model)

        x = np.transpose(x.astype(np.float32), [0, 3, 1, 2])
        self.assertTrue(run_onnx_runtime('channel_last_input', onnx_model, x, expected, self.model_files))

    def test_sub_model(self):
        class IdentityLayer(Layer):
            def __init__(self, **kwargs):
                super(IdentityLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                super(IdentityLayer, self).build(input_shape)

            def call(self, inputs, training=None):
                return inputs

            def compute_output_shape(self, input_shape):
                return input_shape

        input_shape = [700, 420, 1]
        num_classes = 10

        for learning in [True, False]:
            if learning:
                K.set_learning_phase(0)

            image_input = Input(shape=input_shape, name='image_input')

            model = Sequential()  # 28, 28, 1
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                             input_shape=input_shape, padding='valid'))  # 28, 28, 1
            model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))  # 28, 28, 1
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))  # 14, 14, 1
            model.add(Dropout(0.25))
            model.add(Conv2D(128, kernel_size=(12, 12), strides=(14, 14), padding="valid", activation='relu'))
            model.add(Dropout(0.5))

            features = model(image_input)

            outputs = []
            for _ in range(3):
                output1 = Dense(num_classes, activation="softmax")(
                    Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
                output2 = Dense(1, activation="sigmoid")(
                    Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
                output3 = Dense(2, activation="tanh")(
                    Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
                output4 = Dense(2, activation="tanh")(
                    Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
                outputs += [output1, output2, output3, output4]

            output = Concatenate(name="output")(outputs)
            output = IdentityLayer()(output)
            model1 = Model(image_input, output)
            onnx_model = keras2onnx.convert_keras(model1, model1.name, target_opset=7, debug_mode=True)
            x = np.random.rand(2, 700, 420, 1).astype(np.float32)
            expected = model1.predict(x)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
