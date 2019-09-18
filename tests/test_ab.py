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


class TestKerasTF2ONNX_2(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @staticmethod
    def asarray(*a):
        return np.array([a], dtype='f')

    def test_simpleRNN(self):
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

    def test_seq_dynamic_batch_size(self):
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

if __name__ == "__main__":
    unittest.main()
