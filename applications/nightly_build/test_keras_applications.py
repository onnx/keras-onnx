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
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from keras2onnx.proto import keras, is_keras_older_than
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image, run_onnx_runtime, test_level_0
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

Activation = keras.layers.Activation
Average = keras.layers.Average
AveragePooling2D = keras.layers.AveragePooling2D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
concatenate = keras.layers.concatenate
Convolution2D = keras.layers.Convolution2D
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Model = keras.models.Model
Sequential = keras.models.Sequential

class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_MobileNet(self):
        mobilenet = keras.applications.mobilenet
        model = mobilenet.MobileNet(weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    @unittest.skipIf(is_keras_older_than("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_MobileNetV2(self):
        mobilenet_v2 = keras.applications.mobilenet_v2
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_ResNet50(self):
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_InceptionV3(self):
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path, target_size=299)
        self.assertTrue(*res)

    def test_DenseNet121(self):
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_Xception(self):
        from keras.applications.xception import Xception
        model = Xception(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=299)
        self.assertTrue(*res)

    def test_SmileCNN(self):
        # From https://github.com/kylemcdonald/SmileCNN/blob/master/2%20Training.ipynb
        nb_filters = 32
        nb_pool = 2
        nb_conv = 3
        nb_classes = 2

        model = Sequential()

        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=32)
        self.assertTrue(*res)

    @unittest.skipIf(is_keras_older_than("2.2.4"),
                     "keras-resnet requires keras 2.2.4 or later.")
    def test_keras_resnet_batchnormalization(self):
        N, C, H, W = 2, 3, 120, 120
        import keras_resnet

        model = Sequential()
        model.add(ZeroPadding2D(padding=((3, 3), (3, 3)), input_shape=(H, W, C), data_format='channels_last'))
        model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=False,
                         data_format='channels_last'))
        model.add(keras_resnet.layers.BatchNormalization(freeze=True, axis=3))

        onnx_model = keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(N, H, W, C).astype(np.float32).reshape((N, H, W, C))
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    # model from https://github.com/titu1994/Image-Super-Resolution
    def test_ExpantionSuperResolution(self):
        init = Input(shape=(32, 32, 3))
        x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(init)
        x1 = Convolution2D(32, (1, 1), activation='relu', padding='same', name='lavel1_1')(x)
        x2 = Convolution2D(32, (3, 3), activation='relu', padding='same', name='lavel1_2')(x)
        x3 = Convolution2D(32, (5, 5), activation='relu', padding='same', name='lavel1_3')(x)
        x = Average()([x1, x2, x3])
        out = Convolution2D(3, (5, 5), activation='relu', padding='same', name='output')(x)
        model = keras.models.Model(init, out)
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=32)
        self.assertTrue(*res)

    def test_tcn(self):
        from tcn import TCN
        batch_size, timesteps, input_dim = None, 20, 1
        actual_batch_size = 3
        i = Input(batch_shape=(batch_size, timesteps, input_dim))
        np.random.seed(1000)  # set the random seed to avoid the output result discrepancies.
        for return_sequences in [True, False]:
            o = TCN(return_sequences=return_sequences)(i)  # The TCN layers are here.
            o = Dense(1)(o)
            model = keras.models.Model(inputs=[i], outputs=[o])
            onnx_model = keras2onnx.convert_keras(model, model.name)
            data = np.random.rand(actual_batch_size, timesteps, input_dim).astype(np.float32).reshape((actual_batch_size, timesteps, input_dim))
            expected = model.predict(data)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    # model from https://github.com/titu1994/LSTM-FCN
    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_lstm_fcn(self):
        MAX_SEQUENCE_LENGTH = 176
        NUM_CELLS = 8
        NB_CLASS = 37
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

        x = LSTM(NUM_CELLS)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(NB_CLASS, activation='softmax')(x)

        model = Model(ip, out)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        batch_size = 2
        data = np.random.rand(batch_size, 1, MAX_SEQUENCE_LENGTH).astype(np.float32).reshape(batch_size, 1, MAX_SEQUENCE_LENGTH)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    # model from https://github.com/CyberZHG/keras-self-attention
    @unittest.skipIf(test_level_0 or get_maximum_opset_supported() < 11,
                     "Test level 0 only.")
    def test_keras_self_attention(self):
        from keras_self_attention import SeqSelfAttention
        keras.backend.clear_session()

        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=10000,
                                         output_dim=300,
                                         mask_zero=True))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                               return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(keras.layers.Dense(units=5))
        onnx_model = keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(5, 10).astype(np.float32).reshape(5, 10)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
