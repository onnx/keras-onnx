###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import onnx
import unittest
import keras2onnx
import keras_segmentation
import numpy as np
from keras2onnx.proto import keras, is_tf_keras, is_keras_older_than
from distutils.version import StrictVersion
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image, run_onnx_runtime
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

K = keras.backend
Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling2D = keras.layers.MaxPooling2D
Model = keras.models.Model
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D
if not (is_keras_older_than("2.2.4") or is_tf_keras):
    ReLU = keras.layers.ReLU

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

    # TODO: Move this to test_layers.py after this PR (https://github.com/microsoft/onnxconverter-common/pull/22) is in pypi.
    @unittest.skipIf(is_keras_older_than("2.2.4"),
                     "ReLU support requires keras 2.2.4 or later.")
    def test_shared_model_3(self):
        def _bottleneck(x, filters, activation, strides, block_id):
            padding = 'same' if strides == 1 else 'valid'
            ch_axis = 1 if K.image_data_format() == 'channels_first' else -1
            if strides > 1:
                x = ZeroPadding2D(((0, 1), (0, 1)), data_format=K.image_data_format())(x)

            x = Conv2D(filters // 2, (1, 1), padding='same', name=f'bottleneck_{block_id}_conv_0',
                          use_bias=False, data_format=K.image_data_format())(x)

            x = BatchNormalization(axis=ch_axis, name=f'bottleneck_{block_id}_bnorm_0')(x)

            if activation == 'relu':
                x = ReLU(name=f'bottleneck_{block_id}_relu_0')(x)
            elif activation == 'leaky':
                x = LeakyReLU(name=f'bottleneck_{block_id}_leaky_0')(x)
            else:
                assert False

            x = Conv2D(filters // 2, (3, 3), padding=padding, name=f'bottleneck_{block_id}_conv_1',
                          strides=strides, use_bias=False, data_format=K.image_data_format())(x)
            x = BatchNormalization(axis=ch_axis, name=f'bottleneck_{block_id}_bnorm_1')(x)
            if activation == 'relu':
                x = ReLU(name=f'bottleneck_{block_id}_relu_1')(x)
            elif activation == 'leaky':
                x = LeakyReLU(name=f'bottleneck_{block_id}_leaky_1')(x)
            else:
                assert False

            x = Conv2D(filters, (1, 1), padding='same', name=f'bottleneck_{block_id}_conv_2',
                          use_bias=False, data_format=K.image_data_format())(x)
            x = BatchNormalization(axis=ch_axis, name=f'bottleneck_{block_id}_bnorm_2')(x)
            if activation == 'relu':
                x = ReLU(name=f'bottleneck_{block_id}_relu_2')(x)
            elif activation == 'leaky':
                x = LeakyReLU(name=f'bottleneck_{block_id}_leaky_2')(x)
            else:
                assert False

            return x

        def convnet_7(input_shape, activation):
            input = Input(shape=input_shape, name='input_1')
            x = _bottleneck(input, filters=16, strides=1, activation=activation, block_id=1)
            x = _bottleneck(x, filters=32, strides=2, activation=activation, block_id=2)
            return Model(inputs=input, outputs=x, name='convnet_7')

        for activation in ['relu', 'leaky']:
            model = convnet_7(input_shape=(3, 96, 128), activation=activation)
            onnx_model = keras2onnx.convert_keras(model, model.name)
            x = np.random.rand(1, 3, 96, 128).astype(np.float32)
            expected = model.predict(x)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
