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
from keras2onnx.proto import keras
from distutils.version import StrictVersion
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image, run_onnx_runtime
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

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
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D

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

    @unittest.skipIf(StrictVersion(keras.__version__.split('-')[0]) < StrictVersion("2.2.3"),
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

    def test_ACGAN(self):
        # An ACGAN generator from https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py
        latent_dim = 100
        num_classes = 10
        channels = 1
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
        model.add(keras.layers.Reshape((7, 7, 128)))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding="same"))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding="same"))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Conv2D(channels, kernel_size=3, padding='same'))
        model.add(keras.layers.Activation("tanh"))

        noise = keras.layers.Input(shape=(latent_dim,))
        label = keras.layers.Input(shape=(1,), dtype='int32')
        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(num_classes, 100)(label))

        model_input = keras.layers.multiply([noise, label_embedding])
        img = model(model_input)

        keras_model = keras.models.Model([noise, label], img)
        x = np.random.rand(1, 100).astype(np.float32)
        y = np.random.rand(1, 1).astype(np.int32)

        expected = keras_model.predict([x, y])
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [x, y], expected, self.model_files))

    def test_BIGAN(self):
        # A BIGAN discriminator model from https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
        latent_dim = 100
        img_shape = (28, 28, 1)
        z = keras.layers.Input(shape=(latent_dim,))
        img = keras.layers.Input(shape=img_shape)
        d_in = keras.layers.concatenate([z, keras.layers.Flatten()(img)])

        model = keras.layers.Dense(1024)(d_in)
        model = keras.layers.LeakyReLU(alpha=0.2)(model)
        model = keras.layers.Dropout(0.5)(model)
        model = keras.layers.Dense(1024)(model)
        model = keras.layers.LeakyReLU(alpha=0.2)(model)
        model = keras.layers.Dropout(0.5)(model)
        model = keras.layers.Dense(1024)(model)
        model = keras.layers.LeakyReLU(alpha=0.2)(model)
        model = keras.layers.Dropout(0.5)(model)
        validity = keras.layers.Dense(1, activation="sigmoid")(model)

        keras_model = keras.models.Model([z, img], validity)
        x = np.random.rand(5, 100).astype(np.float32)
        y = np.random.rand(5, 28, 28, 1).astype(np.float32)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

        expected = keras_model.predict([x, y])
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, [x, y], expected, self.model_files))


if __name__ == "__main__":
    unittest.main()