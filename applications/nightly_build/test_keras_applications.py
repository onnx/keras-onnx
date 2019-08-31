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
from test_utils import run_onnx_runtime

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
MaxPooling1D = keras.layers.MaxPooling1D
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D

class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def _test_keras_model(self, model, model_name='onnx_conversion', rtol=1.e-3, atol=1.e-5, target_size=224):
        preprocess_input = keras.applications.resnet50.preprocess_input
        image = keras.preprocessing.image

        img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')
        try:
            if not isinstance(target_size, tuple):
                target_size = (target_size, target_size)
            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
        except FileNotFoundError:
            self.assertTrue(False, 'The image data does not exist.')
            return

        try:
            preds = model.predict(x)
        except Exception:
            self.assertTrue(True, 'keras prediction throws an exception, skip it.')
            return

        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(model_name, onnx_model, x, preds, self.model_files, rtol=rtol, atol=atol))


    def test_MobileNet(self):
        mobilenet = keras.applications.mobilenet
        model = mobilenet.MobileNet(weights='imagenet')
        self._test_keras_model(model)

    @unittest.skipIf(StrictVersion(keras.__version__.split('-')[0]) < StrictVersion("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_MobileNetV2(self):
        mobilenet_v2 = keras.applications.mobilenet_v2
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        self._test_keras_model(model)

    def test_ResNet50(self):
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(include_top=True, weights='imagenet')
        self._test_keras_model(model)

    def test_InceptionV3(self):
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet')
        self._test_keras_model(model, target_size=299)

    def test_DenseNet121(self):
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(include_top=True, weights='imagenet')
        self._test_keras_model(model)

    def test_Xception(self):
        from keras.applications.xception import Xception
        model = Xception(include_top=True, weights='imagenet')
        self._test_keras_model(model, atol=5e-3, target_size=299)

    def test_fcn(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/fcn.py
        model = keras_segmentation.models.fcn.fcn_8(101)
        self._test_keras_model(model, target_size=(416, 608))

    def _pool_block(self, feats, pool_factor, IMAGE_ORDERING):
        import keras.backend as K
        if IMAGE_ORDERING == 'channels_first':
            h = K.int_shape(feats)[2]
            w = K.int_shape(feats)[3]
        elif IMAGE_ORDERING == 'channels_last':
            h = K.int_shape(feats)[1]
            w = K.int_shape(feats)[2]
        pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]
        x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING, strides=strides, padding='same')(feats)
        x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = keras_segmentation.models.model_utils.resize_image(x, strides, data_format=IMAGE_ORDERING)
        return x

    def test_pspnet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/pspnet.py
        from keras_segmentation.models.basic_models import vanilla_encoder
        img_input, levels = vanilla_encoder(input_height=384, input_width=576)
        o = levels[4]
        pool_factors = [1, 2, 3, 6]
        pool_outs = [o]
        IMAGE_ORDERING = 'channels_last'
        if IMAGE_ORDERING == 'channels_first':
            MERGE_AXIS = 1
        elif IMAGE_ORDERING == 'channels_last':
            MERGE_AXIS = -1
        for p in pool_factors:
            pooled = self._pool_block(o, p, IMAGE_ORDERING)
            pool_outs.append(pooled)
        o = Concatenate(axis=MERGE_AXIS)(pool_outs)
        o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False)(o)
        o = BatchNormalization()(o)
        o = Activation('relu')(o)
        o = Conv2D(101, (3, 3), data_format=IMAGE_ORDERING, padding='same')(o)
        o = keras_segmentation.models.model_utils.resize_image(o, (8, 8), data_format=IMAGE_ORDERING)

        model = keras_segmentation.models.model_utils.get_segmentation_model(img_input, o)
        model.model_name = "pspnet"

        self._test_keras_model(model, target_size=(384, 576))

    def test_segnet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/segnet.py
        model = keras_segmentation.models.segnet.segnet(101)
        self._test_keras_model(model, target_size=(416, 608))

    def test_vgg_segnet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/segnet.py
        model = keras_segmentation.models.segnet.vgg_segnet(101)
        self._test_keras_model(model, target_size=(416, 608))

    def test_unet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/unet.py
        model = keras_segmentation.models.unet.unet(101)
        self._test_keras_model(model, target_size=(416, 608))

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
