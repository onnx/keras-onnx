# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import os
import unittest
from distutils.version import StrictVersion

import numpy as np
import keras
import onnx
import keras2onnx
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from common_test_class import CommonTestCase


working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


class TestKerasMobileNet(CommonTestCase):

    def _test_keras_model(self, model, model_name='onnx_conversion', rtol=1.e-3, atol=1.e-5, img_size=224):
        img_path = os.path.join(os.path.dirname(__file__), 'data', 'elephant.jpg')
        img = image.load_img(img_path, target_size=(img_size, img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        r = self.run_onnx_runtime(model_name, onnx_model, x, preds, rtol=rtol, atol=atol)
        self.assertTrue(r)

    def test_MobileNet(self):
        from keras.applications import mobilenet
        model = mobilenet.MobileNet(weights='imagenet')
        self._test_keras_model(model)

    @unittest.skipIf(StrictVersion(keras.__version__) < StrictVersion("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_MobileNetV2(self):
        from keras.applications import mobilenet_v2
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        self._test_keras_model(model)


if __name__ == "__main__":
    unittest.main()
