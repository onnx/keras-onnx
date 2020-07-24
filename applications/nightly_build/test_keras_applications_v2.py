###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import numpy as np
import keras2onnx
from keras2onnx.proto import keras
from keras2onnx.proto.tfcompat import is_tf2
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image, run_onnx_runtime

img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')


@unittest.skipIf(not is_tf2, "Tensorflow 2.x only tests")
class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_DenseNet121(self):
        DenseNet121 = keras.applications.densenet.DenseNet121
        model = DenseNet121(include_top=True, weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_MobileNet(self):
        MobileNet = keras.applications.mobilenet.MobileNet
        model = MobileNet(weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_MobileNetV2(self):
        MobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2
        model = MobileNetV2(weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_NASNetMobile(self):
        NASNetMobile = keras.applications.nasnet.NASNetMobile
        model = NASNetMobile(weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_InceptionV3(self):
        keras.backend.set_learning_phase(0)
        InceptionV3 = keras.applications.inception_v3.InceptionV3
        model = InceptionV3(include_top=True)
        res = run_image(model, self.model_files, img_path, target_size=299, tf_v2=True)
        self.assertTrue(*res)
    
    def test_InceptionResNetV2(self):
        InceptionResNetV2 = keras.applications.inception_resnet_v2.InceptionResNetV2
        model = InceptionResNetV2(include_top=True)
        res = run_image(model, self.model_files, img_path, target_size=299, tf_v2=True)
        self.assertTrue(*res)

    def test_ResNet50(self):
        ResNet50 = keras.applications.resnet_v2.ResNet50V2
        model = ResNet50(include_top=True, weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_Xception(self):
        Xception = keras.applications.xception.Xception
        model = Xception(include_top=True, weights=None)
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=299, tf_v2=True)
        self.assertTrue(*res)

    def test_keras_ocr(self):
        import keras_ocr
        build_params = keras_ocr.recognition.DEFAULT_BUILD_PARAMS
        backbone, model, training_model, prediction_model = keras_ocr.recognition.build_model(
            alphabet=keras_ocr.recognition.DEFAULT_ALPHABET, **build_params)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(2, 31, 200, 1).astype(np.float32)
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files, rtol=1e-2, atol=5e-3))



if __name__ == "__main__":
    unittest.main()
