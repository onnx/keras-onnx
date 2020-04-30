###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
from keras2onnx.proto import keras
from keras2onnx.proto.tfcompat import is_tf2
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image

img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')


@unittest.skipIf(not is_tf2, "Tensorflow 2.x only tests")
class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_DenseNet121(self):
        print("\ttest_DenseNet121")
        DenseNet121 = keras.applications.densenet.DenseNet121
        model = DenseNet121(include_top=True, weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_MobileNet(self):
        print("\ttest_MobileNet")
        MobileNet = keras.applications.mobilenet.MobileNet
        model = MobileNet(weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_MobileNetV2(self):
        print("\ttest_MobileNetV2")
        MobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2
        model = MobileNetV2(weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_NASNetMobile(self):
        print("\ttest_NASNetMobile")
        NASNetMobile = keras.applications.nasnet.NASNetMobile
        model = NASNetMobile(weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_InceptionV3(self):
        print("\ttest_InceptionV3")
        keras.backend.set_learning_phase(0)
        InceptionV3 = keras.applications.inception_v3.InceptionV3
        model = InceptionV3(include_top=True)
        model.save('inception.h5')
        res = run_image(model, self.model_files, img_path, target_size=299, tf_v2=True)
        self.assertTrue(*res)

    def test_ResNet50(self):
        print("\ttest_ResNet50")
        ResNet50 = keras.applications.resnet_v2.ResNet50V2
        model = ResNet50(include_top=True, weights=None)
        res = run_image(model, self.model_files, img_path, tf_v2=True)
        self.assertTrue(*res)

    def test_Xception(self):
        print("\ttest_Xception")
        Xception = keras.applications.xception.Xception
        model = Xception(include_top=True, weights=None)
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=299, tf_v2=True)
        self.assertTrue(*res)


if __name__ == "__main__":
    unittest.main()
