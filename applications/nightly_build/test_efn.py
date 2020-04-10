###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras2onnx
from os.path import dirname, abspath
from keras2onnx.proto import keras, is_keras_older_than
from keras2onnx.proto.tfcompat import is_tf2

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image, run_onnx_runtime

img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')


class TestEfn(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skip("Minor discrepancy on the model output.")
    def test_custom(self):
        from efficientnet import keras as efn
        keras.backend.set_learning_phase(0)
        base_model = efn.EfficientNetB0(input_shape=(600, 600, 3), weights=None)
        backbone = keras.Model(base_model.input, base_model.get_layer("top_activation").output)
        res = run_image(backbone, self.model_files, img_path, target_size=(600, 600), rtol=1e-1)
        self.assertTrue(*res)

    @unittest.skip("Minor discrepancy on the model output.")
    def test_efn(self):
        from efficientnet import keras as efn
        keras.backend.set_learning_phase(0)
        model = efn.EfficientNetB7(weights='imagenet')
        res = run_image(model, self.model_files, img_path, target_size=(600, 600), rtol=1e-1)
        self.assertTrue(*res)

    @unittest.skipIf(not is_tf2, "Tensorflow 2.x only tests")
    def test_efn_2(self):
        import efficientnet.tfkeras as efn
        import numpy as np
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        model = efn.EfficientNetB0(weights=None)
        expected = model.predict(data)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime('onnx_efn_2', onnx_model, data, expected, self.model_files))

if __name__ == "__main__":
    unittest.main()
