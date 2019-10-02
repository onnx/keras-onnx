###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras2onnx
from keras2onnx.proto import keras
import onnx
import numpy as np
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_onnx_runtime, print_mismatches, tf2onnx_contrib_op_conversion

import urllib.request
MASKRCNN_WEIGHTS_PATH = r'https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5'
model_file_name = 'mask_rcnn_coco.h5'
if not os.path.exists(model_file_name):
    urllib.request.urlretrieve(MASKRCNN_WEIGHTS_PATH, model_file_name)

keras.backend.clear_session()
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../mask_rcnn/'))
from mask_rcnn import model
from distutils.version import StrictVersion

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


# mask rcnn code From https://github.com/matterport/Mask_RCNN
# Split TestMaskRCNN into two files, otherwise it needs too much memory.
class TestMaskRCNN_Conversion(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(StrictVersion(onnx.__version__.split('-')[0]) < StrictVersion("1.5.0"),
                     "NonMaxSuppression op is not supported for onnx < 1.5.0.")
    def test_mask_rcnn_conversion(self):
        onnx_model = keras2onnx.convert_keras(model.keras_model, target_opset=10, custom_op_conversions=tf2onnx_contrib_op_conversion)
        case_name = 'mask_rcnn'

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
