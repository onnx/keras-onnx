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
        import skimage
        img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')
        image = skimage.io.imread(img_path)
        images = [image]
        case_name = 'mask_rcnn'

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)

        # preprocessing
        molded_images, image_metas, windows = model.mold_inputs(images)
        anchors = model.get_anchors(molded_images[0].shape)
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        expected = model.keras_model.predict(
            [molded_images.astype(np.float32), image_metas.astype(np.float32), anchors])

        from onnx import numpy_helper
        tensor0 = numpy_helper.from_array(molded_images.astype(np.float32))
        tensor0.name = 'input_image'
        with open(os.path.join(tmp_path, 'input_0.pb'), 'wb') as f:
            f.write(tensor0.SerializeToString())
        tensor1 = numpy_helper.from_array(anchors)
        tensor1.name = 'input_anchors'
        with open(os.path.join(tmp_path, 'input_1.pb'), 'wb') as f:
            f.write(tensor1.SerializeToString())
        tensor2 = numpy_helper.from_array(image_metas.astype(np.float32))
        tensor2.name = 'input_image_meta'
        with open(os.path.join(tmp_path, 'input_2.pb'), 'wb') as f:
            f.write(tensor2.SerializeToString())

        tensor0 = numpy_helper.from_array(expected[0].astype(np.float32))
        tensor0.name = 'mrcnn_detection/Reshape_1:0'
        with open(os.path.join(tmp_path, 'output_0.pb'), 'wb') as f:
            f.write(tensor0.SerializeToString())
        tensor1 = numpy_helper.from_array(expected[1].astype(np.float32))
        tensor1.name = 'mrcnn_class/Reshape_1:0'
        with open(os.path.join(tmp_path, 'output_1.pb'), 'wb') as f:
            f.write(tensor1.SerializeToString())
        tensor2 = numpy_helper.from_array(expected[2].astype(np.float32))
        tensor2.name = 'mrcnn_bbox/Reshape:0'
        with open(os.path.join(tmp_path, 'output_2.pb'), 'wb') as f:
            f.write(tensor2.SerializeToString())
        tensor3 = numpy_helper.from_array(expected[3].astype(np.float32))
        tensor3.name = 'mrcnn_mask/Reshape_1:0'
        with open(os.path.join(tmp_path, 'output_3.pb'), 'wb') as f:
            f.write(tensor3.SerializeToString())
        tensor4 = numpy_helper.from_array(expected[4].astype(np.float32))
        tensor4.name = 'ROI/packed_2:0'
        with open(os.path.join(tmp_path, 'output_4.pb'), 'wb') as f:
            f.write(tensor4.SerializeToString())
        tensor5 = numpy_helper.from_array(expected[5].astype(np.float32))
        tensor5.name = 'rpn_class/concat:0'
        with open(os.path.join(tmp_path, 'output_5.pb'), 'wb') as f:
            f.write(tensor5.SerializeToString())
        tensor6 = numpy_helper.from_array(expected[6].astype(np.float32))
        tensor6.name = 'rpn_bbox/concat:0'
        with open(os.path.join(tmp_path, 'output_6.pb'), 'wb') as f:
            f.write(tensor6.SerializeToString())

if __name__ == "__main__":
    unittest.main()
