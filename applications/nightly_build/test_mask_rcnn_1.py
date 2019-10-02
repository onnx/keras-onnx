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
from distutils.version import StrictVersion

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


# mask rcnn code From https://github.com/matterport/Mask_RCNN
class TestMaskRCNN_Infer(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(StrictVersion(onnx.__version__.split('-')[0]) < StrictVersion("1.5.0"),
                     "NonMaxSuppression op is not supported for onnx < 1.5.0.")
    def test_mask_rcnn_infer(self):
        from onnx import numpy_helper
        tensor0 = onnx.TensorProto()
        with open(os.path.join(tmp_path, 'input_0.pb'), 'rb') as f:
            tensor0.ParseFromString(f.read())
        molded_images = numpy_helper.to_array(tensor0)
        tensor1 = onnx.TensorProto()
        with open(os.path.join(tmp_path, 'input_1.pb'), 'rb') as f:
            tensor1.ParseFromString(f.read())
        anchors = numpy_helper.to_array(tensor1)
        tensor2 = onnx.TensorProto()
        with open(os.path.join(tmp_path, 'input_2.pb'), 'rb') as f:
            tensor2.ParseFromString(f.read())
        image_metas = numpy_helper.to_array(tensor2)
        expected = []
        for idx in range(7):
            tensor = onnx.TensorProto()
            with open(os.path.join(tmp_path, 'output_'+str(idx)+'.pb'), 'rb') as f:
                tensor.ParseFromString(f.read())
            expected.append(numpy_helper.to_array(tensor))

        case_name = 'mask_rcnn'
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(temp_model_file)
        except ImportError:
            return True

        actual = \
            sess.run(None, {"input_image": molded_images.astype(np.float32),
                            "input_anchors": anchors,
                            "input_image_meta": image_metas.astype(np.float32)})

        rtol = 1.e-3
        atol = 1.e-6
        compare_idx = [0, 3]
        res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in compare_idx)
        if res and temp_model_file not in self.model_files:  # still keep the failed case files for the diagnosis.
            self.model_files.append(temp_model_file)
        if not res:
            for n_ in compare_idx:
                expected_list = expected[n_].flatten()
                actual_list = actual[n_].flatten()
                print_mismatches(case_name, n_, expected_list, actual_list, atol, rtol)

        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
