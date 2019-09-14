###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras2onnx
import onnx
import numpy as np
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))

import urllib.request
YOLOV3_WEIGHTS_PATH = r'https://pjreddie.com/media/files/yolov3.weights'
model_file_name = 'yolov3.weights'

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras-yolo3'))
from yolo import YOLO
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../model_source/yolov3/'))
from convert import convert_weights

from distutils.version import StrictVersion

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


class TestYoloV3(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        return
        for fl in self.model_files:
            os.remove(fl)

    def post_compute(self, all_boxes, all_scores, indices):
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices[0]:
            out_classes.append(idx_[1])
            out_scores.append(all_scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(all_boxes[idx_1])
        return [out_boxes, out_scores, out_classes]

    @unittest.skipIf(StrictVersion(onnx.__version__.split('-')[0]) < StrictVersion("1.5.0"),
                     "NonMaxSuppression op is not supported for onnx < 1.5.0.")
    def test_yolov3(self):
        yolo3_dir = os.path.join(os.path.dirname(__file__), '../../../keras-yolo3')
        model_dir = os.path.join(yolo3_dir, 'model_data')
        img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')
        yolo3_yolo3_dir = os.path.join(os.path.dirname(__file__), '../../../keras-yolo3/yolo3')

        yolov3_weights_path = os.path.join(yolo3_dir, 'yolov3.weights')
        yolov3_cfg_path = os.path.join(yolo3_dir, 'yolov3.cfg')
        yolo_h5_path = os.path.join(model_dir, 'yolo.h5')

        if not os.path.exists(yolov3_weights_path):
            urllib.request.urlretrieve(YOLOV3_WEIGHTS_PATH, yolov3_weights_path)

        yolo_weights = None
        if not os.path.exists(yolo_h5_path):
            yolo_weights = convert_weights(yolov3_cfg_path, yolov3_weights_path, yolo_h5_path)

        my_yolo = YOLO(yolo3_yolo3_dir)
        my_yolo.load_model(yolo_weights)
        case_name = 'yolov3'
        target_opset = 10
        onnx_model = keras2onnx.convert_keras(my_yolo.final_model, target_opset=target_opset, channel_first_inputs=['input_1'])

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)

        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(temp_model_file)
        except ImportError:
            return True

        from PIL import Image
        image = Image.open(img_path)
        image_data = my_yolo.prepare_keras_data(image)

        all_boxes_k, all_scores_k, indices_k = my_yolo.final_model.predict([image_data, np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2)])

        image_data_onnx = np.transpose(image_data, [0, 3, 1, 2])

        feed_f = dict(zip(['input_1', 'image_shape'],
                          (image_data_onnx, np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2))))
        all_boxes, all_scores, indices = sess.run(None, input_feed=feed_f)

        expected = self.post_compute(all_boxes_k, all_scores_k, indices_k)
        actual = self.post_compute(all_boxes, all_scores, indices)

        res = all(np.allclose(expected[n_], actual[n_]) for n_ in range(3))
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
