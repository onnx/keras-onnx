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
import numpy as np
from keras2onnx.proto import keras, is_tf_keras
from distutils.version import StrictVersion


working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @staticmethod
    def asarray(*a):
        return np.array([a], dtype='f')

    @staticmethod
    def get_temp_file(name):
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        return os.path.join(tmp_path, name)

    def run_onnx_runtime(self, case_name, onnx_model, data, expected, rtol=1.e-3, atol=1.e-6):
        temp_model_file = TestKerasApplications.get_temp_file('temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)
        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(temp_model_file)
        except ImportError:
            return True

        if not isinstance(expected, list):
            expected = [expected]

        data = data if isinstance(data, list) else [data]
        input_names = sess.get_inputs()
        # to avoid too complicated test code, we restrict the input name in Keras test cases must be
        # in alphabetical order. It's always true unless there is any trick preventing that.
        feed = zip(sorted(i_.name for i_ in input_names), data)
        actual = sess.run(None, dict(feed))
        res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in range(len(expected)))
        if res and temp_model_file not in self.model_files:  # still keep the failed case files for the diagnosis.
            self.model_files.append(temp_model_file)

        if not res:
            for n_ in range(len(expected)):
                expected_list = expected[n_].flatten()
                actual_list = actual[n_].flatten()
                diff_list = abs(expected_list - actual_list)
                count_total = len(expected_list)
                count_error = 0

                for e_, a_, d_ in zip(expected_list, actual_list, diff_list):
                    if d_ > atol + rtol * abs(a_):
                        if count_error < 10:  # print the first 10 mismatches
                            print(
                                "case = " + case_name + ", result mismatch for expected = " + str(e_) +
                                ", actual = " + str(a_), file=sys.stderr)
                        count_error = count_error + 1

                print("case = " + case_name + ", " +
                      str(count_error) + " mismatches out of " + str(count_total) + " for list " + str(n_),
                      file=sys.stderr)

        return res

    def _test_keras_model(self, model, model_name='onnx_conversion', rtol=1.e-3, atol=1.e-5, img_size=224):
        preprocess_input = keras.applications.resnet50.preprocess_input
        image = keras.preprocessing.image

        img_path = os.path.join(os.path.dirname(__file__), 'data', 'elephant.jpg')
        try:
            img = image.load_img(img_path, target_size=(img_size, img_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            onnx_model = keras2onnx.convert_keras(model, model.name)
            self.assertTrue(self.run_onnx_runtime(model_name, onnx_model, x, preds, rtol=rtol, atol=atol))
        except FileNotFoundError:
            self.assertTrue(False, 'The image data does not exist.')

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
        self._test_keras_model(model, img_size=299)

    def test_DenseNet121(self):
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(include_top=True, weights='imagenet')
        self._test_keras_model(model)

    def test_Xception(self):
        from keras.applications.xception import Xception
        model = Xception(include_top=True, weights='imagenet')
        self._test_keras_model(model, atol=5e-3, img_size=299)

    if __name__ == "__main__":
        unittest.main()
