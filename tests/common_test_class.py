# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import os
import unittest

import numpy as np
import keras
import onnx
from keras2onnx.common import keras2onnx_logger


working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


class CommonTestCase(unittest.TestCase):

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

    def run_onnx_runtime(self, case_name, onnx_model, data, expected, rtol=1.e-4, atol=1.e-8):
        temp_model_file = CommonTestCase.get_temp_file('temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)
        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(temp_model_file)
        except ImportError:
            return True

        if not isinstance(expected, list):
            expected = [expected]

        data = data if isinstance(data, list) else [data]
        feed = dict([(x.name, data[n]) for n, x in enumerate(sess.get_inputs())])
        actual = sess.run(None, feed)
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
                            keras2onnx_logger().error(
                                "case = " + case_name + ", result mismatch for expected = " + str(e_) +
                                ", actual = " + str(a_))
                        count_error = count_error + 1

                keras2onnx_logger().error("case = " + case_name + ", " +
                                          str(count_error) + "mismatches out of " + str(count_total) + " for list " + str(n_))
            assert False

        return res
