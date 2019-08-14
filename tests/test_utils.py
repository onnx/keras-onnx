###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import onnx
import numpy as np

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')

def run_onnx_runtime(case_name, onnx_model, data, expected, model_files, rtol=1.e-3, atol=1.e-6):
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
    onnx.save_model(onnx_model, temp_model_file)
    try:
        import onnxruntime
        sess = onnxruntime.InferenceSession(temp_model_file)
    except ImportError:
        return True

    if not isinstance(expected, list):
        expected = [expected]

    if isinstance(data, dict):
        feed_input = data
    else:
        data = data if isinstance(data, list) else [data]
        input_names = sess.get_inputs()
        # to avoid too complicated test code, we restrict the input name in Keras test cases must be
        # in alphabetical order. It's always true unless there is any trick preventing that.
        feed = zip(sorted(i_.name for i_ in input_names), data)
        feed_input = dict(feed)
    actual = sess.run(None, feed_input)
    res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in range(len(expected)))
    if res and temp_model_file not in model_files:  # still keep the failed case files for the diagnosis.
        model_files.append(temp_model_file)

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
