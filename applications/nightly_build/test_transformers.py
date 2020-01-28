###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras2onnx
import json
import os
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_onnx_runtime
from keras2onnx.proto import is_tensorflow_older_than
from transformers import BertTokenizer, TFBertForSequenceClassification


class TestTransformers(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(is_tensorflow_older_than('2.1.0'),
                     "TFBertForSequenceClassification conversion needs TF 2.1.0+.")
    def test_TFBertForSequenceClassification(self):
        pretrained_weights = 'bert-base-uncased'
        raw_data = json.dumps({
            'text': 'The quick brown fox jumps over the lazy dog.'
        })
        text = json.loads(raw_data)['text']

        labels = ['c#', '.net', 'java', 'asp.net', 'c++', 'javascript', 'php', 'python', 'sql', 'sql-server']
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = TFBertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=len(labels))
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        inputs_onnx = {}
        for input_ in inputs:
            inputs_onnx[input_] = inputs[input_].numpy()

        predictions = model.predict(inputs)
        # Make sure to set TF_KERAS = 1 since this is tf-keras model.
        os.environ["TF_KERAS"] = "1"
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))


if __name__ == "__main__":
    unittest.main()
