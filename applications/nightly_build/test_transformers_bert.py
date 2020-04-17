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
import urllib.request
import pickle
from os.path import dirname, abspath
from keras2onnx.proto.tfcompat import is_tf2
from keras2onnx.proto import keras

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_onnx_runtime
from keras2onnx.proto import is_tensorflow_older_than

enable_transformer_test = True
if os.environ.get('ENABLE_TRANSFORMER_TEST', '0') != '0':
    enable_transformer_test = True


@unittest.skipIf(is_tensorflow_older_than('2.1.0') or not enable_transformer_test,
                 "Need enable transformer test before Transformers conversion.")
class TestTransformersBert(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def _get_token_path(self, file_name):
        return 'https://lotus.blob.core.windows.net/converter-models/transformer_tokenizer/' + file_name

    def _prepare_inputs(self, tokenizer):
        raw_data = json.dumps({
            'text': 'The quick brown fox jumps over the lazy dog.'
        })
        text = json.loads(raw_data)['text']
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        inputs_onnx = {k_: v_.numpy() for k_, v_ in inputs.items()}
        return text, inputs, inputs_onnx

    def test_TFBertModel(self):
        from transformers import BertConfig, TFBertModel
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFBertForPreTraining(self):
        from transformers import BertConfig, TFBertForPreTraining
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForPreTraining(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFBertForMaskedLM(self):
        from transformers import BertConfig, TFBertForMaskedLM
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFBertForNextSentencePrediction(self):
        from transformers import BertConfig, TFBertForNextSentencePrediction
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForNextSentencePrediction(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForSequenceClassification(self):
        from transformers import BertConfig, TFBertForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForTokenClassification(self):
        from transformers import BertConfig, TFBertForTokenClassification
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForTokenClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForQuestionAnswering(self):
        from transformers import BertConfig, TFBertForQuestionAnswering
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForQuestionAnswering(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

if __name__ == "__main__":
    unittest.main()
