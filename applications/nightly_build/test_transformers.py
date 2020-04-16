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

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_onnx_runtime
from keras2onnx.proto import is_tensorflow_older_than

enable_transformer_test = True
if os.environ.get('ENABLE_TRANSFORMER_TEST', '0') != '0':
    enable_transformer_test = True


@unittest.skipIf(not enable_transformer_test or not is_tf2,
                 "Need enable transformer test before Transformers conversion.")
class TestTransformers(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def _prepare_inputs(self, tokenizer):
        raw_data = json.dumps({
            'text': 'The quick brown fox jumps over the lazy dog.'
        })
        text = json.loads(raw_data)['text']
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        inputs_onnx = {k_: v_.numpy() for k_, v_ in inputs.items()}
        return text, inputs, inputs_onnx

    def test_3layer_gpt2(self):
        from transformers import GPT2Config, TFGPT2Model, BertTokenizer
        keras2onnx.proto.keras.backend.set_learning_phase(0)
        config = GPT2Config(n_layer=3)
        model = TFGPT2Model(config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertModel(self):
        from transformers import BertConfig, TFBertModel
        tokenizer_file = 'bertModel_bert-base-uncased.pickle'
        token_path = r'https://lotus.blob.core.windows.net/converter-models/transformer_tokenizer/bertModel_bert-base-uncased.pickle'
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    '''
    def test_TFBertForPreTraining(self):
        from transformers import BertTokenizer, TFBertForPreTraining
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFBertForPreTraining.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFBertForMaskedLM(self):
        from transformers import BertTokenizer, TFBertForMaskedLM
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFBertForMaskedLM.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFBertForNextSentencePrediction(self):
        from transformers import BertTokenizer, TFBertForNextSentencePrediction
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFBertForNextSentencePrediction.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForSequenceClassification(self):
        from transformers import BertTokenizer, TFBertForSequenceClassification
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFBertForSequenceClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForTokenClassification(self):
        from transformers import BertTokenizer, TFBertForTokenClassification
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFBertForTokenClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForQuestionAnswering(self):
        from transformers import BertTokenizer, TFBertForQuestionAnswering
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFBertForQuestionAnswering.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFOpenAIGPTModel(self):
        from transformers import OpenAIGPTTokenizer, TFOpenAIGPTModel
        pretrained_weights = 'openai-gpt'
        tokenizer = OpenAIGPTTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFOpenAIGPTModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFOpenAIGPTLMHeadModel(self):
        from transformers import OpenAIGPTTokenizer, TFOpenAIGPTLMHeadModel
        pretrained_weights = 'openai-gpt'
        tokenizer = OpenAIGPTTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFOpenAIGPTLMHeadModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFOpenAIGPTDoubleHeadsModel(self):
        from transformers import OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel
        pretrained_weights = 'openai-gpt'
        tokenizer = OpenAIGPTTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFOpenAIGPTDoubleHeadsModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFXLMModel(self):
        from transformers import XLMTokenizer, TFXLMModel
        pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer = XLMTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFXLMModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFXLMWithLMHeadModel(self):
        from transformers import XLMTokenizer, TFXLMWithLMHeadModel
        pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer = XLMTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFXLMWithLMHeadModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFXLMForSequenceClassification(self):
        from transformers import XLMTokenizer, TFXLMForSequenceClassification
        pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer = XLMTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFXLMForSequenceClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFXLMForQuestionAnsweringSimple(self):
        from transformers import XLMTokenizer, TFXLMForQuestionAnsweringSimple
        pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer = XLMTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFXLMForQuestionAnsweringSimple.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertModel(self):
        from transformers import DistilBertTokenizer, TFDistilBertModel
        pretrained_weights = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFDistilBertModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForMaskedLM(self):
        from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM
        pretrained_weights = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFDistilBertForMaskedLM.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFDistilBertForSequenceClassification(self):
        from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
        pretrained_weights = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFDistilBertForSequenceClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForTokenClassification(self):
        from transformers import DistilBertTokenizer, TFDistilBertForTokenClassification
        pretrained_weights = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFDistilBertForTokenClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForQuestionAnswering(self):
        from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
        pretrained_weights = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFDistilBertForQuestionAnswering.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFRobertaModel(self):
        from transformers import RobertaTokenizer, TFRobertaModel
        pretrained_weights = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFRobertaModel.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFRobertaForMaskedLM(self):
        from transformers import RobertaTokenizer, TFRobertaForMaskedLM
        pretrained_weights = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFRobertaForMaskedLM.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFRobertaForSequenceClassification(self):
        from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
        pretrained_weights = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFRobertaForSequenceClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFRobertaForTokenClassification(self):
        from transformers import RobertaTokenizer, TFRobertaForTokenClassification
        pretrained_weights = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        model = TFRobertaForTokenClassification.from_pretrained(pretrained_weights)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))
    '''

if __name__ == "__main__":
    unittest.main()
