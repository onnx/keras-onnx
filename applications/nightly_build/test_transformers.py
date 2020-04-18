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

enable_full_transformer_test = False
if os.environ.get('ENABLE_FULL_TRANSFORMER_TEST', '0') != '0':
    enable_transformer_test = True


@unittest.skipIf(is_tensorflow_older_than('2.1.0'),
                 "Transformers conversion need tensorflow 2.1.0+")
class TestTransformers(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def _get_token_path(self, file_name):
        return 'https://lotus.blob.core.windows.net/converter-models/transformer_tokenizer/' + file_name

    def _get_tokenzier(self, tokenizer_file):
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    def _prepare_inputs(self, tokenizer):
        raw_data = json.dumps({
            'text': 'The quick brown fox jumps over the lazy dog.'
        })
        text = json.loads(raw_data)['text']
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        inputs_onnx = {k_: v_.numpy() for k_, v_ in inputs.items()}
        return text, inputs, inputs_onnx

    @unittest.skip("Output shape mismatch for tf model prediction.")
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
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFBertForPreTraining(self):
        from transformers import BertConfig, TFBertForPreTraining
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
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
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFBertForNextSentencePrediction(self):
        from transformers import BertConfig, TFBertForNextSentencePrediction
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
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
        tokenizer = self._get_tokenzier(tokenizer_file)
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
        tokenizer = self._get_tokenzier(tokenizer_file)
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
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForQuestionAnswering(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFOpenAIGPTModel(self):
        from transformers import OpenAIGPTConfig, TFOpenAIGPTModel
        keras.backend.clear_session()
        # pretrained_weights = 'openai-gpt'
        tokenizer_file = 'openai_openai-gpt.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = OpenAIGPTConfig()
        model = TFOpenAIGPTModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFOpenAIGPTLMHeadModel(self):
        from transformers import OpenAIGPTConfig, TFOpenAIGPTLMHeadModel
        keras.backend.clear_session()
        # pretrained_weights = 'openai-gpt'
        tokenizer_file = 'openai_openai-gpt.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = OpenAIGPTConfig()
        model = TFOpenAIGPTLMHeadModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFOpenAIGPTDoubleHeadsModel(self):
        from transformers import OpenAIGPTConfig, TFOpenAIGPTDoubleHeadsModel
        keras.backend.clear_session()
        # pretrained_weights = 'openai-gpt'
        tokenizer_file = 'openai_openai-gpt.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = OpenAIGPTConfig()
        model = TFOpenAIGPTDoubleHeadsModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMModel(self):
        from transformers import XLMConfig, TFXLMModel
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMWithLMHeadModel(self):
        from transformers import XLMConfig, TFXLMWithLMHeadModel
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMWithLMHeadModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMForSequenceClassification(self):
        from transformers import XLMConfig, TFXLMForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMForQuestionAnsweringSimple(self):
        from transformers import XLMConfig, TFXLMForQuestionAnsweringSimple
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMForQuestionAnsweringSimple(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertModel(self):
        from transformers import DistilBertConfig, TFDistilBertModel
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForMaskedLM(self):
        from transformers import DistilBertConfig, TFDistilBertForMaskedLM
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFDistilBertForSequenceClassification(self):
        from transformers import DistilBertConfig, TFDistilBertForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForTokenClassification(self):
        from transformers import DistilBertConfig, TFDistilBertForTokenClassification
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForTokenClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForQuestionAnswering(self):
        from transformers import DistilBertConfig, TFDistilBertForQuestionAnswering
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForQuestionAnswering(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFRobertaModel(self):
        from transformers import RobertaConfig, TFRobertaModel
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaModel(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFRobertaForMaskedLM(self):
        from transformers import RobertaConfig, TFRobertaForMaskedLM
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFRobertaForSequenceClassification(self):
        from transformers import RobertaConfig, TFRobertaForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFRobertaForTokenClassification(self):
        from transformers import RobertaConfig, TFRobertaForTokenClassification
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaForTokenClassification(config)
        predictions = model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))


if __name__ == "__main__":
    unittest.main()
