#Ketone
| Linux | Windows |


# Introduction 
Ketone (Ke-T-ON = Keras-Tensorflow-ONNX) enables you convert the keras models into [ONNX](https://onnx.ai).
Initially Keras converter was developer in the project onnxmltools. To support more kinds of keras models and reduce the complexity of mixing multiple converters, ketone was created to convert the keras model only. 

Ketone supports the keras lambda/custom layer by parsing the TF graph built from Keras model.
More intro will be coming soon...

# Testing

## Validate pre-trained Keras application models
It will be useful to convert the models from Keras to ONNX from a python script.
You can use the following API:
```
import ketone
ketone.convert_keras(model, name=None, doc_string='', target_opset=None, channel_first_inputs=None):
    # type: (keras.Model, str, str, int, []) -> onnx.ModelProto
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string:
    :param target_opset:
    :param channel_first_inputs: A list of channel first input.
    :return:
    """
```

We converted successfully for the keras application models such as:
Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201.
The following unit test can be embedded in ```tests/test_layers.py``` for testing.

```
def test_Xception(self):
    from keras.applications.xception import Xception
    model = Xception(include_top=True, weights='imagenet')
    self._test_keras_model(model, img_size=299, atol=5e-3)

def test_VGG16(self):
    from keras.applications.vgg16 import VGG16
    model = VGG16(include_top=True, weights='imagenet')
    self._test_keras_model(model)

def test_VGG19(self):
    from keras.applications.vgg19 import VGG19
    model = VGG19(include_top=True, weights='imagenet')
    self._test_keras_model(model)

def test_ResNet50(self):
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(include_top=True, weights='imagenet')
    self._test_keras_model(model)

def test_InceptionV3(self):
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(include_top=True, weights='imagenet')
    self._test_keras_model(model, rtol=1.e-3)

def test_InceptionResNetV2(self):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    model = InceptionResNetV2(include_top=True, weights='imagenet')
    self._test_keras_model(model)

def test_DenseNet121(self):
    from keras.applications.densenet import DenseNet121
    model = DenseNet121(include_top=True, weights='imagenet')
    self._test_keras_model(model)

def test_DenseNet169(self):
    from keras.applications.densenet import DenseNet169
    model = DenseNet169(include_top=True, weights='imagenet')
    self._test_keras_model(model)

def test_DenseNet201(self):
    from keras.applications.densenet import DenseNet201
    model = DenseNet201(include_top=True, weights='imagenet')
    self._test_keras_model(model)
```