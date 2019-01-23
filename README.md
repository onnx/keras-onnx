#Ketone
| Linux | Windows |


# Introduction 
Ketone (Ke-T-ON = Keras-Tensorflow-ONNX) enables you convert the keras models into [ONNX](https://onnx.ai).
Initially Keras converter was developer in the project onnxmltools. To support more kinds of keras models and reduce the complexity of mixing multiple converters, ketone was created to convert the keras model only. 

Ketone supports the keras lambda/custom layer by parsing the TF graph built from Keras model.
More intro will be coming soon...

# Testing

## Validate pre-trained Keras application models
In some cases it will be useful to convert the models from Keras to ONNX from a python script. You can use the following API:
```buildoutcfg
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

See the examples:

```buildoutcfg
import keras
import ketone
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

def test_ResNet50(self):
    model = ResNet50(weights='imagenet')
    img_path = 'data/elephant.jpg'   # make sure the image exists in img_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    onnx_model = ketone.convert_keras(model, model.name)
    self.assertTrue(self.run_onnx_runtime('onnx_ResNet50', onnx_model, x, preds, rtol=1.e-4, atol=1.e-8))
```