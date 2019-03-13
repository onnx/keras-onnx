# Introduction
The original lpcnet model was coming from: <https://github.com/mozilla/LPCNet/>, we made changes to adapt to convert to onnx model.

# Inference
```
python test_lpcnet.py [feature_file] ./output.pcm [model_file]
```
feature_file is the input features, e.g., test_features.f32.
output.pcm is the synthesized wave saved with PCM format at 16k 16bits mono.
model_file is the model with trained weights, it is a *.h5 file.
