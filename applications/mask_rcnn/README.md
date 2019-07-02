# Introduction
The original Keras project of Masked RCNN is: <https://github.com/matterport/Mask_RCNN>. Follow the 'Installation' section in README.md to set up the model.
There is also a good [tutorial](https://github.com/matterport/Mask_RCNN#step-by-step-detection) to learn about the object detection.

The conversion supports since opset 10.

# Convert
```
cd <mask_rcnn directory>
pip install -e .
cd <keras2onnx directory>/applications/mask_rcnn
python mask_rcnn.py -c # convert to onnx
python mask_rcnn.py <image url> # run object detection
```
