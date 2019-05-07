import os
import sys
import time
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import onnx
import keras2onnx

from mrcnn.config import Config
from mrcnn import model as modellib, utils

from keras2onnx._builtin import on_StridedSlice, on_Round, on_TopKV2, on_Pad
import tensorflow as tf


ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


from keras2onnx import set_converter
from keras2onnx.ke2onnx.batch_norm import convert_keras_batch_normalization
from mrcnn.model import PyramidROIAlign, BatchNorm


def create_onnx_node(scope, operator, container, type):
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer, str) -> None
    container.add_node(type, operator.input_full_names, operator.output_full_names, op_version=operator.target_opset)


def convert_PyramidROIAlign(scope, operator, container):
    # create_onnx_node(scope, operator, container, 'PyramidROIAlign')
    #create_onnx_node(scope, operator, container, 'RoiAlign')
    preprocessor_type = 'RoiAlign'
    temp_name_list = []
    from keras2onnx.proto import onnx_proto
    for i_ in range(2, 6):
        shape_name = scope.get_unique_variable_name('roi_shape')
        container.add_node('Shape', operator.input_full_names[1], shape_name, op_version=operator.target_opset)

        starts_name = scope.get_unique_variable_name('roi_slice_starts')
        starts = np.asarray([0], dtype=np.int32)
        container.add_initializer(starts_name, onnx_proto.TensorProto.INT32, starts.shape, starts.flatten())

        ends_name = scope.get_unique_variable_name('roi_slice_ends')
        ends = np.asarray([np.iinfo(np.int32).max], dtype=np.int32)
        container.add_initializer(ends_name, onnx_proto.TensorProto.INT32, ends.shape, ends.flatten())

        axes_name = scope.get_unique_variable_name('roi_slice_axes')
        axes = np.asarray([0], dtype=np.int32)
        container.add_initializer(axes_name, onnx_proto.TensorProto.INT32, axes.shape, axes.flatten())

        slice_name = scope.get_unique_variable_name('roi_slice')
        container.add_node('Slice', [shape_name, starts_name, ends_name, axes_name], slice_name, op_version=operator.target_opset)

        constant_of_shape_name = scope.get_unique_variable_name('roi_constant_of_shape')
        container.add_node('ConstantOfShape', slice_name, constant_of_shape_name, op_version=operator.target_opset)

        cast_name = scope.get_unique_variable_name('roi_cast')
        attrs = {'to': 7}
        container.add_node('Cast', constant_of_shape_name, cast_name, op_version=operator.target_opset, **attrs)

        temp_name = scope.get_unique_variable_name('pyramid_roi')
        attrs = {'name': scope.get_unique_operator_name(preprocessor_type),
                 'output_height': operator.raw_operator.pool_shape[0],
                 'output_width': operator.raw_operator.pool_shape[1]}
        container.add_node('RoiAlign', [operator.input_full_names[i_], operator.input_full_names[0], cast_name], temp_name, op_version=operator.target_opset,
                           **attrs)
        temp_name_list.append(temp_name)

    attrs = {'axis': 0}
    container.add_node('Concat', temp_name_list, operator.output_full_names, op_version=operator.target_opset, **attrs)


def convert_BatchNorm(scope, operator, container):
    convert_keras_batch_normalization(scope, operator, container)


def convert_DenseToDenseSetOperation(scope, operator, container):
    container.add_node('DenseIntersection', operator.input_full_names, operator.output_full_names, op_domain='com.microsoft', op_version=operator.target_opset)


def convert_SparseToDense(scope, operator, container):
    container.add_node('Identity', operator.input_full_names, operator.output_full_names, op_version=operator.target_opset)


from keras2onnx.common.onnx_ops import apply_transpose, apply_identity
from keras2onnx.proto import onnx_proto

def convert_DetectionLayer(scope, operator, container):
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer) -> None
    DETECTION_MAX_INSTANCES = 100
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7

    box_transpose = scope.get_unique_variable_name(operator.inputs[0].full_name + '_tx')
    score_transpose = scope.get_unique_variable_name(operator.inputs[1].full_name + '_tx')

    # apply_transpose(scope, operator.inputs[0].full_name, box_transpose, container, perm=[2, 0, 1])
    apply_identity(scope, operator.inputs[0].full_name, box_transpose, container)
    apply_transpose(scope, operator.inputs[1].full_name, score_transpose, container, perm=[1, 0])

    box_batch = scope.get_unique_variable_name(operator.inputs[0].full_name + '_btc')
    score_batch = scope.get_unique_variable_name(operator.inputs[1].full_name + '_btc')

    container.add_node("Unsqueeze", box_transpose,
                       box_batch, op_version=operator.target_opset, axes=[0])
    container.add_node("Unsqueeze", score_transpose,
                       score_batch, op_version=operator.target_opset, axes=[0])

    max_output_size = scope.get_unique_variable_name('max_output_size')
    iou_threshold = scope.get_unique_variable_name('iou_threshold')
    score_threshold = scope.get_unique_variable_name('layer.score_threshold')

    container.add_initializer(max_output_size, onnx_proto.TensorProto.INT64,
                              [], [DETECTION_MAX_INSTANCES])
    container.add_initializer(iou_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [DETECTION_NMS_THRESHOLD])
    container.add_initializer(score_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [DETECTION_MIN_CONFIDENCE])

    nms_node = next((nd_ for nd_ in operator.node_list if nd_.type == 'NonMaxSuppressionV3'), operator.node_list[0])
    nms_output = scope.get_unique_variable_name(operator.output_full_names[0] + '_nms')
    container.add_node("NonMaxSuppression",
                       [box_batch, score_batch, max_output_size, iou_threshold, score_threshold],
                       nms_output,
                       op_version=operator.target_opset,
                       name=nms_node.name)

    cast_name = scope.get_unique_variable_name(operator.output_full_names[0] + '_nms_cast')
    attrs = {'to': 1}
    container.add_node('Cast', nms_output, cast_name, op_version=operator.target_opset, **attrs)

    concat_node = next((nd_ for nd_ in operator.node_list if nd_.type == 'Concat'), operator.node_list[0])
    attrs = {'axis': 1}
    container.add_node("Concat",
                       [box_batch, cast_name, score_batch],
                       operator.output_full_names[0],
                       op_version=operator.target_opset,
                       name=concat_node.name, **attrs)


set_converter(modellib.DetectionLayer, convert_DetectionLayer)
set_converter(PyramidROIAlign, convert_PyramidROIAlign)
set_converter(BatchNorm, convert_BatchNorm)
#set_converter(tf.sets.set_intersection, convert_DenseToDenseSetOperation)
#set_converter(tf.sparse_tensor_to_dense, convert_SparseToDense)

_custom_op_handlers = {
    'Round': (on_Round, []),
    'StridedSlice': (on_StridedSlice, []),
    'TopKV2': (on_TopKV2, []),
    'Pad': (on_Pad, []),
    'PadV2': (on_Pad, [])
}

model.keras_model.save('mrcnn.h5')
oml = keras2onnx.convert_keras(model.keras_model, target_opset=10, debug_mode=True, custom_op_conversions=_custom_op_handlers)
onnx.save_model(oml, './mrcnn.onnx')

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']
#
# # Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#
# # Run detection
# results = model.detect([image], verbose=1)
import onnxruntime
sess = onnxruntime.InferenceSession('./mrcnn.onnx')