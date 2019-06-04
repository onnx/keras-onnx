###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from onnxconverter_common.onnx_ops import *
from onnxconverter_common.onnx_ops import _create_name_or_use_existing_one

def apply_thresholded_relu(scope, input_name, output_name, container, operator_name=None, alpha=None):
    if alpha == None:
        alpha = [1.0]

    name = _create_name_or_use_existing_one(scope, 'ThresholdedRelu', operator_name)
    attrs = {'name': name, 'alpha': alpha[0]}
    if container.target_opset < 10:
    # ThresholdedRelu graduated from an experimental op to a full op in opset 10
    # onnxruntime maintains support in the ONNX domain for ThresholdedRelu as a contrib op
        attrs['op_domain'] = "ai.onnx"
        op_version = 1
    else:
        op_version = 10
    container.add_node('ThresholdedRelu', input_name, output_name, op_version=op_version, **attrs)
