###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .common.onnx_ops import apply_identity, apply_reshape
from .funcbook import set_converter
from .ke2onnx.main import apply_reshape


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


def convert_reshape_timedistributed(scope, operator, container):
    iop = operator.raw_operator
    target_shape = iop.target_shape
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.raw_operator.name, desired_shape=target_shape)


set_converter('identity', default_convert)
set_converter('reshape_timedistributed', convert_reshape_timedistributed)
