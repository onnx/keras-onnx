###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .common.onnx_ops import apply_identity, apply_reshape
from .funcbook import set_converter


class TYPES:
    # tf-node types:
    Identity = 'Identity'

    # converter internal types:
    TD_Reshape = 'reshape_timedistributed'


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


def convert_reshape_timedistributed(scope, operator, container):
    target_shape = operator.get_attr('target_shape')
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.full_name, desired_shape=target_shape)


set_converter(TYPES.Identity, default_convert)
set_converter(TYPES.TD_Reshape, convert_reshape_timedistributed)
