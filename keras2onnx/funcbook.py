###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from uuid import uuid4
import re
import six

_converters = {}
fb_key, fb_id, fb_additional = range(3)


def set_converter(op_type, functor):
    _converters[op_type] = functor
    return functor


def get_converter(op_type):
    return _converters.get(op_type)


def create_pattern_dict():
    dict_p = {}
    for k_, v_ in six.iteritems(_converters):
        if hasattr(v_, 'patterns') and len(v_.patterns) > 0:
            dict_p[re.compile(v_.patterns[0])] = (k_, uuid4(), [re.compile(p) for p in v_.patterns[1:]])

    return dict_p


def set_converters(op_conv_dict):
    _converters.update(op_conv_dict)


def converter_func(*types):
    def my_func(func):
        for type in types:
            set_converter(type, func)
        return func

    return my_func
