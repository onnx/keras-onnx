###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import functools
import tensorflow as tf


class FunctionStaticVariable(object):
    def __init__(self, *args, **kwargs):
        self.variable = args[0]

    def __call__(self, func):
        return functools.partial(FunctionStaticVariable.retrieve_attr, func, self.variable)

    @staticmethod
    def retrieve_attr(func, var_name, *args, **kwargs):
        if not hasattr(func, var_name):
            result = func(*args, **kwargs)
            setattr(func, var_name, result)

        return getattr(func, var_name)


with_variable = functools.partial(FunctionStaticVariable)


@with_variable('logger')
def k2o_logger():  # type: () -> logging.Logger
    logger = logging.getLogger('keras2onnx')
    logger.setLevel(logging.WARNING)
    tf.logging.set_verbosity(tf.logging.WARN)
    return logger


def set_logger_level(lvl):
    logger = k2o_logger()
    if logger.level != lvl:
        logger.setLevel(lvl)


def get_producer():
    """
    Internal helper function to return the producer
    """
    from .. import __producer__
    return __producer__


def get_producer_version():
    """
    Internal helper function to return the producer version
    """
    from .. import __producer_version__
    return __producer_version__


def get_domain():
    """
    Internal helper function to return the model domain
    """
    from .. import __domain__
    return __domain__


def get_model_version():
    """
    Internal helper function to return the model version
    """
    from .. import __model_version__
    return __model_version__
