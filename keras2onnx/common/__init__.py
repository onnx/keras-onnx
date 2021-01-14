# SPDX-License-Identifier: Apache-2.0

from .utils import with_variable, get_default_batch_size
from .utils import k2o_logger, set_logger_level
from .cvtfunc import cvtfunc
from .intop import Operator
from .interim import OnnxObjectContainer, InterimContext, Variable


# keras2onnx common code has been refactored into onnxconverter-common.

def name_func(scope, operator):
    """Returns a function that can generate unique names for an operator based on the
    scope.
    """

    def _name_func(name):
        return scope.get_unique_operator_name(operator.full_name + '_' + name)

    return _name_func
