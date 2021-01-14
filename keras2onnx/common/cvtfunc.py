# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools


class ConverterFunction(object):
    """
    The converter function decoration class
    """

    @staticmethod
    def _functor_more(functor, pattern):
        functor.patterns.append(pattern)
        return functor

    def __init__(self, *args, **kwargs):
        self.name = args
        self.pattern = kwargs.get('pattern')
        self.shape_infer = kwargs.get('shape_infer')

    def __call__(self, func):
        setattr(func, 'shape_infer', self.shape_infer)
        setattr(func, 'patterns', [] if self.pattern is None else [self.pattern])
        func.more = functools.partial(ConverterFunction._functor_more, func)
        return func


cvtfunc = functools.partial(ConverterFunction)
