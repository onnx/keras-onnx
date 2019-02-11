# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Extension for sphinx.
"""
import sphinx
from docutils import nodes
from docutils.parsers.rst import Directive
import keras2onnx
import onnxruntime


def kerasonnx_version_role(role, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Defines custom role *keras2onnx-version* which returns
    *keras2onnx* version.
    """
    if options is None:
        options = {}
    if content is None:
        content = []
    if text == 'v':
        version = 'v' + keras2onnx.__version__
    elif text == 'rt':
        version = 'v' + onnxruntime.__version__
    else:
        raise RuntimeError("keras2onnx_version_role cannot interpret content '{0}'.".format(text))
    node = nodes.Text(version)
    return [node], []


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    app.add_role('keras2onnxversion', kerasonnx_version_role)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}

