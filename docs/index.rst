..  SPDX-License-Identifier: Apache-2.0


keras-onnx: Convert your Keras model into ONNX
==============================================

.. list-table:
    :header-rows: 1
    :widths: 5 5
    * - Linux
      - Windows
    * - .. image:: https://dev.azure.com/onnxmltools/ketone/_apis/build/status/linux-conda-ci?branchName=master
            :target: https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=9&branchName=master
      - .. image:: https://dev.azure.com/onnxmltools/ketone/_apis/build/status/win32-conda-ci?branchName=master
            :target: https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=10&branchName=master


*keras-onnx* enables you to convert models from
`keras <https://keras.io/>`_
toolkits into `ONNX <https://onnx.ai>`_.

.. toctree::
    :maxdepth: 1

    api_summary
    auto_examples/index

**Issues, questions**

You should look for `existing issues <https://github.com/onnx/keras-onnx/issues?utf8=%E2%9C%93&q=is%3Aissue>`_
or submit a new one. Sources are available on
`onnx/keras-onnx <https://github.com/onnx/keras-onnx>`_.

**ONNX version**

If you want the converted model is compatible with certain ONNX version,
please specify the *target_opset* parameter on invoking convert function,
and the following Keras converter example code shows how it works.

**Backend**

*keras-onnx* converts models in ONNX format which
can be then used to compute predictions with the
backend of your choice. However, there exists a way
to automatically check every converter with
`onnxruntime <https://pypi.org/project/onnxruntime/>`_,
`onnxruntime-gpu <https://pypi.org/project/onnxruntime-gpu>`_.
Every converter is tested with this backend.

**Related converters**

*keras-onnx* converts models from *scikit-learn*.
It was initially part of `onnxmltools <https://github.com/onnx/onnxmltools>`_
which can still be used to convert models for *xgboost* and *libsvm*.
Other converters can be found on `github/onnx <https://github.com/onnx/>`_,
`torch.onnx <https://pytorch.org/docs/stable/onnx.html>`_,
`ONNX-MXNet API <https://mxnet.incubator.apache.org/api/python/contrib/onnx.html>`_,
`Microsoft.ML.Onnx <https://www.nuget.org/packages/Microsoft.ML.Onnx/>`_...

**License**

It is licensed with `Apache License v2.0 <../LICENSE>`_.


