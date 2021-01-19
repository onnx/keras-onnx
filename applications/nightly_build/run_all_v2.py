# SPDX-License-Identifier: Apache-2.0

import os

files = ['test_keras_applications_v2.py', 'test_transformers.py', 'test_chatbot.py', 'test_efn.py', \
         'test_resnext.py']
files.sort()

res_final = True
for f_ in files:
    res = os.system("pytest " + f_ +  " --doctest-modules --junitxml=junit/test-results-" + f_[5:-3] + ".xml")
    if res > 0:
        res_final = False

if res_final:
    assert(True)
else:
    assert(False)
