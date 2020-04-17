import os
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exclude')
args = parser.parse_args()

files = ['test_keras_applications_v2.py', 'test_transformers_bert.py']
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
