import os
from os import listdir
from os.path import isfile, join

mypath = '.'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("test_") == 0]
res_final = True
for f_ in onlyfiles:
    res = os.system("pytest " + f_ +  " --doctest-modules --junitxml=junit/test-results-" + f_[5:-3] + ".xml")
    if not res:
        res_final = False

if res_final:
    assert(True)
else:
    assert(False)