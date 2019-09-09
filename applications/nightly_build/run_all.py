import os
from os import listdir, system
from os.path import isfile, join

mypath = '.'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("test_") == 0]
res_final = True
count = 0
print(len(onlyfiles))
for f_ in onlyfiles:
    res = os.system("pytest " + f_ +  " --doctest-modules --junitxml=junit/test-results-" + f_[5:-3] + ".xml")
    print(f_)

    if not res:
        res_final = False
    if count == 1:
        assert (False)
    count = count + 1


if res_final:
    assert(True)
else:
    assert(False)