import os
from os import listdir, system
from os.path import isfile, join

if __name__ == "__main__":
    mypath = '.'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("test_") == 0]
    res_final = True
    count = 0
    for f_ in onlyfiles:
        res = os.system("pytest " + f_)

        if not res:
            res_final = False
        if count == 1:
            assert (False)
        count = count + 1

    if not res_final:
        assert(False)