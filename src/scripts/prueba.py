import subprocess
import platform
import sys
sys.path.append("/home/npellejero/digits/nvidia-caffe/python/")
import caffe
caffe.set_mode_gpu()
import lmdb

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

print "OS:     ", platform.platform()
print "Python: ", sys.version.split("\n")[0]
print "CUDA:   ", subprocess.Popen(["nvcc","--version"], stdout=subprocess.PIPE).communicate()[0].split("\n")[3]
print "LMDB:   ", ".".join([str(i) for i in lmdb.version()])
