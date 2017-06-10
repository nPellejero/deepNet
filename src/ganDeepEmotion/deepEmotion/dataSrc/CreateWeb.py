import os
import numpy as np
import sys, csv
import cv2
import itertools as ITER
import Image as I
from scipy import misc
fileName = 'casia'
commPath = '/home/npellejero/tesis/AMFED/'
dataPathCrop = commPath+'webface/'
txtPath  = commPath+'src/theano/data/casia100'
masterCount = 0
nombres = []
datos= []
linesMaster = []

def img2arrayGS(path):
    return misc.imread(path, flatten=True)

def img2arrayRGB(path):
    return misc.imread(path)
counter = 0
print dataPathCrop
for subdir, dirs, files in os.walk(dataPathCrop):
  myDir = os.path.split(subdir)[-1]
  for fi in files:
    if fi.endswith(".png"):
       counter = counter + 1
       if counter % 1000 == 0:
         print counter
       if counter > 100000:
         break
       absFile = os.path.join(subdir, fi)
       clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
       img = img2arrayGS(absFile)
       img = cv2.convertScaleAbs(img)
       img = clahe.apply(img)
       img = cv2.resize(img,(64,64))
       nombres.append(np.ndarray.flatten(img))
       linesMaster.append(myDir)
       
print len(nombres)
print len(linesMaster)
txtFile = open(txtPath+fileName+'BD.npz','w')
np.savez(txtFile, x=nombres, y=linesMaster)
