import os
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
fileName = 'fer'
commPath = '/home/npellejero/tesis/AMFED/'
dataPathCrop = commPath+'imgs/'
txtPath  = commPath+'redes/txtFiles/BDtxtFer/'
masterCount = 0
maxIm = 255
maxLabs = 100
numLab = 0
nombres = []
datos= []
linesMaster = []
picCount = 0
labCount = 0
dirCount = 0

print dataPathCrop
for subdir, dirs, files in os.walk(dataPathCrop):
   for fi in files:
    if not fi.startswith("."):
       absFile = os.path.join(subdir, fi)
       nombres.append(absFile)
       linesMaster.append(fi[-5])
print len(nombres)
print len(linesMaster)
datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x,y], datos)
txtFile = open(txtPath+fileName+'BD.txt','w')
writer = csv.writer(txtFile, delimiter=' ')
writer.writerows(datos)


