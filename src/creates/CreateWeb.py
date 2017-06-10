import os
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
fileName = 'afew'
commPath = '/home/npellejero/tesis/AMFED/'
dataPathCrop = commPath+'webface/'
txtPath  = commPath+'redes/txtFiles/BDtxtWeb/'
masterCount = 0
nombres = []
datos= []
linesMaster = []

print dataPathCrop
for subdir, dirs, files in os.walk(dataPathCrop):
  myDir = os.path.split(subdir)[-1]
  for fi in files:
    if fi.endswith(".png"):
       absFile = os.path.join(subdir, fi)
       nombres.append(absFile)
       linesMaster.append(myDir)
print len(nombres)
print len(linesMaster)
datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x,y], datos)
txtFile = open(txtPath+fileName+'BD.txt','w')
writer = csv.writer(txtFile, delimiter=' ')
writer.writerows(datos)


