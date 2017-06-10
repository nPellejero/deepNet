import os
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
fileName = 'jaffe'
commPath = '/home/npellejero/tesis/AMFED/'
dataPathCrop = commPath+'jaffe/'
txtPath  = commPath+'redes/txtFiles/BDtxtJaffe/'
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

emoc = {'HA':'0', 'NE':'1', 'FE':'2', 'SA':'3', 'SU':'4', 'DI':'5', 'AN':'6' }

print dataPathCrop
for subdir, dirs, files in os.walk(dataPathCrop):
   for fi in files:
    if fi.endswith(".png"):
       absFile = os.path.join(subdir, fi)
       nombres.append(absFile)
       print fi
       fiList = fi.split(".")
       print fiList
       linesMaster.append(emoc[fiList[1][0:2]])
print len(nombres)
print len(linesMaster)
datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x,y], datos)
txtFile = open(txtPath+fileName+'BD.txt','w')
writer = csv.writer(txtFile, delimiter=' ')
writer.writerows(datos)


