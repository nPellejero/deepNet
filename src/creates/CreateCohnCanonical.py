import os
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
fileName = 'cohn'
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'Emotion/'
dataPathCrop = commPath+'cohn-kanade-images-Manual/'
txtPath  = commPath+'redes/txtFiles/BDtxtCohnKanadeManual/'
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


for subdir, dirs, files in os.walk(dataPathCrop):
   for fi in files:
    if not fi.startswith("."):
     absFile = os.path.join(subdir+"/", fi)
     path = os.path.normpath(absFile)
     path = path.split(os.sep)	
     pathEmo = os.path.join(os.sep,path[0],path[1],path[2],path[3],path[4],"Emotion",path[6],path[7])
     if not(os.path.isdir(pathEmo)):
       continue
     labFileN = os.listdir(pathEmo) 
     if not(labFileN):
       continue
     if not labFileN[0].endswith(".txt"):
       continue
     labFile = open(pathEmo+"/"+labFileN[0])
     lines = labFile.readlines()
     lines = int(lines[0].strip()[0])
     nombres.append(absFile)
     linesMaster.append(lines)
print len(nombres)
print len(linesMaster)
datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x,y], datos)
txtFile = open(txtPath+fileName+'BD.txt','w')
writer = csv.writer(txtFile, delimiter=' ')
writer.writerows(datos)
