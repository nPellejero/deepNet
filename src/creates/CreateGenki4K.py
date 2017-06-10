import os
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'GENKI-4K_Labels.txt'
dataPath = commPath+'images-genki-4k/'
txtPath  = commPath+'redes/txtFiles/BDtxtGenkis4k/'
masterCount = 0
maxIm = 255
maxLabs = 100
numLab = 0
def refactor(lines):
	lines = map(lambda x: x.split(' '),lines)
	lines = map(lambda x: map( lambda y: y.strip(),x), lines)	
	return lines


nombres = []
datos= []
linesMaster = []
picCount = 0
labCount = 0
dirCount = 0
step = 4000
labFile = open(labsPath)
labels = labFile.readlines()	
labels = map( lambda x: x.strip(), labels)
for num in range(1,step):
		lenNum = len(str(num))
		cant0  = 16-lenNum
		pic = "file"+'0'*cant0+str(num)+".png"
		if not os.path.isfile(dataPath+pic):
			continue 
		nombres.append(dataPath+pic)
		linesMaster.append(labels[num-1][0])	
		picCount = picCount + 1

linesMaster = list(ITER.chain.from_iterable(linesMaster))
print len(nombres)
print len(linesMaster)
datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x,y], datos)
#print datos
with open(txtPath+'genkisBD.txt','w') as f:
	writer = csv.writer(f, delimiter=' ')
	writer.writerows(datos)


