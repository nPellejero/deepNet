import os
import numpy as np
import sys, csv
import itertools as ITER
fileName = sys.argv[1]
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'AULabels/'
dataCropPath = commPath+'images-croped/'
dataPath = commPath+'images/'
txtPath   = commPath+'redes/txtFiles/BDtxtAllLabs/'
masterCount = 0
maxIm = 255
maxLabs = 100
numLab = 0
def myFun(y):
	y = float(y)
	if y > 0:
		return 1
	else:
		return 0
def refactor(lines):
	lines = lines[1:]
	lines = map(lambda x: x.split(' '),lines)
	lines = map(lambda x: map( lambda y: y.strip(),x), lines)	
	lines = map(lambda x: map( lambda y: myFun(y),x), lines)	
	return lines
with open(fileName) as f:
	dirNames = f.readlines()
	dirNames = map( lambda x: x.strip(), dirNames)

for myDir in dirNames:
	with open(labsPath+'2new-'+myDir+'-label.csv') as labFile:
		lines = refactor(labFile.readlines())
	countLabs = len(lines)
	countData = len(os.listdir(dataPath+myDir))
	masterCount = masterCount + min(countLabs,countData)

print masterCount
nombres = []
datos= []
linesMaster = []
picCount = 0
labCount = 0
dirCount = 0
print len(dirNames)
for myDir in dirNames:
	print dirCount 
	print myDir
	dirCount = dirCount + 1
	with open(labsPath+'2new-'+myDir+'-label.csv') as labFile:
		lines = labFile.readlines()
		lines = refactor(lines)
	countLabs = len(lines)
	countData = len(os.listdir(dataPath+myDir))
	step = min(countLabs,countData)
	linesMaster.append(lines[:step])	
	print labCount+step
	print step
	print masterCount	
	labCount = labCount + step	 	
	for num in range(1,step+1):
		lenNum = len(str(num))
		cant0  = 5-lenNum
		pic = "Pictures"+'0'*cant0+str(num)+".jpg"
		nombres.append(dataPath+myDir+"/"+pic)
		picCount = picCount + 1

linesMaster = list(ITER.chain.from_iterable(linesMaster))
print len(nombres)
print len(linesMaster)
datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x]+y, datos)
#print datos
with open(txtPath+fileName+'BD.txt','w') as f:
	writer = csv.writer(f, delimiter=' ')
	writer.writerows(datos)


