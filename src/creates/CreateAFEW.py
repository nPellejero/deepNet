import os
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
fileName = 'afewVal'
commPath = '/home/npellejero/tesis/AMFED/AFEW/challenge/'
dataPathCrop = commPath+'Val_Aligned_Faces/'
txtPath  = commPath+'../../redes/txtFiles/BDtxtAfewChallenge/'

nombres = []
datos = []
linesMaster = []
picCount = 0
labCount = 0
dirCount = 0

dirNames = ['Sad', 'Disgust', 'Happy', 'Angry', 'Surprise', 'Fear', 'Neutral']

emoDict = {'Sad': '0', 'Disgust': '1', 'Happy': '2', 'Angry': '3', 'Surprise': '4', 'Fear': '5', 'Neutral': '6' }

for myDir in dirNames:
	print myDir
	myDirNew = myDir
	for fi in os.listdir(os.path.join(dataPathCrop,myDirNew)):
		absFile = os.path.join(dataPathCrop, myDirNew, fi)
		nombres.append(absFile)
		linesMaster.append(emoDict[myDir])

print len(nombres)
print len(linesMaster)

datos = zip(nombres,linesMaster)
datos = map(lambda (x,y): [x,y], datos)
txtFile = open(txtPath+fileName+'BD.txt','w')
writer = csv.writer(txtFile, delimiter=' ')
writer.writerows(datos)
	
