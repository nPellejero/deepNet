import os
import cv2
import numpy as np
import sys, csv
import itertools as ITER
import Image as I
from scipy import misc
fileName = 'afewTest'
commPath = '/home/npellejero/tesis/AMFED/AFEW/challenge/100x100/'
dataPathCrop = commPath+'Test_Aligned_Faces/'
txtPath  = commPath+'../../../src/theano/data/afew100/'

nombres = []
datos = []
linesMaster = []
picCount = 0
labCount = 0
dirCount = 0

def img2arrayGS(path):
    return misc.imread(path, flatten=True)

def img2arrayRGB(path):
    return misc.imread(path)

dirNames = ['Sad', 'Disgust', 'Happy', 'Angry', 'Surprise', 'Fear', 'Neutral']

emoDict = {'Sad': '0', 'Disgust': '1', 'Happy': '2', 'Angry': '3', 'Surprise': '4', 'Fear': '5', 'Neutral': '6' }
counter = 0

for myDir in dirNames:
	print myDir
	myDirNew = myDir
	for fi in os.listdir(os.path.join(dataPathCrop,myDirNew)):
		absFile = os.path.join(dataPathCrop, myDirNew, fi)
		counter = counter + 1
		img = img2arrayGS(absFile)
		img = cv2.resize(img,(64,64))
		nombres.append(np.ndarray.flatten(img))
		linesMaster.append(emoDict[myDir])
		if counter % 1000 == 0:
			print counter
print len(nombres)

txtFile = open(txtPath+fileName+'BD.npz','w')
np.savez(txtFile, x=nombres, y=linesMaster)	
