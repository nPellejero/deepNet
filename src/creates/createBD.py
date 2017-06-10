import h5py, os
import numpy as np
import sys
import Image as I

fileName = sys.argv[1]
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'AULabels/'
dataPath = commPath+'images/'
h5Path   = commPath+'redes/pruebaBD/'
masterCount = 0
maxIm = 255
maxLabs = 100

def refactor(lines):
	lines = lines[1:]
	lines = map(lambda x: x.split(' '),lines)
	lines = map(lambda x: map( lambda y: y.strip(),x), lines)	
	lines = map(lambda x: map( lambda y: float(y)/maxLabs,x), lines)	
	return lines
with open(fileName) as f:
	dirNames = f.readlines()
	dirNames = map( lambda x: x.strip(), dirNames)

for myDir in dirNames:
	with open(labsPath+'2new-'+myDir+'-label.csv') as labFile:
		lines = labFile.readlines()
	countLabs = len(lines)
	countData = len(os.listdir(dataPath+myDir))
	masterCount = masterCount + min(countLabs,countData)

print masterCount

f = h5py.File(h5Path+fileName+'BD.h5', 'w')
# 1200 data, each is a 320*240 Img
f.create_dataset('data', (masterCount,1,320,240), dtype='f8')
# Data's labels, each is a 19-dim vector
f.create_dataset('label', (masterCount,19), dtype='f4')

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
	print labCount+step
	print step
	print masterCount	
	f['label'][labCount:labCount+step] = lines[:step]
	labCount = labCount + step
	for num in range(1,step+1):
		lenNum = len(str(num))
		cant0  = 5-lenNum
		pic = "Pictures"+'0'*cant0+str(num)+".jpg"
		im = I.open(dataPath+myDir+'/'+pic)
		picData = np.reshape(np.asarray(im.getdata()),(320,240))
		picList = map(lambda x: map(lambda y: float(y/maxIm),x), list(picData))
		f['data'][picCount] = picList
		picCount = picCount + 1	 	


f.close()
