import h5py, os
import numpy as np
import sys
import Image as I
import caffe
fileName = sys.argv[1]
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'AULabels/'
dataPath = commPath+'images/'
h5Path   = commPath+'redes/hdf519Labs/'
maxIm = 255
maxLabs = 100

picCount = 0
labCount = 0
dirCount = 0
cantPerDir = 10
masterCount = 0
numLab = 0

def myFun(y):
	y = float(y)
	if y > 0.1:
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
    lines = labFile.readlines()
  countLabs = len(lines)
  countData = len(os.listdir(dataPath+myDir))
  masterCount = masterCount + min(countLabs,countData)

print masterCount

f = h5py.File(h5Path+fileName+'BD.h5', 'w')
# 1200 data, each is a 320*240 Img
f.create_dataset('data', (masterCount,1,240,320), dtype='f8')
# Data's labels, each is a 19-dim vector
f.create_dataset('label', (masterCount,19), dtype='f4')

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
	f['label'][labCount:labCount+step] = lines[:step]
	labCount = labCount + cantPerDir
	for num in range(1,step+1):
		lenNum = len(str(num))
		cant0  = 5-lenNum
		pic = "Pictures"+'0'*cant0+str(num)+".jpg"
		image = caffe.io.load_image(dataPath+myDir+'/'+pic,False)
		image = np.transpose(image,(2,0,1))
		#image = caffe.io.resize( image, (1, 240, 320) ) # resize to fixed size
		f['data'][picCount] = image
		picCount = picCount + 1	 	


f.close()
