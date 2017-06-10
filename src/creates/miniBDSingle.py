import h5py, os
import numpy as np
import sys
import Image as I
import caffe

fileName = sys.argv[1]
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'AULabels/'
dataPath = commPath+'images/'
h5Path   = commPath+'redes/miniBDSingle/'
maxIm = 255
maxLabs = 100

picCount = 0
labCount = 0
dirCount = 0
cantPerDir = 100

numLab = 0

def myFun(y):
	y = float(y)
	if y > 0.5:
		return 1
	else:
		return 0
def refactor(lines):
	lines = lines[1:]
	lines = map(lambda x: x.split(' '),lines)
	lines = map(lambda x: map( lambda y: y.strip(),x), lines)	
	lines = map(lambda x: map( lambda y: myFun(y),x), lines)	
	return map(lambda x: x[0],lines)

with open(fileName) as f:
	dirNames = f.readlines()
	dirNames = map( lambda x: x.strip(), dirNames)

cantPics = len(dirNames)*cantPerDir

X = [] 
y = []

print len(dirNames)
for myDir in dirNames:
	print dirCount 
	print myDir
	dirCount = dirCount + 1
	with open(labsPath+'2new-'+myDir+'-label.csv') as labFile:
		lines = labFile.readlines()
		lines = refactor(lines)	
	print labCount+cantPerDir
	print cantPerDir
	
	for num in range(1,cantPerDir+1):
		y.append( lines[num] )
		lenNum = len(str(num))
		cant0  = 5-lenNum
		pic = "Pictures"+'0'*cant0+str(num)+".jpg"
		im = caffe.io.load_image(dataPath+myDir+'/'+pic, False)
		#im = np.asarray(I.open(  dataPath+myDir+'/'+pic))
		im = im.transpose((2, 0, 1))
		#im = map (lambda x: map(lambda y: float(y/255),x), im[0])
		#print (np.shape(im))	
		X.append( im )
		picCount = picCount + 1	 	
	labCount = labCount + cantPerDir
print type( X ) 

f = h5py.File(h5Path+fileName+'BD.h5', 'w')
f.create_dataset('data',data=X , dtype='f')
f.create_dataset('label',data=y, dtype='f')


f.close()
