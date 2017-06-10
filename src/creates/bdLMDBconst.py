import numpy as np
import sys
import Image as I
import caffe
import lmdb
from random import shuffle
import os

fileName = sys.argv[1]
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'AULabels/'
dataPath = commPath+'images/'
h5Path   = commPath+'redes/miniBDSingleLMDB/'
maxIm = 255
maxLabs = 100

picCount = 0
labCount = 0
dirCount = 0
cantPerDir = 200
accum = 0

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

im_db = lmdb.open(h5Path+fileName+'image-lmdb', map_size=int(1e12))
im_txn = im_db.begin(write=True) 
	
print len(dirNames)
for myDir in dirNames:
	print dirCount 
	print myDir
	with open(labsPath+'2new-'+myDir+'-label.csv') as labFile:
		lines = labFile.readlines()
		lines = refactor(lines)	
	step = min( len(lines), cantPerDir)
	for num in range(1,step):
		lenNum = len(str(num))
		cant0  = 5-lenNum
		pic = "Pictures"+'0'*cant0+str(num)+".jpg"
		image = caffe.io.load_image(dataPath+myDir+'/'+pic,False)
		image = image.transpose((2, 0, 1))
		image = image.astype(np.float, copy=False)
		lab = lines[num]
		im_dat = caffe.io.array_to_datum(image)
		im_dat.label = int(lab)
		im_txn.put('{:0>10d}'.format(num+accum), im_dat.SerializeToString())
		accum = accum + step
	dirCount = dirCount + 1
im_db.close()


