import numpy as np
import sys
import Image as I
import caffe
import lmdb
from random import shuffle
import os, gc

def memUsage():
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

fileName = sys.argv[1]
commPath = '/home/npellejero/tesis/AMFED/'
labsPath = commPath+'AULabels/'
dataPath = commPath+'images/'
h5Path   = commPath+'redes/miniBDSingleLMDB/'
maxIm = 255
maxLabs = 100

dirCount = 0
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



datas = np.array([])

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
	print "memory en MB: "+str(memUsage()	)
	print "step: "+str(step)
	print "garbage: "+str(gc.garbage)	
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
		datas = np.append(datas,im_dat) 
	if dirCount % 5 == 0:
		im_db = lmdb.open(h5Path+fileName+'image-lmdb', map_size=int(1e12))
		with im_db.begin(write=True) as im_txn:
			shuffle(datas)
			gc.collect()
			print "agregamos algo de shape: "+str(np.shape(datas))
			for num1 in range(len(datas)):
				im_dat = datas[num1]
				im_txn.put('{:0>10d}'.format(num1+accum), im_dat.SerializeToString())
			print "lo vaciamos aca"
			accum = accum + len(datas)
			print "el accum es:"+str(accum)	
			datas = np.array([])
		im_db.close()

