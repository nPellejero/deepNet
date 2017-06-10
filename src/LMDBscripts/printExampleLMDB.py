import caffe
import lmdb
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import Image
import itertools 
import numpy as np
lmdb_env = lmdb.open('../redes/miniBDSingleLMDB/xabcimage-lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
		datum.ParseFromString(value)
		label = datum.label
		print "label["+str(key)+"] de test: "
		data = caffe.io.datum_to_array(datum)
		data = np.transpose(data,(1,2,0))
		data = map (lambda x: map(lambda y: int(y*255),x), data)
		data = list(itertools.chain.from_iterable(data))
		print np.shape(data)
		print label
		#print len(data[0])
		#print len(data[0][0])
		#new_img = Image.new("L", (320, 240))
		#new_img.putdata(data)
		#new_img.save('picture_'+str(key)+"-"+str(label)+'_.png')		
