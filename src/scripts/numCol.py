#cuenta cuantos archivos hay con cada cantidad y clase de etiquetas.

import numpy as np
import os
rootdir = os.getcwd()

myDic = {}; 
for fi in os.listdir(rootdir):
  if fi.endswith("-label.csv"):
        absFile = os.path.join(rootdir, fi)
	#print "archivo:\t"+absFile+"\n"
	with open(absFile) as f:
    		header = f.readlines()[0]
		#print "header:\t"+header+"\n"
		cond = header in myDic
		if cond:
			myDic[header] = myDic[header] + 1;
		else:
			myDic[header] = 1;

	
for k, v in myDic.iteritems():
	print k,v 
	print  "\n\n"
	
print len(myDic.keys())		
