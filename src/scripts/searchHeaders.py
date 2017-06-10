#busca headers que cumplen condicion.
# para ejecutar en AULabels
import numpy as np
import os
rootdir = os.getcwd()

for fi in os.listdir(rootdir):
  if fi.endswith("-label.csv"):
        absFile = os.path.join(rootdir, fi)
	#print "archivo:\t"+absFile+"\n"
	with open(absFile) as f:
    		header = f.readlines()[0]
		#print "header:\t"+header+"\n"
		cond = len(header) > 300 
		if cond:
			print absFile
			#print header
