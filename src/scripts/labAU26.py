#cuenta cuantos archivos hay con cada cantidad y clase de etiquetas.
# para ejecutar en el dir. AULabels.
import numpy as np
import os, csv, sys

def funAux(myItem, index):
	del myItem[index]
	return myItem
def funShorty(item):
	 item[len(item)-1]=item[len(item)-1][:-2]
rootdir = os.getcwd()
fi = sys.argv[1]
val = sys.argv[2]
print fi
print val
absFile = os.path.join(rootdir, fi)
print "archivo:\t"+absFile+"\n"
with open(absFile,'rw') as f:
	lines = f.readlines()
	lines = map(lambda x: x.split(','), lines)
	if fi.startswith("0eb21") and fi.endswith("-label.csv"):
		ind=lines[0].index("Occlusion")
		lines = map (lambda x: funAux(x,ind),lines)
	if fi.startswith("96e4") and fi.endswith("-label.csv"):
		lines[0].insert(1,"Smile")
		end = len(lines)
		map(lambda x: x.insert(1,"1"),lines[1:])
	if fi.endswith("-label.csv"):
		lines[0].insert(18,"AU26")
		end = len(lines)
		map(lambda x: x.insert(1,val),lines[1:])
		map(funShorty,lines)
		with open(os.path.join(rootdir,"new-"+fi),'w') as myF:
			writer = csv.writer(myF)
			writer.writerows(lines)
	else:
		print "nombre archivo mal ingresado"
			
