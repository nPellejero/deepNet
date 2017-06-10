import numpy as np
import os
# no toma argumentos. para ejecutar en images, images-orig, images-crop.
rootdir = os.getcwd()
newLen = 16

def zeroGen(n):
	string = "" 
	for i in range(n):
		string = string + "0"
	return string

for subdir, dirs, files in os.walk(rootdir):
   for fi in files:
			if fi.endswith(".jpg")
				absFile = os.path.join(subdir, fi)
				pre = fi[:8]
				post = fi[-4:]
				num = fi.replace(pre,"").replace(post,"")
				myLen = 5 - len(num)
				fi1 = pre+zeroGen(myLen)+num+post
				abs1 = os.path.join(subdir, fi1)
				print "renombrando: "+absFile+" a "+abs1
				os.rename(absFile,abs1)
