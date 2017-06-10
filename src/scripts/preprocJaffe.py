# para ejecutar en el directorio images
# pasa imagenes a escala de grises, equaliza su histograma de intensidad y luego pasa un filtro bilateral.
import numpy as np
import cv2
import os
import preprocess

contador = 0
contLoc = 0
contLocM = 0
contMaster = 0
rootdir = "/home/npellejero/tesis/AMFED/jaffe"
newDir = "/home/npellejero/tesis/AMFED/jaffePreproc"
for subdir, dirs, files in os.walk(rootdir):
	print subdir 
	for fi in files:
#for fi in os.listdir(rootdir):
		if fi.endswith(".png"):
			absFile = os.path.join(subdir, fi)
			print absFile
			img = cv2.imread(absFile,0)
			totalFace = preprocess.find_face_from_img(img)
			contLocM = contLocM + 1
			if len(totalFace) > 0:
				contLoc = contLoc + 1
				newAbsFile = os.path.join(newDir, fi)
				print newAbsFile  
				cv2.imwrite(newAbsFile,totalFace[0])
	contador = contador + contLoc
	contMaster = contMaster + contLocM
	print "en este dire encontramos: "+ str(contLoc) + " de " + str(contLocM)
	contLoc = 0
	contLocM = 0
print "encontramos caras en: "+ str(contador) + " de " + str(contMaster) 
