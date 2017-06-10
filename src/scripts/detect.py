# para ejecutar en el directorio images
# pasa imagenes a escala de grises, equaliza su histograma de intensidad y luego pasa un filtro bilateral.
import numpy as np
import cv2
import os
rootdir = os.getcwd()
face_cascade = cv2.CascadeClassifier('../scripts/haarcascade_frontalface_default.xml')
face_cascade1 = cv2.CascadeClassifier('../scripts/haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier('../scripts/haarcascade_profileface.xml')

contador = 0
contLoc = 0
contLocM = 0
contMaster = 0

for subdir, dirs, files in os.walk(rootdir):
	print subdir 
	for fi in files:
#for fi in os.listdir(rootdir):
		if fi.endswith(".jpg"):
			absFile = os.path.join(subdir, fi)
			img = cv2.imread(absFile) 
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
			cl1 = clahe.apply(img)
			contLocM = contLocM + 1
			faces = face_cascade.detectMultiScale(cl1, 1.3, 5)
			faces1 = face_cascade1.detectMultiScale(cl1, 1.3, 5)
			faces2 = face_cascade2.detectMultiScale(cl1, 1.3, 5)
			totalFace = list(faces)+list(faces1)+list(faces2)
			if len(totalFace) > 0:
				contLoc = contLoc + 1
				#for (x,y,w,h) in faces:
					#print x, y, w, h
          # Dubugging boxes
          # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
					#y1 = y - int((yscale-1.)*h/2.)
					#y2 = y + h + int((yscale-1)*h/2)
					#x1 = x - int((xscale-1.)*w/2.)
					#x2 = x + w + int((xscale-1.)*w/2.)
					#image  = img[y1:y2, x1:x2]
	contador = contador + contLoc
	contMaster = contMaster + contLocM
	print "en este dire encontramos: "+ str(contLoc) + " de " + str(contLocM)
	contLoc = 0
	contLocM = 0
			#cv2.imwrite(absFile,image)
print "encontramos caras en: "+ str(contador) + " de " + str(contMaster) 
