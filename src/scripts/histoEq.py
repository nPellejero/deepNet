# para ejecutar en el directorio images
# pasa imagenes a escala de grises, equaliza su histograma de intensidad y luego pasa un filtro bilateral.
import numpy as np
import cv2
import os
rootdir = os.getcwd()
#face_cascade = cv2.CascadeClassifier('../scripts/haarcascade_frontalface_default.xml')


for subdir, dirs, files in os.walk(rootdir):
   for fi in files:
#for fi in os.listdir(rootdir):
#  if fi.endswith(".jpg"):
        absFile = os.path.join(subdir, fi)
	print absFile
	img = cv2.imread(absFile) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	#faces = face_cascade.detectMultiScale(cl1, 1.3, 5)
	#for (x,y,w,h) in faces:
	#	roi_gray = cl1[y:y+h, x:x+w]
	
	cl2 = cv2.bilateralFilter(cl1, 8, 25, 25)
	cv2.imwrite(absFile,cl2)
