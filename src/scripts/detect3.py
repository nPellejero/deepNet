# para ejecutar en el directorio images-originals
# pasa imagenes a escala de grises, equaliza su histograma de intensidad y luego detecta la cara mas grande, recorta la imagen y la guarda en images-croped.
import numpy as np
import cv2
import os,sys
rootdir = os.getcwd()
face_cascade = cv2.CascadeClassifier('../scripts/haarcascade_frontalface_default.xml')
face_cascade1 = cv2.CascadeClassifier('../scripts/haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier('../scripts/haarcascade_profileface.xml')
pathCroped = "/home/npellejero/tesis/AMFED/images-croped2/"
pathOrig = "/home/npellejero/tesis/AMFED/images-originals/"
contador = 0
contLoc = 0
contLocM = 0
contMaster = 0
fileName = sys.argv[1]

def detect(img, cascade):
		detectTot = []
    #for scale in [float(i)/10 for i in range(12, 12)]:
    #    for neighbors in range(3,4):
		scale = 1.1
		neighbors = 3     
		rects = cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=neighbors, minSize=(50, 50), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		detectTot = detectTot + list(rects)
		return detectTot


def find_face_from_img(img):
    rects1 = detect(img, face_cascade)
    rects2 = detect(img, face_cascade1)
    rects3 = detect(img, face_cascade2)
    return rects1 + rects2 + rects3 

def funaux(x):
	if abs(x[1][3] - x[1][2]) < 10: 
			return x[1][3]/x[1][2]

	else:
			return 0 
with open(fileName) as f:
  dirNames = f.readlines()
  dirNames = map( lambda x: x.strip(), dirNames)


for myDir in dirNames:
	print myDir 
	pathHastaDir = pathOrig+myDir+"/"
	pathHastaNewDir = pathCroped+myDir+"/"
	try: 
		os.makedirs(pathHastaNewDir)
	except OSError:
		if not os.path.isdir(pathHastaNewDir):
			raise
	for fi in os.listdir(myDir):
			absFile = pathHastaDir+fi
			img = cv2.imread(absFile) 
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2))	
			cl1 = clahe.apply(img)
			contLocM = contLocM + 1
			totalFace = find_face_from_img(cl1)
			if len(totalFace) > 0:
				contLoc = contLoc + 1
				totalFace = map(lambda x: list(x),totalFace)
				max_index, max_value = max(enumerate(totalFace), key=lambda x: funaux(x) )
				x,y,w,h = max_value
				#print x, y, w, h
        #Dubugging boxes
				#cv2.rectangle(cl1, (x, y), (x+w, y+h), (0, 255, 0), 2)
				y1 = y - int(h/4.)
				y2 = y + h + int(h/4.)
				x1 = x - int(w/8.)
				x2 = x + w + int(w/8.)
				image  = cl1[y1:y2, x1:x2]
				cv2.imwrite(pathCroped+myDir+"/"+fi,image)
	contador = contador + contLoc
	contMaster = contMaster + contLocM
	print "en este dire encontramos: "+ str(contLoc) + " de " + str(contLocM)
	contLoc = 0
	contLocM = 0
print "encontramos caras en: "+ str(contador) + " de " + str(contMaster) 
