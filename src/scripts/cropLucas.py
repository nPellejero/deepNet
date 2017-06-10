import cv2
import sys
import glob

#cascPath = "haarcascade_frontalface_default.xml"
cascPath = "haarcascade_frontalface_alt.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

yscale = 1.5
xscale = yscale*8./9.

files=glob.glob("*.jpg")
for file in files:
    print file

    # Read the image
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    found = len(faces)

    print "Found {0} faces!".format(found)

    if found>0:

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            print x, y, w, h

            # Dubugging boxes
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            y1 = y - int((yscale-1.)*h/2.)
            y2 = y + h + int((yscale-1)*h/2)
            x1 = x - int((xscale-1.)*w/2.)
            x2 = x + w + int((xscale-1.)*w/2.)

            image  = image[y1:y2, x1:x2]

            print "cuadro/{0}".format(str(file),str(x))
            cv2.imwrite("cuadro/{0}".format(str(file),str(x)), image)


