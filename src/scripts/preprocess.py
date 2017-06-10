import cv2
import numpy as np
import PIL
import Image
import os,sys
import sys, math, Image
commonDir = "/home/npellejero/tesis/AMFED/src/scripts"
face_cascade = cv2.CascadeClassifier(commonDir+'/haarcascade_frontalface_default.xml')
face_cascade1 = cv2.CascadeClassifier(commonDir+'/haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier(commonDir+'/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(commonDir+'/haarcascade_eye.xml')

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  referenceH = dest_sz[0] - 2.0*offset_h
  referenceV = dest_sz[0] - 2.0*offset_v
  # scale factor
  scaleH = float(dist)/float(referenceH)
  scaleV = float(dist)/float(referenceV)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scaleH*offset_h, eye_left[1] - scaleV*offset_v)
  crop_size = (dest_sz[0]*scaleH, dest_sz[1]*scaleV)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image

def detect(img, cascade, cascade1, cascade_eyes):
  clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2))
  img = clahe.apply(img)
  clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2))
  img = clahe.apply(img)
  best_diff = 1000
  imgF = None
  for scale in [float(i)/10 for i in range(11, 15)]:
     face = cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=6, minSize=(30, 30))
     if len(face) != 1:
      continue
     x,y,w,h = face[0]
     m = max(w,h)
     y1 = max(y-m/3, 0)
     y2 = min(y + m + m/3, len(img))
     x1 = max(x-m/3, 0)
     x2 = min(x + m + m/3, len(img[0]))
     image = img[y1:y2, x1:x2]
     eyes = cascade_eyes.detectMultiScale(image, minNeighbors=3, scaleFactor=scale)
     if len(eyes) != 2:
      continue
     my_eye_right = max(eyes, key=lambda x:x[0])
     my_eye_left  = min(eyes, key=lambda x:x[0])
     dim_r = my_eye_right[2:]
     dim_l = my_eye_left[2:]
     cel = (my_eye_left[0]+dim_l[0]/2,my_eye_left[1]+dim_l[1]/2)
     cer = (my_eye_right[0]+dim_r[0]/2, my_eye_right[1]+dim_r[1]/2)
     #cv2.rectangle(image,cer,(cer[0]+5,cer[1]+5),3)
     #cv2.rectangle(image,cel,(cel[0]+5,cel[1]+5),3)
     prom = (cer[0] + cel[0])/2.0
     width = x2-x1
     height = y2-y1
     width = x2-x1
     height = y2-y1
     w1 = width*(1.0/3.0)
     w2 = width*(2.0/3.0)
     diff = abs(cer[0]-w2) + abs(cel[0]-w1)
     if diff/float(width) > 0.2 or cel[1] > height/2 or cer[1] > height/2:
      continue
     print diff, width, w1, w2, abs(cel[0]-w1), abs(cer[0]-w2)
     if best_diff > diff:
      best_diff = diff
      cerF = cer
      celF = cel
      imgF = image
  if imgF == None:
   return []
  im = Image.fromarray(imgF)
  im = CropFace(im, eye_left=celF, eye_right=cerF, offset_pct=(0.2,0.3), dest_sz=(128,128))
  image = np.array(im)
  return [image]


def find_face_from_img(img):
    rects1 = detect(img, face_cascade,face_cascade2, eye_cascade )
    return rects1

def funaux(x):
 if abs(x[1][3] - x[1][2]) < 40 and x[1][0] > 50 and x[1][0] < 270 and x[1][1] > 100:
   return x[1][3]+x[1][2]
 else:
   return 0

