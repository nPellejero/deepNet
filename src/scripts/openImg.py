import Image
import numpy as num
im=Image.open('/home/npellejero/tesis/AMFED/images/ff99a4d9-3c7d-4593-850c-fe8db5097cab/Pictures1.jpg')
imarray=num.array(im)
fig=num.shape(imarray)
print fig
