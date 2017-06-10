#! python
import numpy as np

hola = np.load('/home/npellejero/public_html/index-files/imgToClasify/salida.npy')
hola = hola*100
np.savetxt('/home/npellejero/public_html/index-files/imgToClasify/salida.txt',hola,fmt='%f')
