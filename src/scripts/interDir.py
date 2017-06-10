#! python
import numpy as np
import sys

folder = sys.argv[1]
hola = np.load('/home/npellejero/public_html/index_files/imgToClasify/'+folder+'/images/salida.npy')
np.savetxt('/home/npellejero/public_html/index_files/imgToClasify/'+folder+'/images/salida.txt',hola,fmt='%f')
