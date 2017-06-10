import numpy as np

file=np.fromfile('/home/npellejero/disruptive/basesDeDatos/IMFDB_3EmocSquash/mean.jpg')
a=np.asarray(file)
np.save("mean.npy",a)
