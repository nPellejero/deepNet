import numpy as np
import sys
path = sys.argv[1]
data = np.load(path)
print data
#print np.shape(data['x_train'])
#print np.shape(data['x_valid'])
print np.shape(data['y'])
print np.shape(data['x'])
