import numpy as np
a = []
X_test = np.array(([2, 3, 4],[4,6,8]))
X =  np.array(([4,6,7],[9,2,3]))
a = np.concatenate((X,X_test),axis = 0)
print(a)
a = np.array([X,X_test])

