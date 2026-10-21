import numpy as np, os, json
npz = np.load(os.path.join('processed','test.npz'))
print('y array:', npz['y'])
print('unique labels in test:', np.unique(npz['y']))
