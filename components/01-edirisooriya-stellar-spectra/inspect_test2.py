import numpy as np, os, json
npz = np.load(os.path.join('processed','test.npz'))
print('y true array:', npz['y'])
print('unique labels:', np.unique(npz['y']))
print('number of samples:', npz['y'].shape[0])
