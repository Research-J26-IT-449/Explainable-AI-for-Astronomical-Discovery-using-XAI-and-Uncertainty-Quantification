import numpy as np, os, json
npz_path = os.path.join('processed','test.npz')
npz = np.load(npz_path)
print('y array:', npz['y'])
print('unique labels:', np.unique(npz['y']))
print('shape X:', npz['X'].shape)
