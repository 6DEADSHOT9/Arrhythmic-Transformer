import numpy as np
# import tensorflow as tf

NORM = np.load('Heartbeats/Normal/N_beats.npy',allow_pickle = True)
ARRH = np.load('Heartbeats/Arrhythmia/A_beats.npy',allow_pickle = True)
print('loaded')
NORM = list(NORM)
ARRH = list(ARRH)
print('listed')
# NORM = tf.ragged.constant(NORM)
# ARRH = tf.ragged.constant(ARRH)
# print('ragged')
all_data = NORM + ARRH
all_data = list(all_data)
print('concat')
k = 0
for i in all_data:
    if int(len(list(i))) > k:
        k = int(len(list(i)))
        print(k)
print(f'haha :{k}, {len(all_data)}')