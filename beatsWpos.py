import tensorflow as tf
import numpy as np
import os
import gc
import time


NORM = np.load('Heartbeats/Normal/N_beats.npy',allow_pickle = True)
ARRH = np.load('Heartbeats/Arrhythmia/A_beats.npy',allow_pickle = True)

norm_p = r'./Folder1/norm/'
arrh_p = r'./Folder1/arrh/'

print('loaded')
NORM = list(NORM)
ARRH = list(ARRH)
print('listed')

NORM = tf.ragged.constant(NORM)
ARRH = tf.ragged.constant(ARRH)
print('ragged')

arrh_labels = np.array(tf.cast(np.array(np.ones(int(ARRH.shape[0]))),tf.float16))
norm_labels = np.array(tf.cast(np.array(np.zeros(int(NORM.shape[0]))),tf.float16))

def one_hot(beat, d_model):
    if type(d_model) != int:
        raise TypeError("d_model must be an int type")
    encodings = []
    beat = tf.cast((tf.cast(beat,tf.float16)*int(d_model)),tf.int64)
    
    # print(beat.dtype) # troubleshooting
    
    for i in beat:
#         i = tf.round(i)
        i = tf.cast(i,tf.int32)
        # print(i,i.dtype)
        if i == 0:
            encodings.append(list(np.zeros(d_model)))
        else:
            encodings.append(list(np.zeros(i-1)) + list(np.ones(1)) + list(np.zeros(d_model-i)))
    return encodings

def input_embeddings(encodings,max_sequence_length,d_model):
    
    temp = int(max_sequence_length) - int(len(list(encodings)))
    for i in range(temp):
        encodings.append(list(np.zeros(int(d_model))))
    return encodings

os.system('cls')

t1 = time.time()
#163323
count = 0
for i in NORM:
    temp = np.array(one_hot(i,512)).astype(dtype=np.int8)
    np.save(norm_p + str(count),temp)
    count += 1
    if count % 1000 == 0:
        print(count)
    # print estimated time for completion
        t2 = time.time()
        print('Estimated time for completion NORM: ',(t2-t1)*(130665/1000)/60,' minutes')
        t1 = time.time()
    del temp
    gc.collect()

count = 0
for i in ARRH:
    temp = np.array(one_hot(i,512)).astype(dtype=np.int8)
    np.save(arrh_p + str(count),temp)
    count += 1
    if count % 1000 == 0:
        print(count)
    # print estimated time for completion
        t2 = time.time()
        print('Estimated time for completion ARRH: ',(t2-t1)*(163323/1000)/60,' minutes')
        t1 = time.time()
    del temp
    gc.collect()

print('------------------ Done ------------------')
