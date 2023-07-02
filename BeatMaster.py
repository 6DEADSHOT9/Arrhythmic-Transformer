import pandas as pd
import numpy as np
import os
import gc
import neurokit2 as nk2
# import warnings
# warnings.filterwarnings("error")

path_NORM = r'./Class01/Not Arrhythmia/NORMAL.npy'
path_ARRHYTHMIA = r'./Class01/Arrhythmia/ARRHYTHMIA.npy'

# Load data
NORM = np.load(path_NORM)
ARRH = np.load(path_ARRHYTHMIA)

# segment the heartbeats for all the classes
# discard the first and last beats

N_beats = []
A_beats = []
count1 = 0
count2 = 0
for i in NORM[:]:
    count1 += 1
    temp_beats = []
    try:
        _, results = nk2.ecg_peaks(i, sampling_rate=500)
        rpeaks = results["ECG_R_Peaks"]
        segmented_beats = nk2.ecg_segment(i, rpeaks, sampling_rate=500, show=False)
    except Exception as e:
        count1 -= 1
        print(f'\nNORM : {count1} encountered {e}]]\n')
        continue
    for i in range(2, int(list(segmented_beats.keys())[-1])):
        signal = segmented_beats[str(i)]['Signal'].to_numpy()
        temp_beats.append(signal)
    N_beats += temp_beats
    print(f'NORM  : {count1}')
    del temp_beats, signal, segmented_beats, rpeaks, results, _
    gc.collect()
print('NORMAL done')
for i in ARRH[:]:
    count2 += 1
    temp_beats = []
    try:
        _, results = nk2.ecg_peaks(i, sampling_rate=500)
        rpeaks = results["ECG_R_Peaks"]
        segmented_beats = nk2.ecg_segment(i, rpeaks, sampling_rate=500, show=False)
    except Exception as e:
        count2 -= 1
        print(f'\nARRHYTHMIA : {count2} encountered {e}]]\n')
        continue
    for i in range(2, int(list(segmented_beats.keys())[-1])):
        signal = segmented_beats[str(i)]['Signal'].to_numpy()
        temp_beats.append(signal)
    A_beats += temp_beats
    print(f'ARRHYTHMIA  : {count2}')
    del temp_beats, signal, segmented_beats, rpeaks, results, _
    gc.collect()
print('ARRHYTHMIA done')
print(f'\n\nNormal heartbeats : {len(N_beats)} \nArrhythmic heartbeats : {len(A_beats)}')

os.system('cls')
# save the segmented beats in a folder

N_beats = np.array(N_beats, dtype = object)
A_beats = np.array(A_beats, dtype = object)

np.save(r'./Heartbeats/Normal/N_beats.npy', N_beats)
np.save(r'./Heartbeats/Arrhythmia/A_beats.npy', A_beats)

print(f'{len(NORM)} Normal heartbeats segmented into {N_beats.shape[0]} and saved')
print(f'{len(ARRH)} Arrhythmic heartbeats segmented into {A_beats.shape[0]} and saved')