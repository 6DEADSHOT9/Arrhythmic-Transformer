import transformations as tfm
import warnings
warnings.filterwarnings("error")
from timeout_decorator import timeout
import numpy as np
import pandas as pd
import wfdb
import os
import h5py
import gc

snomed_ct_dict = {'Atrial fibrillation':164889003,'Atrial flutter':164890007,'Normal sinus rhythm':426783006}
training_path = r'Data/The PhysioNetComputing in Cardiology Challenge 2021/physionet.org/files/challenge-2021/1.0.3/training/'
errors_Norm = [ 288,  393, 1170, 1175, 1200, 1218, 1251, 1255, 1261, 1265, 1285, 1287,
               1298, 1335, 1352, 1375, 1396, 1420, 1431, 1450, 1468, 1497, 1503, 1507,
               1509, 1542, 1562, 1566, 1582, 1604, 1605, 1622, 1651, 1662, 1663, 1682,
               2361, 2376, 2612, 2623, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734,
               2735, 2736, 2739, 2742, 2746, 2747, 2748, 2749, 2750, 2751, 2753, 2755,
               2756, 2757, 2758, 2759, 2760, 2761, 2762, 2764, 2768, 2769, 2770, 2771,
               2772, 2773, 2776, 2777, 2778, 2779, 2780, 2781, 2783, 2784, 2785, 2786,
               2787, 2788, 2789, 2790, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2801,
               2802, 2803, 2804, 2805, 2806, 2808, 2809, 2810, 2811, 2812, 2813, 2814,
               2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826,
               2827, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2840,
               2841, 2842, 2843, 2845, 2846, 2847, 2849, 2850, 2851, 2853, 2854, 2855,
               2858, 2860, 2861, 2862, 2864, 2867, 2916, 2917, 2973, 2981, 3128, 3164,
               3214, 3260, 3313, 3452, 3505, 3550, 3552, 3703, 3770, 3814, 3815]



lis = os.listdir(training_path)
temp = []
temp2 = []
for i in lis:
    if os.path.isdir(training_path+i):
        temp.append(i)
        tmp = []
        for j in os.listdir(training_path+i):
            if os.path.isdir(training_path+i+'/'+j):
                tmp.append(j)
        temp2.append(tmp)
    else:
        continue

all_records = []
all_records_path = []
for i in range(len(temp)):
    for j in temp2[i]:
        with open (training_path+temp[i]+'/'+j+'/'+'RECORDS','r',newline='') as f:
            for k in f.readlines():
                all_records.append(k.split('\n')[0])
                all_records_path.append(training_path+temp[i]+'/'+j+'/'+k.split('\n')[0])

# Filter Normal Sinus Rhythm, Atrial Fibrillation, Atrial Flutter
all_num = len(all_records_path)
sample = []
Normal_Sinus_Rhythm = []
Atrial_Fibrillation = []
Atrial_Flutter = []
for i in all_records_path:
    try:
        ecg = wfdb.io.rdheader(i)
        dct = ecg.__dict__
        if int(dct['comments'][2].split(': ')[1].split(',')[0]) == snomed_ct_dict['Normal sinus rhythm']:
            Normal_Sinus_Rhythm.append(i)
            sample.append(i)
        elif int(dct['comments'][2].split(': ')[1].split(',')[0]) == snomed_ct_dict['Atrial fibrillation']:
            Atrial_Fibrillation.append(i)
            sample.append(i)
        elif int(dct['comments'][2].split(': ')[1].split(',')[0]) == snomed_ct_dict['Atrial flutter']:
            Atrial_Flutter.append(i)
            sample.append(i)
        else:
            all_records_path.remove(i)
            all_records.remove(i.split('/')[-1])

    except FileNotFoundError:
        print(f'File {i} Not Found')
        all_records_path.remove(i)
        all_records.remove(i.split('/')[-1])
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        print(f'{all_num-len(all_records_path)} files done out of {all_num}\n{len(all_records_path)} left out of {all_num}')
        break
    except:
        print(f'Error in {i}')
        print()
        all_records_path.remove(i)
        all_records.remove(i.split('/')[-1])

del temp, temp2, all_num, i, j, k, ecg, dct, sample, all_records_path, all_records

Normal_Sinus_rhythm_trace = []
Atrial_Fibrillation_trace = []
Atrial_Flutter_trace = []

for i in range(len(Normal_Sinus_Rhythm)):
    if i in errors_Norm:
        print(f'Error in {i}')
        print()
        continue
    temp = wfdb.io.rdrecord(Normal_Sinus_Rhythm[i])
    temp = temp.__dict__['p_signal']
    try:
        
        temp = tfm.all_transform(temp)
    except Exception as e:
        # temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
        print(f'Error {e} in {i}')
        print()
        errors_Norm.append(i)
        print(errors_Norm)
        continue
    Normal_Sinus_rhythm_trace.append(temp)
    print(f'Normal_Sinus_rhythm_trace {i} saved')

with h5py.File('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', 'w') as hf:
    hf.create_dataset("Normal_Sinus_rhythm_trace", data=Normal_Sinus_rhythm_trace)
    hf.close()
print('- - - - - -Normal sinus rhythms saved- - - - - -')
del temp
del Normal_Sinus_rhythm_trace
gc.collect()

for i in range(len(Atrial_Fibrillation)):
    temp = wfdb.io.rdrecord(Atrial_Fibrillation[i])
    temp = temp.__dict__['p_signal']
    try:
        temp = tfm.all_transform(temp)
    except Exception as e:
        # temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
        print(f'Error {e} in {i}')
        print()
        continue
    Atrial_Fibrillation_trace.append(temp)
    print(f'Atrial_Fibrillation_trace {i} saved')

with h5py.File('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', 'w') as hf:
    hf.create_dataset("Atrial_Fibrillation_trace", data=Atrial_Fibrillation_trace)
    hf.close()
print('- - - - - -Atrial_Fibrillation_trace saved- - - - - -')
del temp
del Atrial_Fibrillation_trace
gc.collect()

for i in range(len(Atrial_Flutter)):
    temp = wfdb.io.rdrecord(Atrial_Flutter[i])
    temp = temp.__dict__['p_signal']
    try:
        temp = tfm.all_transform(temp)
    except Exception as e:
        # temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
        print(f'Error {e} in {i}')
        print()
        continue
    Atrial_Flutter_trace.append(temp)
    print(f'Atrial_Flutter_trace {i} saved')

with h5py.File('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', 'w') as hf:
    hf.create_dataset("Atrial_Flutter_trace", data=Atrial_Flutter_trace)
    hf.close()
print('- - - - - -Atrial_Flutter_trace saved- - - - - -')
del temp
del Atrial_Flutter_trace
gc.collect()


# with h5py.File('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', 'w') as hf:
#     hf.create_dataset("Normal_Sinus_rhythm_trace", shape = (0,5000),maxshape = (None,None))
#     hf.close()
# with h5py.File('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', 'w') as hf:
#     hf.create_dataset("Atrial_Fibrillation_trace", shape = (0,5000),maxshape = (None,None))
#     hf.close()
# with h5py.File('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', 'w') as hf:
#     hf.create_dataset("Atrial_Flutter_trace", shape = (0,5000),maxshape = (None,None))
#     hf.close()

# for i in range(len(Normal_Sinus_Rhythm)):
#     with h5py.File('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', 'a') as file:
#         file["Normal_Sinus_rhythm_trace"].resize((file["Normal_Sinus_rhythm_trace"].shape[0] + 1), axis=0)
#         temp = wfdb.io.rdrecord(Normal_Sinus_Rhythm[i])
#         temp = temp.__dict__['p_signal']
#         if temp.shape[0]>5000:
#             file["Normal_Sinus_rhythm_trace"].resize((temp.shape[0]), axis=1)
#         try:
#             temp = tfm.all_transform(temp)
#         except Exception as e:
#             temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
#             print(f'Error {e} in {i}')
#             print()
#             continue
#         file["Normal_Sinus_rhythm_trace"][-1:] = temp
#         del temp
#         gc.collect()
#         file.close()
#     print(f'Normal_Sinus_rhythm_trace {i} saved')
# print('- - - - - -Normal sinus rhythms saved- - - - - -')

# for i in range(len(Atrial_Fibrillation)):
#     with h5py.File('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', 'a') as file:
#         file["Atrial_Fibrillation_trace"].resize((file["Atrial_Fibrillation_trace"].shape[0] + 1), axis=0)
#         temp = wfdb.io.rdrecord(Atrial_Fibrillation[i])
#         temp = temp.__dict__['p_signal']
#         if temp.shape[0]>5000:
#             file["Atrial_Fibrillation_trace"].resize((temp.shape[0]), axis=1)
#         try:
#             temp = tfm.all_transform(temp)
#         except Exception as e:
#             temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
#             print(f'Error {e} in {i}')
#             print()
#             continue
#         file["Atrial_Fibrillation_trace"][-1:] = temp
#         del temp
#         gc.collect()
#         file.close()
#     print(f'Atrial_Fibrillation_trace {i} saved')
# print('- - - - - -Atrial_Fibrillation_trace saved- - - - - -')

# for i in range(len(Atrial_Flutter)):
#     with h5py.File('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', 'a') as file:
#         file["Atrial_Flutter_trace"].resize((file["Atrial_Flutter_trace"].shape[0] + 1), axis=0)
#         temp = wfdb.io.rdrecord(Atrial_Flutter[i])
#         temp = temp.__dict__['p_signal']
#         if temp.shape[0]>5000:
#             file["Atrial_Flutter_trace"].resize((temp.shape[0]), axis=1)
#         try:
#             temp = tfm.all_transform(temp)
#         except Exception as e:
#             temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
#             print(f'Error {e} in {i}')
#             print()
#             continue
#         file["Atrial_Flutter_trace"][-1:] = temp
#         del temp
#         gc.collect()
#         file.close()
#     print(f'Atrial_Flutter_trace {i} saved')
# print('- - - - - -Atrial_Flutter_trace saved- - - - - -')
os.system('cls')
print('All saved')