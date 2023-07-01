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
               3214, 3260, 3313, 3452, 3505, 3550, 3552, 3703, 3770, 3814, 3815, 3826,
               3827, 3837, 3970, 3976, 3984, 4015, 4022, 4253, 4262, 4285, 4346, 4414,
               4417, 4429, 4601, 4630, 4678, 4691, 4831, 4848, 4891, 5018, 5028, 5032,
               5046, 5054, 5195, 5227, 5255, 5329, 5460, 5481, 5550, 5909, 5910, 5913,
               5970, 5976, 5987, 5989, 6015, 6030, 6038, 6055, 6056, 6057, 6058, 6067,
               6068, 6081, 6087, 6088, 6090, 6091, 6092, 6093, 6094, 6109, 6110, 6111,
               6124, 6127, 6136, 6137, 6138, 6156, 6160, 6162, 6166, 6168, 6176, 6190,
               6208, 6211, 6213, 6217, 6229, 6235, 6236, 6237, 6242, 6245, 6253, 6255,
               6258, 6311, 6312, 6316, 6326, 6327, 6344, 6345, 6628, 6869, 7533, 7538,
               7635, 8681, 8943, 9969, 10124, 10381, 10492, 10746, 12097, 12676]

errors_Afib = [371,  996, 1019, 1140, 1167, 1198, 1202, 1212, 1213, 1220, 1231, 1244,
              1250, 1253, 1261, 1302, 1304, 1332, 1336, 1373, 1382, 1387, 1390, 1399,
              1428, 1429, 1446, 1452, 1453, 1459, 1482, 1500, 1503, 1542, 1550, 1580,
              1592, 1594, 1607, 1616, 1624, 1636, 1670, 1679, 1695, 1703, 1704, 1723,
              1741, 1752, 1756, 1767, 1785, 1790, 1822, 1834, 1928, 2247]

errors_Afl = [31,  51, 181, 328, 329, 330, 331, 333, 334, 335, 336, 337, 338, 339, 349,
             350, 351, 360, 371, 377, 380, 433, 450, 474, 475, 476, 478, 479, 480, 482,
             494, 568, 859,
             1101, 1204, 1439, 1449, 1457, 1469, 1488, 2118, 2230, 2480, 2709, 2756,
             2778, 2964, 3004, 3142, 3263]
'''
with open('E:/Arrythmia/txt/Normal_Sinus_rhythm.txt', 'r') as f:
    Normal_Sinus_Rhythm = f.readlines()
    Normal_Sinus_Rhythm = [x.strip() for x in Normal_Sinus_Rhythm]
    f.close()
    
with open('E:/Arrythmia/txt/Atrial_Fibrillation.txt', 'r') as f:
    Atrial_Fibrillation = f.readlines()
    Atrial_Fibrillation = [x.strip() for x in Atrial_Fibrillation]
    f.close()
'''
with open('E:/Arrythmia/txt/Atrial_Flutter.txt', 'r') as f:
    Atrial_Flutter = f.readlines()
    Atrial_Flutter = [x.strip() for x in Atrial_Flutter]
    f.close()


Normal_Sinus_rhythm_trace = []
Atrial_Fibrillation_trace = []
Atrial_Flutter_trace = []
'''
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
np.save('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', np.array(Normal_Sinus_rhythm_trace,dtype=object))
# with h5py.File('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', 'w') as hf:
#     hf.create_dataset("Normal_Sinus_rhythm_trace", data=Normal_Sinus_rhythm_trace)
#     hf.close()
print('- - - - - -Normal sinus rhythms saved- - - - - -')
del temp
del Normal_Sinus_rhythm_trace
gc.collect()

for i in range(len(Atrial_Fibrillation)):
    if i in errors_Afib:
        print(f'Error in {i}')
        print()
        continue
    temp = wfdb.io.rdrecord(Atrial_Fibrillation[i])
    temp = temp.__dict__['p_signal']
    try:
        temp = tfm.all_transform(temp)
    except Exception as e:
        # temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
        print(f'Error {e} in {i}')
        print()
        errors_Afib.append(i)
        print(errors_Afib)
        continue
    Atrial_Fibrillation_trace.append(temp)
    print(f'Atrial_Fibrillation_trace {i} saved')
np.save('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', np.array(Atrial_Fibrillation_trace,dtype=object))

# with h5py.File('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', 'w') as hf:
#     hf.create_dataset("Atrial_Fibrillation_trace", data=Atrial_Fibrillation_trace)
#     hf.close()
print('- - - - - -Atrial_Fibrillation_trace saved- - - - - -')
del temp
del Atrial_Fibrillation_trace
gc.collect()
'''
for i in range(len(Atrial_Flutter)):
    if i in errors_Afl:
        print(f'Error in {i}')
        print()
        continue
    temp = wfdb.io.rdrecord(Atrial_Flutter[i])
    temp = temp.__dict__['p_signal']
    try:
        temp = tfm.all_transform(temp)
    except Exception as e:
        # temp = tfm.normalize(tfm.calc_baseline(tfm.powerline(tfm.rms_transform(temp))))
        print(f'Error {e} in {i}')
        print()
        errors_Afl.append(i)
        print(errors_Afl)
        continue
    Atrial_Flutter_trace.append(temp)
    print(f'Atrial_Flutter_trace {i} saved')
np.save('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', np.array(Atrial_Flutter_trace,dtype=object))

# with h5py.File('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', 'w') as hf:
#     hf.create_dataset("Atrial_Flutter_trace", data=Atrial_Flutter_trace)
#     hf.close()
print('- - - - - -Atrial_Flutter_trace saved- - - - - -')
del temp
del Atrial_Flutter_trace
gc.collect()
os.system('cls')
print('All saved')