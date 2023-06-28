import numpy as np
import pandas as pd
import wfdb
import os
import transformations as tfm
import h5py
import gc

snomed_ct_dict = {'Atrial fibrillation':164889003,'Atrial flutter':164890007,'Normal sinus rhythm':426783006}
training_path = r'Data/The PhysioNetComputing in Cardiology Challenge 2021/physionet.org/files/challenge-2021/1.0.3/training/'

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

with h5py.File('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', 'w') as hf:
    hf.create_dataset("Normal_Sinus_rhythm_trace", shape = (0,5000),maxshape = (None,5000))
    hf.close()
with h5py.File('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', 'w') as hf:
    hf.create_dataset("Atrial_Fibrillation_trace", shape = (0,5000),maxshape = (None,5000))
    hf.close()
with h5py.File('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', 'w') as hf:
    hf.create_dataset("Atrial_Flutter_trace", shape = (0,5000),maxshape = (None,5000))
    hf.close()

for i in range(len(Normal_Sinus_Rhythm)):
    with h5py.File('E:/Arrythmia/Converted data/Normal sinus rhythm/' + 'Normal_Sinus_rhythm', 'a') as file:
        file["Normal_Sinus_rhythm_trace"].resize((file["Normal_Sinus_rhythm_trace"].shape[0] + 1), axis=0)
        temp = wfdb.io.rdrecord(Normal_Sinus_Rhythm[i])
        temp = temp.__dict__['p_signal']
        temp = tfm.all_transform(temp)
        file["Normal_Sinus_rhythm_trace"][-1:] = temp
        del temp
        gc.collect()
        file.close()
    print(f'Normal_Sinus_rhythm_trace {i} saved')
print('- - - - - -Normal sinus rhythms saved- - - - - -')

for i in range(len(Atrial_Fibrillation)):
    with h5py.File('E:/Arrythmia/Converted data/Atrial fibrillation/' + 'Atrial_Fibrillation', 'a') as file:
        file["Atrial_Fibrillation_trace"].resize((file["Atrial_Fibrillation_trace"].shape[0] + 1), axis=0)
        temp = wfdb.io.rdrecord(Atrial_Fibrillation[i])
        temp = temp.__dict__['p_signal']
        temp = tfm.all_transform(temp)
        file["Atrial_Fibrillation_trace"][-1:] = temp
        del temp
        gc.collect()
        file.close()
    print(f'Atrial_Fibrillation_trace {i} saved')
print('- - - - - -Atrial_Fibrillation_trace saved- - - - - -')

for i in range(len(Atrial_Flutter)):
    with h5py.File('E:/Arrythmia/Converted data/Atrial flutter/' + 'Atrial_Flutter', 'a') as file:
        file["Atrial_Flutter_trace"].resize((file["Atrial_Flutter_trace"].shape[0] + 1), axis=0)
        temp = wfdb.io.rdrecord(Atrial_Flutter[i])
        temp = temp.__dict__['p_signal']
        temp = tfm.all_transform(temp)
        file["Atrial_Flutter_trace"][-1:] = temp
        del temp
        gc.collect()
        file.close()
    print(f'Atrial_Flutter_trace {i} saved')
print('- - - - - -Atrial_Flutter_trace saved- - - - - -')
os.system('cls')
print('All saved')