import sys
sys.path.append('/home/camp/warnert/neurolytics/')
import binary_recording as br
import joined_recording as jr
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
from cycler import cycler
from matplotlib import cm
import os
from sklearn.svm import LinearSVC
import psutil
from sklearn.model_selection import StratifiedShuffleSplit

available_cpu_count = len(psutil.Process().cpu_affinity())
os.environ["MKL_NUM_THREADS"] = str(available_cpu_count)

home_dir = '/home/camp/warnert/working/Recordings/binary_pulses'

rec1_sb = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_19-56-29/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_ventral.txt'), sniff_basis=True)
rec2_sb = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_16-37-36/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_dorsal.txt'), sniff_basis=True)
rec3_sb = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_16-44-23/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_dorsal.txt'), sniff_basis=True)
rec4_sb = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_19-57-03/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_ventral.txt'), sniff_basis=True)
rec5_sb = br.Binary_recording(os.path.join(home_dir, '200309/2020-03-09_16-20-42/'), 32, os.path.join(home_dir, '200309/2020-03-09trial_name_joined.txt'), sniff_basis=True)
rec6_sb = br.Binary_recording(os.path.join(home_dir, '200311/2020-03-11_16-51-10/'), 32, os.path.join(home_dir, '200311/2020-03-11trial_name_binary_joined.txt'), sniff_basis=True)
rec7_sb = br.Binary_recording(os.path.join(home_dir, '200318/2020-03-18_15-24-43/'), 32, os.path.join(home_dir, '200318/2020-03-18trial_name.txt'), sniff_basis=True)
rec8_sb = br.Binary_recording(os.path.join(home_dir, '200319/2020-03-19_16-08-45/'), 32, os.path.join(home_dir, '200319/2020-03-19_16-08-45_trial_names.txt'), sniff_basis=True)

recs_sb = jr.JoinedRecording(recordings=[rec1_sb, rec2_sb, rec3_sb, rec4_sb, rec5_sb, rec6_sb, rec7_sb, rec8_sb])


rec1 = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_19-56-29/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_ventral.txt'))
rec2 = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_16-37-36/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_dorsal.txt'))
rec3 = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_16-44-23/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_dorsal.txt'))
rec4 = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_19-57-03/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_ventral.txt'))
rec5 = br.Binary_recording(os.path.join(home_dir, '200309/2020-03-09_16-20-42/'), 32, os.path.join(home_dir, '200309/2020-03-09trial_name_joined.txt'))
rec6 = br.Binary_recording(os.path.join(home_dir, '200311/2020-03-11_16-51-10/'), 32, os.path.join(home_dir, '200311/2020-03-11trial_name_binary_joined.txt'))
rec7 = br.Binary_recording(os.path.join(home_dir, '200318/2020-03-18_15-24-43/'), 32, os.path.join(home_dir, '200318/2020-03-18trial_name.txt'))
rec8 = br.Binary_recording(os.path.join(home_dir, '200319/2020-03-19_16-08-45/'), 32, os.path.join(home_dir, '200319/2020-03-19_16-08-45_trial_names.txt'))

recs = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])

unit_count = list(sys.argv)[1]
unit_count = int(unit_count)

## Standard
scores = []
predicts = []
trial_names = ['31_1', '30_1', '28_1', '24_1', '16_1', '0_1']
for j in range(100):
    resps = []
    saved_trials = []
    for i in trial_names:
        xs, resp, saved_trial = recs.get_binned_trial_response(i, pre_trial_window=0, post_trial_window=0.38, w_bootstrap=True, bootstrap_limit=1000, saved_trial=True)
        resps.append(resp)
        saved_trials.append(saved_trial)
    saved_trials = [np.concatenate(i) for i in saved_trials]
    resps = np.array(resps)
    saved_trials = np.array(saved_trials)
    summed_resp = np.sum(resps, axis=-1)
    summed_saved = np.sum(saved_trials, axis=-1)
    reshaped_resps = summed_resp.reshape(len(trial_names)*1000, 145)
    scaler = StandardScaler()
    scaled_reshaped_resps = scaler.fit_transform(reshaped_resps)
    scaled_saved = scaler.transform(summed_saved)
    scaled_resps = scaled_reshaped_resps.reshape(len(trial_names), 1000, 145)
    conjoined_resps = np.concatenate(scaled_resps)
    ys = [i for i in trial_names for j in range(1000)]
    svm = LinearSVC(C=1000)
    unit_indexes = random.sample(range(145), k=unit_count)
    svm.fit(conjoined_resps[:, unit_indexes], ys)
    scores.append(svm.score(scaled_saved[:, unit_indexes], trial_names))
    predicts.append(svm.predict(scaled_saved[:, unit_indexes]))

np.save(os.path.join(home_dir, 'classifier_outputs', '200909', 'conc_%d_summed_count_acc.npy' % unit_count), scores)
np.save(os.path.join(home_dir, 'classifier_outputs', '200909', 'conc_%d_summed_count_predicts.npy' % unit_count), predicts)

## Sniff basis
scores = []
predicts = []
for j in range(100):
    resps = []
    saved_trials = []
    for i in trial_names:
        xs, resp, saved_trial = recs_sb.get_binned_trial_response(i, pre_trial_window=0, post_trial_window=2, bin_size=0.05, w_bootstrap=True, bootstrap_limit=1000, saved_trial=True)
        resps.append(resp)
        saved_trials.append(saved_trial)
    saved_trials = [np.concatenate(i) for i in saved_trials]
    resps = np.array(resps)
    saved_trials = np.array(saved_trials)
    summed_resp = np.sum(resps, axis=-1)
    summed_saved = np.sum(saved_trials, axis=-1)
    reshaped_resps = summed_resp.reshape(len(trial_names)*1000, 145)
    scaler = StandardScaler()
    scaled_reshaped_resps = scaler.fit_transform(reshaped_resps)
    scaled_saved = scaler.transform(summed_saved)
    scaled_resps = scaled_reshaped_resps.reshape(len(trial_names), 1000, 145)
    conjoined_resps = np.concatenate(scaled_resps)
    ys = [i for i in trial_names for j in range(1000)]
    svm = LinearSVC(C=1000)
    unit_indexes = random.sample(range(145), k=unit_count)
    svm.fit(conjoined_resps[:, unit_indexes], ys)
    scores.append(svm.score(scaled_saved[:, unit_indexes], trial_names))
    predicts.append(svm.predict(scaled_saved[:, unit_indexes]))

np.save(os.path.join(home_dir, 'classifier_outputs', '200909', 'conc_%d_summed_count_acc_sb.npy' % unit_count), scores)
np.save(os.path.join(home_dir, 'classifier_outputs', '200909', 'conc_%d_summed_count_predicts_sb.npy' % unit_count), predicts)

## Shuffled
scores = []
predicts = []
for j in range(100):
    resps = []
    saved_trials = []
    for i in trial_names:
        xs, resp, saved_trial = recs.get_binned_trial_response(i, pre_trial_window=0, post_trial_window=0.38, w_bootstrap=True, bootstrap_limit=1000, saved_trial=True)
        resps.append(resp)
        saved_trials.append(saved_trial)
    saved_trials = [np.concatenate(i) for i in saved_trials]
    resps = np.array(resps)
    saved_trials = np.array(saved_trials)
    summed_resp = np.sum(resps, axis=-1)
    summed_saved = np.sum(saved_trials, axis=-1)
    reshaped_resps = summed_resp.reshape(len(trial_names)*1000, 145)
    scaler = StandardScaler()
    scaled_reshaped_resps = scaler.fit_transform(reshaped_resps)
    scaled_saved = scaler.transform(summed_saved)
    scaled_resps = scaled_reshaped_resps.reshape(len(trial_names), 1000, 145)
    conjoined_resps = np.concatenate(scaled_resps)
    ys = [i for i in trial_names for j in range(1000)]
    np.random.shuffle(ys)
    svm = LinearSVC(C=1000)
    unit_indexes = random.sample(range(145), k=unit_count)
    svm.fit(conjoined_resps[:, unit_indexes], ys)
    scores.append(svm.score(scaled_saved[:, unit_indexes], trial_names))
    predicts.append(svm.predict(scaled_saved[:, unit_indexes]))

np.save(os.path.join(home_dir, 'classifier_outputs', '200909', 'conc_%d_summed_count_acc_shuffled.npy' % unit_count), scores)
np.save(os.path.join(home_dir, 'classifier_outputs', '200909', 'conc_%d_summed_count_predicts_shuffled.npy' % unit_count), predicts)
