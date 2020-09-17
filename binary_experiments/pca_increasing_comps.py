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
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
import psutil
import os

available_cpu_count = len(psutil.Process().cpu_affinity())
os.environ["MKL_NUM_THREADS"] = str(available_cpu_count)

home_dir = '/home/camp/warnert/working/Recordings/binary_pulses'

rec1 = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_19-56-29/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_ventral.txt'))
rec2 = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_16-37-36/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_dorsal.txt'))
rec3 = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_16-44-23/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_dorsal.txt'))
rec4 = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_19-57-03/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_ventral.txt'))
rec5 = br.Binary_recording(os.path.join(home_dir, '200309/2020-03-09_16-20-42/'), 32, os.path.join(home_dir, '200309/2020-03-09trial_name_joined.txt'))
rec6 = br.Binary_recording(os.path.join(home_dir, '200311/2020-03-11_16-51-10/'), 32, os.path.join(home_dir, '200311/2020-03-11trial_name_binary_joined.txt'))
rec7 = br.Binary_recording(os.path.join(home_dir, '200318/2020-03-18_15-24-43/'), 32, os.path.join(home_dir, '200318/2020-03-18trial_name.txt'))
rec8 = br.Binary_recording(os.path.join(home_dir, '200319/2020-03-19_16-08-45/'), 32, os.path.join(home_dir, '200319/2020-03-19_16-08-45_trial_names.txt'))

recs = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])

out_scores = os.path.join(home_dir, 'classifier_outputs', '200912', 'all_trials_pca_%d_comps_accs.npy' % int(list(sys.argv)[1]))
out_predicts = os.path.join(home_dir, 'classifier_outputs', '200912', 'all_trials_pca_%d_comps_preds.npy' % int(list(sys.argv)[1]))

comps = int(list(sys.argv)[1])

scores = []
predicts = []
for j in tqdm(range(100), leave=False):
    resps = []
    saved_trials = []
    trial_names = ['%d_1' % i for i in range(32)]
    for i in trial_names:
        xs, resp, saved_trial = recs.get_binned_trial_response(i, pre_trial_window=0, post_trial_window=.88, w_bootstrap=True, bootstrap_limit=1000, saved_trial=True)
        saved_trials.append(saved_trial)
        resps.append(resp)
    saved_trials = [np.concatenate(i) for i in saved_trials]
    resps = np.array(resps)
    saved_trials = np.array(saved_trials)
    reshaped_resps = np.rollaxis(resps, axis=2).reshape(145, 32*1000*100)
    reshaped_saved = np.rollaxis(saved_trials, axis=1).reshape(145, 32*100)
    scaler = StandardScaler()
    scaled_resps = scaler.fit_transform(reshaped_resps.T).T
    scaled_saved = scaler.transform(reshaped_saved.T).T
    reshaped_scaled = scaled_resps.reshape(145, 32, 1000, 100)
    reshaped_scaled = reshaped_scaled.reshape(145*32*1000, 100)

    reshaped_saved = scaled_saved.reshape(145, 32, 100)
    reshaped_saved = reshaped_saved.reshape(145*32, 100)

    pca = PCA(n_components=comps)
    pcad_reshaped = pca.fit_transform(reshaped_scaled)
    pcad_saved_reshaped = pca.transform(reshaped_saved)
    pcad_resps = pcad_reshaped.reshape(145, 32, 1000, comps)
    pcad_saved = pcad_saved_reshaped.reshape(145, 32, comps)
    reshaped_pcad = np.concatenate(np.moveaxis(pcad_resps, 0, 2)).T
    ys = [i for i in trial_names for j in range(1000)]
    svm = LinearSVC(C=1000)
    data = np.concatenate(reshaped_pcad).T
    svm.fit(data, ys)
    score = svm.score(np.concatenate(np.rollaxis(pcad_saved, axis=-1)).T, trial_names)[np.newaxis]
    preds = svm.predict(np.concatenate(np.rollaxis(pcad_saved, axis=-1)).T)[np.newaxis]
    if os.path.isfile(out_scores):
        prev_scores = np.load(out_scores)
        score = np.append(prev_scores, score, axis=0)
    if os.path.isfile(out_predicts):
        prev_preds = np.load(out_predicts)
        preds = np.append(prev_preds, preds, axis=0)
    np.save(out_scores, score)
    np.save(out_predicts, preds)