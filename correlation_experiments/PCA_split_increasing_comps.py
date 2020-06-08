# Classifiers on data projected onto PCs of only the training data
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate
from  sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from copy import deepcopy
import time
import os
from scipy.fftpack import fft, fftfreq

temp_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/PCA/Correlation_PCA_outputs/Increasing_components/temp'
unit_response = np.load(os.path.join(temp_dir, 'unit_response.npy'))
y_var = np.load(os.path.join(temp_dir, 'y_var.npy'))
sss = StratifiedShuffleSplit(n_splits=1000, test_size=2, random_state=0)
split_indexes = sss.split(unit_response, y_var)

split_indexes = list(split_indexes)

trial_index = int(list(sys.argv)[1])

train_indexes, test_indexes = split_indexes[trial_index]

window_sizes = [None, 5, 10, 50, 100, 200]
window_train_accs = []
window_comps = []
window_test_accs = []
window_shuf = []

for ws in window_sizes:
    if ws is None:

        X_train = unit_response[train_indexes]
        X_test = unit_response[test_indexes]

    else:
        snipped_response = np.cumsum(unit_response, dtype=float, axis=-1)
        snipped_response[:, :, ws:] = snipped_response[:, :, ws:] - snipped_response[:, :, :-ws]
        snipped_response = snipped_response[:, :, :1 - ws] / ws

        X_train = snipped_response[train_indexes]
        X_test = snipped_response[test_indexes]

    train_shape = np.array(X_train.shape)
    test_shape = np.array(X_test.shape)

    train_shape[-1] = train_shape[-1] - 400
    test_shape[-1] = test_shape[-1] - 400

    X_train = np.concatenate(X_train)[:, 400:]
    X_test = np.concatenate(X_test)[:, 400:]

    y_train = np.array(y_var)[train_indexes]
    y_test = np.array(y_var)[test_indexes]

    n_components = X_train.shape[-1]
    pca = PCA(n_components=n_components)

    pcad_train = pca.fit_transform(X_train)
    pcad_test = pca.transform(X_test)

    scaler = StandardScaler()
    pcad_train = scaler.fit_transform(pcad_train)
    pcad_test = scaler.transform(pcad_test)

    pcad_train = np.reshape(pcad_train, train_shape)
    pcad_test = np.reshape(pcad_test, test_shape)

    all_accs = []
    all_comps = []
    svm = LinearSVC(C=1000)
    test_accs = []
    shuf_accs = []
    for j in range(100):
        crosses = []
        used_comps = []
        for i in range(n_components):
            if i not in all_comps:
                new_comps = deepcopy(all_comps)
                new_comps.append(i)
                used_comps.append(i)
                reordered_train = np.concatenate(np.rollaxis(pcad_train[:, :, new_comps], axis=1), axis=1)
                cross_val = cross_validate(svm, reordered_train, y_train, cv=5, return_train_score=True)
                crosses.append(np.mean(cross_val['test_score']))
        max_comp = used_comps[np.argmax(crosses)]
        all_comps.append(max_comp)

        all_accs.append(np.max(crosses))
        reordered_train = np.concatenate(np.rollaxis(pcad_train[:, :, all_comps], axis=1), axis=1)

        reordered_test = np.concatenate(np.rollaxis(pcad_test[:, :, all_comps], axis=1), axis=1)
        count = 0
        svm.fit(reordered_train, y_train)
        for k, m in zip(svm.predict(reordered_test), y_test):
            if k == m:
                count += 1
        test_accs.append(count / len(y_test))

        shuf_test = deepcopy(y_train)
        np.random.shuffle(shuf_test)
        svm.fit(reordered_train, shuf_test)
        count = 0
        for k, m in zip(svm.predict(reordered_test), y_test):
            if k == m:
                count += 1
        shuf_accs.append(count / len(y_test))

    window_train_accs.append(all_accs)
    window_test_accs.append(test_accs)
    window_comps.append(all_comps)
    window_shuf.append(shuf_accs)

np.save(os.path.join(temp_dir, 'one_trial_split_PCA_classifier_%d.npy' % trial_index), [window_train_accs, window_test_accs, window_shuf, window_comps])
