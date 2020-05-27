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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm
from copy import deepcopy
import time
import os
from scipy.fftpack import fft, fftfreq

temp_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/PCA/Correlation_PCA_outputs/Increasing_components/temp'
trialbank_loc = '/home/camp/warnert/working/Recordings/trialbanks/190910SqPulseFreqCorrelationLongRandom.trialbank'
rec1 = cr.Correlation_Recording("/home/camp/warnert/working/Recordings/Correlation_project_2019/190910/2019-09-10_12-42-15", 32, trialbank_loc)


rec2 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/190911/2019-09-11_15-27-40', 32, trialbank_loc)

rec3 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/2019-09-12_14-55-50', 32, trialbank_loc)

rec4 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/191008/2019-10-08_13-56-53', 32, trialbank_loc)
rec5 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/191009/2019-10-09_14-48-27', 32, trialbank_loc)
rec6 = cr.Correlation_Recording("/home/camp/warnert/working/Recordings/Correlation_project_2019/191010/2019-10-10_16-00-06", 32, trialbank_loc)


rec1.set()
rec2.set()
rec3.set()
rec4.set()
rec5.set()
rec6.set()


trial_index = int(list(sys.argv)[1])

recordings = [rec1, rec2, rec3, rec4, rec5, rec6]

test_class = cl.Classifier()
test_class.recordings = recordings
test_class.post_trial_window = 2
test_class.pre_trial_window = 4
if os.path.isfile(os.path.join(temp_dir, 'unit_response.npy')):
    unit_response = np.load(os.path.join(temp_dir, 'unit_response.npy'))
    y_var = np.load(os.path.join(temp_dir, 'y_var.npy'))
else:
    test_class.make_unit_response(['20Hz_cor_AB', '20Hz_acor_AB', '20Hz_acor_BA'], baseline=True)
    test_class.reassign_trial_label('20Hz_acor_BA', '20Hz_acor_AB')

    unit_response = np.concatenate(test_class.unit_response)
    y_var = test_class.y_var
    np.save(os.path.join(temp_dir, 'unit_response.npy'), unit_response)
    np.save(os.path.join(temp_dir, 'y_var.npy'), y_var)





sss = StratifiedShuffleSplit(n_splits=1000, test_size=26, random_state=0)

split_indexes = sss.split(np.sum(unit_response[:, :, :50], axis=-1), y_var)

split_indexes = list(split_indexes)

train_indexes, test_indexes = split_indexes[trial_index]
window_sizes = [None, 5, 10, 50, 100, 200]
X_trains = []
X_tests = []

y_train = np.array(y_var)[train_indexes]
y_test = np.array(y_var)[test_indexes]



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

    X_train = np.concatenate(X_train)[:, 400:]
    X_test = np.concatenate(X_test)[:, 400:]
    X_trains.append(X_train)
    X_tests.append(X_test)




window_accs = []
window_comps = []
for X_train, X_test in zip(X_trains, X_tests):
    n_components = X_train.shape[-1]
    pca = PCA(n_components=n_components)
    pcad_train = pca.fit_transform(X_train)
    pcad_test = pca.transform(X_test)

    reordered_train = []
    train_num = unit_response.shape[0] - sss.test_size
    for i in range(train_num):
        reordered_train.append(pcad_train[i*97:(i+1)*97])
    reordered_train = np.array(reordered_train)

    reordered_test = []
    test_num = sss.test_size

    for i in range(test_num):
        reordered_test.append(pcad_test[i*97:(i+1)*97])
    reordered_test = np.array(reordered_test)

    svm = LinearSVC(C=1000)

    accs_test = []
    max_comps = []
    for j in range(100):
        rank_acc = []
        all_comps = []
        for i in range(n_components):
            if i not in max_comps:
                new_comps = deepcopy(max_comps)
                new_comps.append(i)
                scaler = StandardScaler()

                cut_train = scaler.fit_transform(np.concatenate(reordered_train[:, :, new_comps], axis=-1).T)
                cut_test = scaler.transform(np.concatenate(reordered_test[:, :, new_comps], axis=-1).T)
                svm.fit(cut_train, y_train)
                count = 0
                for k, m in zip(svm.predict(cut_test), y_test):
                    if k == m:
                        count += 1
                rank_acc.append(count/len(y_test))
                all_comps.append(i)
        max_comp = all_comps[np.argmax(rank_acc)]
        max_comps.append(max_comp)
        accs_test.append(np.max(rank_acc))
    window_accs.append(accs_test)
    window_comps.append(max_comps)
np.save(os.path.join(temp_dir, '%d_components.npy' % trial_index), [window_accs, window_comps])
