import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sys


temp_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/PCA/Correlation_PCA_outputs/Increasing_components/temp'

unit_response = np.load(os.path.join(temp_dir, 'unit_response.npy'))
y_var = np.load(os.path.join(temp_dir, 'y_var.npy'))


window_sizes = [None, 5, 10, 50, 100, 200, 'shuf']

ws_components = {None: 19, 5: 19, 10: 15, 50: 22, 100: 10, 200: 17, 'shuf': 19}

window_index = int(list(sys.argv)[1])
ws = window_sizes[window_index]

all_units_accs = []
if ws is None:
    snipped_response = unit_response
elif ws == 'shuf':
    snipped_response = unit_response
else:
    snipped_response = np.cumsum(unit_response, dtype=float, axis=-1)
    snipped_response[:, :, ws:] = snipped_response[:, :, ws:] - snipped_response[:, :, :-ws]
    snipped_response = snipped_response[:, :, :1 - ws] / ws

sss = StratifiedShuffleSplit(n_splits=1000, test_size=2, random_state=0)
w_acc = []
for train_indexes, test_indexes in sss.split(unit_response, y_var):
    X_train = snipped_response[train_indexes]
    X_test = snipped_response[test_indexes]

    train_shape = np.array(X_train.shape)
    test_shape = np.array(X_test.shape)

    train_shape[-1] = ws_components[ws]
    test_shape[-1] = ws_components[ws]

    X_train = np.concatenate(X_train)[:, 400:]
    X_test = np.concatenate(X_test)[:, 400:]

    y_train = np.array(y_var)[train_indexes]
    y_test = np.array(y_var)[test_indexes]

    pca = PCA(n_components=ws_components[ws])
    pcad_train = pca.fit_transform(X_train)
    pcad_test = pca.transform(X_test)
    scaler = StandardScaler()

    pcad_train = scaler.fit_transform(pcad_train)
    pcad_test = scaler.transform(pcad_test)

    pcad_train = np.reshape(pcad_train, train_shape)
    pcad_test = np.reshape(pcad_test, test_shape)

    svm = LinearSVC(C=1000)

    unit_indexes = np.arange(97)
    if ws == "shuf":
        np.random.shuffle(y_train)
    np.random.shuffle(unit_indexes)
    units_accs = []
    for i in range(1, 98):
        collapsed_train = np.concatenate(np.rollaxis(pcad_train[:, unit_indexes[:i], :], axis=1), axis=1)
        svm.fit(collapsed_train, y_train)
        collapsed_test = np.concatenate(np.rollaxis(pcad_test[:, unit_indexes[:i], :], axis=1), axis=1)
        units_accs.append(svm.score(collapsed_test, y_test))
    w_acc.append(units_accs)
all_units_accs.append(w_acc)

np.save(os.path.join(temp_dir, 'pca_increasing_units_2_ws_%s.npy' % str(ws)), all_units_accs)
