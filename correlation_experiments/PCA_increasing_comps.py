import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sys


#all_accs = list(all_accs)
temp_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/PCA/Correlation_PCA_outputs/Increasing_components/temp'

unit_response = np.load(os.path.join(temp_dir, 'unit_response.npy'))
y_var = np.load(os.path.join(temp_dir, 'y_var.npy'))

window_sizes = [None, 5, 10, 50, 100, 200]

window_index = int(list(sys.argv)[1])

ws = window_sizes[window_index]

if ws is not None:
    # If has a window then run the window response over it
    snipped_response = np.cumsum(unit_response, dtype=float, axis=-1)
    snipped_response[:, :, ws:] = snipped_response[:, :, ws:] - snipped_response[:, :, :-ws]
    snipped_response = snipped_response[:, :, :1 - ws] / ws
else:
    snipped_response = unit_response


sss = StratifiedShuffleSplit(n_splits=1000, test_size=2, random_state=0)
count = 0
w_acc = []
for train_indexes, test_indexes in sss.split(unit_response, y_var):
    X_train = snipped_response[train_indexes]
    X_test = snipped_response[test_indexes]
    # Save the shapes of the data arrays for later
    train_shape = np.array(X_train.shape)
    test_shape = np.array(X_test.shape)
    train_shape[-1] = train_shape[-1] - 400
    test_shape[-1] = test_shape[-1] - 400

    # Remove the prior trial window
    X_train = np.concatenate(X_train)[:, 400:]
    X_test = np.concatenate(X_test)[:, 400:]
    y_train = np.array(y_var)[train_indexes]
    y_test = np.array(y_var)[test_indexes]

    # Find the number of components for the pca
    n_components = X_train.shape[-1]
    pca = PCA(n_components=n_components)

    # Apply the PCA to the test and train data
    pcad_train = pca.fit_transform(X_train)
    pcad_test = pca.transform(X_test)

    # Scale the data
    scaler = StandardScaler()
    pcad_train = scaler.fit_transform(pcad_train)
    pcad_test = scaler.transform(pcad_test)

    # Reshape back into the train and test shapes
    pcad_train = np.reshape(pcad_train, train_shape)
    pcad_test = np.reshape(pcad_test, test_shape)
    svm = LinearSVC(C=1000)
    test_accs = []
    comps_accs = []
    for i in range(1, 100):

        colapsed_train = np.concatenate(np.rollaxis(pcad_train[:, :, :i], axis=1), axis=1)
        svm.fit(colapsed_train, y_train)
        colapsed_test = np.concatenate(np.rollaxis(pcad_test[:, :, :i], axis=1), axis=1)
        comps_accs.append(svm.score(colapsed_test, y_test))
    w_acc.append(comps_accs)
    count += 1
    print(count)
np.save(os.path.join(temp_dir, 'pca_increasing_comps_ws_%s.npy' % str(ws)), w_acc)
