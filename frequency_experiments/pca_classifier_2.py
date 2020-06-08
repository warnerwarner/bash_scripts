import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import sys
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr

out_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/pca_outputs'
responses = ['A_unit_resp.npy', 'B_unit_resp.npy', 'C_unit_resp.npy', 'D_unit_resp.npy']
trial_names = ['2Hz', '5Hz', '10Hz', '15Hz', '20Hz']
odours = ['A', 'B', 'C', 'D']
y_var = np.load(os.path.join(out_dir, 'y_var.npy'))
all_scores = []
for i, j in zip(responses, odours):
    if not os.path.isfile(os.path.join(out_dir, i)):
        odour_trial_names = ['_'.join([k, j]) for k in trial_names]
        rec1 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/190910/2019-09-10_12-42-15', 32, '/home/camp/warnert/working/Recordings/Correlation_project_2019/190910/190910SqPulseFreqCorrelationLongRandom.trialbank')
        rec2 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/190911/2019-09-11_15-27-40/', 32, '/home/camp/warnert/working/Recordings/Correlation_project_2019/190911/190910SqPulseFreqCorrelationLongRandom.trialbank')
        rec3 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/2019-09-12_14-55-50/', 32, '/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/190910SqPulseFreqCorrelationLongRandom.trialbank')
        rec4 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/191008/2019-10-08_13-56-53', 32, '/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/190910SqPulseFreqCorrelationLongRandom.trialbank')
        rec5 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/191009/2019-10-09_14-48-27', 32, '/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/190910SqPulseFreqCorrelationLongRandom.trialbank')
        rec6 = cr.Correlation_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/191010/2019-10-10_16-00-06', 32, '/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/190910SqPulseFreqCorrelationLongRandom.trialbank')
        rec1.set()
        rec2.set()
        rec3.set()
        rec4.set()
        rec5.set()
        rec6.set()
        classifier = cl.Classifier()
        classifier.recordings = [rec1, rec2, rec3, rec4, rec5, rec6]
        classifier.pre_trial_window = 2
        classifier.post_trial_window = 2
        classifier.make_unit_response(odour_trial_names, baseline=True)
        np.save(os.path.join(out_dir, i), classifier.unit_response)
        unit_resp = classifier.unit_response
    else:
        unit_resp = np.load(os.path.join(out_dir, i))


    conced_resp = np.concatenate(np.concatenate(unit_resp, axis=0))[:, 200:]
    train_shape = np.concatenate(unit_resp, axis=0)[:, :, 200:].shape
    pca = PCA(n_components=399)
    pcad = pca.fit_transform(conced_resp)
    pcad = np.reshape(pcad, train_shape)
    conced_pcad = np.array([np.concatenate(i) for i in pcad])

    strat_index = int(list(sys.argv)[1])

    sss = StratifiedShuffleSplit(n_splits=1000, test_size=5, random_state=0)
    strats = list(sss.split(np.concatenate(unit_resp), y_var))
    train_index, test_index = strats[strat_index]

    pcad_train = pcad[train_index]
    pcad_test = pcad[test_index]
    y_train = y_var[train_index]
    y_test = y_var[test_index]

    scaler = StandardScaler()
    svm = LinearSVC(C=1000)
    comp_score = []
    for i in range(1, 200):
        colapsed_train = np.array([np.concatenate(i) for i in pcad_train[:, :, :i]])
        colapsed_test = np.array([np.concatenate(i) for i in pcad_test[:, :, :i]])
        #np.random.shuffle(y_train)
        X_train = scaler.fit_transform(colapsed_train)
        X_test = scaler.transform(colapsed_test)
        svm.fit(X_train, y_train)
        comp_score.append(svm.score(X_test, y_test))
    all_scores.append(comp_score)

np.save(os.path.join(out_dir, '%d_strat_index_all_odours_1_test.npy' % strat_index), all_scores)
