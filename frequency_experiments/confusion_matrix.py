import sys
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr
import numpy as np
import psutil
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import pickle


available_cpu_count = len(psutil.Process().cpu_affinity())
os.environ["MKL_NUM_THREADS"] = str(available_cpu_count)


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

repeats = 1000

trial_names = ['2Hz_A', '5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A',
               '2Hz_B', '5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B',
               '2Hz_C', '5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C',
               '2Hz_D', '5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D',
               '2Hz_blank', '5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank']

classifier = cl.Classifier()
classifier.recordings = [rec1, rec2, rec3, rec4, rec5, rec6]

classifier.make_unit_response(trial_names, baseline=False)

unit_response = classifier.unit_response

bin_start = int((classifier.pre_trial_window + 0)/classifier.bin_size)
bin_end = int((classifier.pre_trial_window + 0.5)/classifier.bin_size)
num_of_bins = int(len(unit_response[0])/classifier.num_of_units)

window_unit_response = []
for i in range(classifier.num_of_units):
    window_unit_response.append(unit_response[:, bin_start+num_of_bins*i:bin_end+num_of_bins*i])

window_unit_response = np.sum(window_unit_response, axis=2).T

sss = StratifiedShuffleSplit(n_splits=1000, test_size=25)
classifier_guesses = {}
for i, j in tqdm(sss.split(window_unit_response, classifier.y_var)):
    X_train = window_unit_response[j]
    X_test = window_unit_response[i]
    y_train = np.array(classifier.y_var)[j]
    y_test = np.array(classifier.y_var)[i]
    svm = LinearSVC(C=1000)
    svm.fit(X_train, y_train)

    for k, m in zip(y_test, svm.predict(X_test)):
        if k not in classifier_guesses:
            classifier_guesses[k] = {m: 1}
        else:
            if m not in classifier_guesses[k]:
                classifier_guesses[k][m] = 1
            else:
                classifier_guesses[k][m] += 1
pickle.dump(classifier_guesses, open('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/confusion_matrix_outputs.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
