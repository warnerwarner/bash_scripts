import sys
sys.path.append('/home/camp/warnert/neurolytics')
import correlation_recording as cr
import classifier as cl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import classifier as cl
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from copy import deepcopy
import time
from tqdm import tqdm
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from sklearn.datasets import make_classification

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
classifier.make_unit_response(['2Hz_A', '5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'])
classifier.make_difference_response(['2Hz_A', '5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'],
                                    ['2Hz_blank', '5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'])

classifier.test_size = 0.2

scale_types = [None, 'standard', 'robust', 'minmax']
scale_index = list(sys.argv)[1]
scale_type = scale_types[int(scale_index)]
classifier.scale = scale_type


accs = []
for i in tqdm(range(400)):
    step_acc = []
    step_acc_shuf = []
    for j in range(100):
        classifier.window_classifier(['2Hz_A', '5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'], 0, 0.01+i*0.01, baseline=False)
        classifier.find_accuracy()
        step_acc.append(classifier.find_accuracy())
        classifier.window_classifier(['2Hz_A', '5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'], 0, 0.01+i*0.01, baseline=False, shuffle=True)
        classifier.find_accuracy()
        step_acc_shuf.append(classifier.find_accuracy())
    accs.append([np.mean(step_acc), np.mean(step_acc_shuf), np.std(step_acc), np.std(step_acc_shuf)])
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/window_228_accuracy_num_of_units_%s.npy' % scale_type, accs)
