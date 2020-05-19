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
import time
from scipy.fftpack import fft, fftfreq

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


classifier = cl.Classifier()
classifier.recordings = [rec1, rec2, rec3, rec4, rec5, rec6]
classifier.pre_trial_window = 2
classifier.post_trial_window = 2
classifier.make_unit_response(['20Hz_cor_AB', '20Hz_acor_AB', '20Hz_acor_BA'], baseline=False)
classifier.reassign_trial_label('20Hz_acor_BA', '20Hz_acor_AB')
classifier.test_size = 0.2

window_start = int(list(sys.argv)[1])
reps = 1000
window_sizes = [0.01, 0.1, 0.5, 1, 2]

repeats = 1000
all_accs_02 = []
for window_size in window_sizes:
    window_acc = []
    for i in range(repeats):
        classifier.window_classifier(['20Hz_cor_AB', '20Hz_acor_AB'], window_start*0.01, window_start*0.01+window_size, shuffle=True)
        window_acc.append(classifier.find_accuracy())
    all_accs_02.append(window_acc)

classifier.test_size = 1
repeats = 1000
all_accs_1 = []
for window_size in window_sizes:
    window_acc = []
    for i in range(repeats):
        classifier.window_classifier(['20Hz_cor_AB', '20Hz_acor_AB'], window_start*0.01, window_start*0.01+window_size, shuffle=True)
        window_acc.append(classifier.find_accuracy())
    all_accs_1.append(window_acc)
all_accs = [all_accs_02, all_accs_1]

np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/window_classifier_outputs/window_start_%d_noBL_shuf.npy' % window_start, all_accs)
