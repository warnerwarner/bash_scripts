'''
Unfinished
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

out_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/Correlation_PCA_outputs/Increasing_components'
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

recordings = [rec1, rec2, rec3, rec4, rec5, rec6]

n_components_sub = int(list(sys.argv)[1])

window_sizes = [None, 5, 10, 50, 100, 200]


test_class = cl.Classifier()
test_class.recordings = recordings
test_class.pre_trial_window = 2
test_class.post_trial_window = 2
test_class.test_size = 1
n_components = 399



all_accs = []
baselined=False
test_class.make_unit_response(['20Hz_cor_AB', '20Hz_acor_AB', '20Hz_acor_BA'], baselined=baselined)
pcad_response, y_var = test_class.make_pcad_response(n_components, ['20Hz_cor_AB', '20Hz_acor_AB', '20Hz_acor_BA'],
                                                         reassign_y_var=[['20Hz_acor_AB', '20Hz_acor_BA']], window_size=window_size, baseline=baselined)

reduced_unit_response = [i[:, :, 200:] for i in test_class.unit_response]
test_class.unit_response = reduced_unit_response


for pcad_index in range(6):
    window_size = window_sizes[pcad_index]
    n_components = pcad_response.shape[-1]
    index_accuracy = []
    for i in range(1000):
        test_class.pca_classifier(pcad_response[:, :, :n_components_sub], y_var)
        index_accuracy.append(test_class.find_accuracy())
    all_accs.append(index_accuracy)



np.save(os.path.join(out_dir, '20Hz_4s_n_comps_%d_all_ws_baselined.npy') % n_components_sub, all_accs)
