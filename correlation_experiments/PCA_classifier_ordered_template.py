import numpy as np
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

temp_dir = os.path.join(out_dir, 'temp')
recordings = [rec1, rec2, rec3, rec4, rec5, rec6]

comp_index = list(sys.argv)[1:]
comp_index = [int(i) for i in comp_index]


test_class = cl.Classifier()
test_class.recordings = recordings
test_class.pre_trial_window = 2
test_class.post_trial_window = 2

# test_class.make_unit_response(['20Hz_cor_AB', '20Hz_acor_BA', '20Hz_acor_AB'], baseline=baselined)
test_class.test_size = 0.2
n_components = 599
if os.path.isfile(os.path.join(temp_dir, 'PCA.npy')):
    pcad_response = np.load(os.path.join(temp_dir, 'PCA.npy'))
    y_var = np.load(os.path.join(temp_dir, 'y_var.npy'))
else:
    pcad_response, y_var = test_class.make_pcad_response(n_components, ['20Hz_cor_AB', '20Hz_acor_AB', '20Hz_acor_BA'],
                                                         reassign_y_var=[['20Hz_acor_AB', '20Hz_acor_BA']], window_size=None, baselined=True)
    np.save(os.path.join(temp_dir, 'PCA.npy'), pcad_response)
    np.save(os.path.join(temp_dir, 'y_var.npy'), y_var)


accs = []
for i in tqdm(range(100)):
    test_class.pca_classifier(pcad_response[:, :, comp_index], y_var)
    accs.append(test_class.find_accuracy())
accuracy = np.mean(accs)
out_txt = open(os.path.join(out_dir, 'accuracy_component_%d.txt' % len(comp_index)), 'a')
out_txt.writelines('%d-%f\n' % (comp_index[-1], accuracy))
out_txt.close()
out_txt = open(os.path.join(out_dir, 'accuracy_component_%d.txt' % len(comp_index)), 'r')

if len(out_txt.readlines() == n_components):
    outputs = out_txt.readlines()
    all_comps = [int(i.split('-')[0]) for i in outputs]
    all_accs = [float(i.split('-')[1][:-1]) for i in outputs]
    max_comp_index = np.argmax(all_accs)
    max_comp = all_comps[max_comp_index]
    if max_comp in comp_index[:-1]:
        del all_accs[max_comp_index]
        del all_comps[max_comp_index]
        max_comp_index = np.argmax(all_accs)
        max_comp = all_comps[max_comp_index]
    new_comp_index = comp_index[:-1].append(max_comp)



if baselined:
    np.save(os.path.join(out_dir, '20Hz_ordered_accuracy_ws_%s_baselined.npy') % str(window_size), all_accs)
    np.save(os.path.join(out_dir, '20Hz_ordered_components_ws_%s_baselined.npy') % str(window_size), max_comps)
else:
    np.save(os.path.join(out_dir, '20Hz_ordered_accuracy_ws_%s.npy') % str(window_size), all_accs)
    np.save(os.path.join(out_dir, '20Hz_ordered_ws_components.npy') % str(window_size), max_comps)
