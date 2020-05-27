import numpy as np
import sys
from string import Template
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import subprocess
import time
from copy import deepcopy



# Set the paths
out_dir = '/home/camp/warnert/working/Recordings/Correlation_project_2019/Correlation_PCA_outputs/Increasing_components'
trialbank_loc = '/home/camp/warnert/working/Recordings/trialbanks/190910SqPulseFreqCorrelationLongRandom.trialbank'
bash_temp_dir = '/home/camp/warnert/bash_scripts/correlation_experiments/PCA_classifier_ordered_template.sh'

# Make a temporary folder to hold files that can be deleted later
temp_dir = os.path.join(out_dir, 'temp')
if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir)


# Load in the experimental files
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


# Receive the inputs for the components to be used
comp_index = list(sys.argv)[1:]
comp_index = [int(i) for i in comp_index]

# Create a classifier instance
test_class = cl.Classifier()
test_class.recordings = recordings
test_class.pre_trial_window = 2
test_class.post_trial_window = 2
test_class.test_size = 1
window_sizes = [None, 5, 10, 50, 100, 200]

ws = window_sizes[0]
# Set the number of components for the classifier, by default this trace will have 399 time points (4s * 100 bins/s - 1)
n_components = 399
# Make a PCA'd response if one doesn't exist
if os.path.isfile(os.path.join(temp_dir, 'PCA_%s.npy' % str(ws))):
    pcad_response = np.load(os.path.join(temp_dir, 'PCA_%s.npy' % str(ws)))
    y_var = np.load(os.path.join(temp_dir, 'y_var.npy'))
else:
    test_class.make_unit_response(['20Hz_cor_AB', '20Hz_acor_AB', '20Hz_acor_BA'], baseline=True)
    test_class.reassign_trial_label('20Hz_acor_BA', '20Hz_acor_AB')
    pcad_response, y_var = test_class.make_pcad_response(n_components, ['20Hz_cor_AB', '20Hz_acor_AB'],
                                                         window_size=ws, baseline=True, trace_start=0)
    np.save(os.path.join(temp_dir, 'PCA_%s.npy' % str(ws)), pcad_response)
    np.save(os.path.join(temp_dir, 'y_var.npy'), y_var)

# Run through and repeat n classifications
accs = []
for i in range(1000):
    test_class.pca_classifier(pcad_response[:, :, comp_index], y_var)
    accs.append(test_class.find_accuracy())

# Find the accuracy and the std
accuracy = np.mean(accs)
std = np.std(accs)

# Write the accuracy of this component out to a txt file
out_txt = open(os.path.join(out_dir, 'accuracy_component_%d_%s.txt' % (len(comp_index), str(ws))), 'a')
out_txt.write('%d-%f-%f\n' % (comp_index[-1], accuracy, std))
out_txt.close()

# Read the text file that was just written
out_txt = open(os.path.join(out_dir, 'accuracy_component_%d_%s.txt' % (len(comp_index), str(ws))), 'r')
out_lines = out_txt.readlines()
out_txt.close()

# If the text file is of a length such that all components have been ran through then:
if ws is not None:
    req_length = n_components + 1 - len(comp_index) - ws
else:
    req_length = n_components + 1 - len(comp_index)
if len(out_lines) == req_length:

    # Read in the components and their accuracies
    all_comps = [int(i.split('-')[0]) for i in out_lines]
    all_accs = [float(i.split('-')[1]) for i in out_lines]

    # Find the maximum accuracy component
    max_comp_index = np.argmax(all_accs)
    max_comp = all_comps[max_comp_index]

    # Remove the last part of the new_comp_index, which is the component tested earlier in this script
    # Add in the highest found component
    new_comp_index = comp_index[:-1]
    new_comp_index.append(max_comp)

    # If not all components have been ordered then:
    if len(comp_index) != n_components:

        # Make a copy of the new ordered component list
        temp_comps = deepcopy(new_comp_index)

        # Convert it into a string to pass
        temp_comps_str = ''
        for j in temp_comps:
            temp_comps_str += str(j) + ' '
        temp_comps_str = temp_comps_str[:-1]  # Remove the last space

        # Create a new string to put in the array component
        array_str = ''
        for i in range(n_components):
            if i not in temp_comps:
                array_str += str(i) + ','
        array_str = array_str[:-1]  # Remove the final comma

        # Open up the bash template and replace place holders with values
        bash_temp = open(bash_temp_dir, 'r')
        src = Template(bash_temp.read())

        # Need to put the SLURM_TASK_ARRAY back in because python gets angry that there is another string with a $
        temp_bash = src.substitute({'pca_components': temp_comps_str, 'new_components': array_str, 'SLURM_ARRAY_TASK_ID': '$SLURM_ARRAY_TASK_ID'})

        # Create a new bash file
        temp_bash_dir = os.path.join(temp_dir, 'PCA_temp_bash_%d_%s.sh' % (len(comp_index), str(ws)))
        temp_bash_out = open(temp_bash_dir, 'w')
        temp_bash_out.write(temp_bash)
        temp_bash_out.close()
        bash_temp.close()

        # Run the new bash command
        bash_command = 'sbatch %s' % temp_bash_dir
        process = subprocess.Popen(bash_command.split())
