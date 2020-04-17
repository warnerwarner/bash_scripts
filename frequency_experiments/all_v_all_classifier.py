import sys
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr
import numpy as np
import psutil
import os
from tqdm import tqdm

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
               '2Hz_D', '5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D']

blanks = ['2Hz_blank', '5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank']
trial1_index = int(list(sys.argv)[1])
trial1 = trial_names[trial1_index]

classifier = cl.Classifier()
classifier.recordings = [rec1, rec2, rec3, rec4, rec5, rec6]

accs = []
for trial2 in trial_names:
    if trial2 != trial1:
        trial1_freq = trial1.split('_')[0]
        trial2_freq = trial2.split('_')[0]
        for i in blanks:
            if trial1_freq in i:
                trial1_blank = i
            if trial2_freq in i:
                trial2_blank = i
        classifier.make_difference_response([trial1, trial2], [trial1_blank, trial2_blank],  baseline=False)
        trial_accs = []
        for i in tqdm(range(repeats)):
            classifier.window_classifier([trial1, trial2], 0.0, 0.5, baseline=False)
            trial_accs.append(classifier.find_accuracy())
        accs.append([trial2, np.mean(trial_accs), np.std(trial_accs)])

np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/%s_vs_each_trial_500ms.npy' % trial1, accs)
