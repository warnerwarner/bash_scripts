import sys
sys.path.append('/home/camp/warnert/neurolytics')
import classifier as cl
import correlation_recording as cr
import numpy as np
import psutil
import os


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


repeats = 10

window_size = int(list(sys.argv)[1])
classifier = cl.Classifier()
classifier.recordings = [rec1, rec2, rec3, rec4, rec5, rec6]
classifier.test_size = 0.2

# A
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'], 0.0, window_size*0.01, baseline=False)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/A_window_size_%f.npy' % window_size, accuracy)

# B
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B'], 0.0, window_size*0.01, baseline=False)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/B_window_size_%f.npy' % window_size, accuracy)

# C
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C'], 0.0, window_size*0.01, baseline=False)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/C_window_size_%f.npy' % window_size, accuracy)

# D
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D'], 0.0, window_size*0.01, baseline=False)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/D_window_size_%f.npy' % window_size, accuracy)

# Blank
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'], 0.0, window_size*0.01, baseline=False)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/Blank_window_size_%f.npy' % window_size, accuracy)


# shuffle

# A
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'], 0.0, window_size*0.01, baseline=False, shuffle=True)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/A_window_size_shuf_%f.npy' % window_size, accuracy)

# B
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B'], 0.0, window_size*0.01, baseline=False, shuffle=True)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/B_window_size_shuf_%f.npy' % window_size, accuracy)


# C
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C'], 0.0, window_size*0.01, baseline=False, shuffle=True)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/C_window_size_shuf_%f.npy' % window_size, accuracy)

# D
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D'], 0.0, window_size*0.01, baseline=False, shuffle=True)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/D_window_size_shuf_%f.npy' % window_size, accuracy)

# Blank
accuracy = []
for start in range(600-window_size):
    start_accuracy = []
    for i in range(repeats):
        classifier.window_classifier(['5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'], 0.0, window_size*0.01, baseline=False, shuffle=True)
        classifier.find_accuracy()
        start_accuracy.append(classifier.accuracy)

    accuracy.append([start, np.mean(start_accuracy), np.std(start_accuracy)])
accuracy = np.array(accuracy)
np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/Blank_window_size_shuf_%f.npy' % window_size, accuracy)
