import sys
sys.path.append('/home/camp/warnert/neurolytics')
import correlation_recording as cr
import numpy as np
import classifier as cl

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

accuracy = []
n_components = int(list(sys.argv)[1])
classifier = cl.Classifier()
classifier.recordings = [rec1, rec2, rec3, rec4, rec5, rec6]
classifier.test_size = 0.2
classifier.make_pca_response(n_components, ['5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'])


pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'], baseline=False)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['A', np.mean(pca_accuracy), np.std(pca_accuracy)])

classifier.make_pca_response(n_components, ['5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B'])
pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B'], baseline=False)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['B', np.mean(pca_accuracy), np.std(pca_accuracy)])

classifier.make_pca_response(n_components, ['5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C'])

pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C'], baseline=False)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['C', np.mean(pca_accuracy), np.std(pca_accuracy)])


classifier.make_pca_response(n_components, ['5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D'])
pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D'], baseline=False)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['D', np.mean(pca_accuracy), np.std(pca_accuracy)])

classifier.make_pca_response(n_components, ['5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'])
pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'], baseline=False)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['Blank', np.mean(pca_accuracy), np.std(pca_accuracy)])



# Shuffle

classifier.make_pca_response(n_components, ['5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'])


pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_A', '10Hz_A', '15Hz_A', '20Hz_A'], baseline=False, shuffle=True)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['A_shuf', np.mean(pca_accuracy), np.std(pca_accuracy)])

classifier.make_pca_response(n_components, ['5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B'])
pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_B', '10Hz_B', '15Hz_B', '20Hz_B'], baseline=False, shuffle=True)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['B_shuf', np.mean(pca_accuracy), np.std(pca_accuracy)])

classifier.make_pca_response(n_components, ['5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C'])

pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_C', '10Hz_C', '15Hz_C', '20Hz_C'], baseline=False, shuffle=True)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['C_shuf', np.mean(pca_accuracy), np.std(pca_accuracy)])


classifier.make_pca_response(n_components, ['5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D'])
pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_D', '10Hz_D', '15Hz_D', '20Hz_D'], baseline=False, shuffle=True)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['D_shuf', np.mean(pca_accuracy), np.std(pca_accuracy)])

classifier.make_pca_response(n_components, ['5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'])
pca_accuracy = []
for i in range(repeats):
    classifier.pca_classifier(n_components, ['5Hz_Blank', '10Hz_Blank', '15Hz_Blank', '20Hz_Blank'], baseline=False, shuffle=True)
    classifier.find_accuracy()
    pca_accuracy.append(classifier.accuracy)

accuracy.append(['Blank_shuf', np.mean(pca_accuracy), np.std(pca_accuracy)])

np.save('/home/camp/warnert/working/Recordings/Correlation_project_2019/frequency/window_classifier_accuracy/pca_%d_components.npy' % n_components, accuracy)
