import sys
sys.path.append('/home/camp/warnert/neurolytics')
import threshold_recording as tr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm

recording_index = int(list(sys.argv)[1])
dirs = ['/home/camp/warnert/working/Recordings/Correlation_project_2019/190910/2019-09-10_12-42-15',
        '/home/camp/warnert/working/Recordings/Correlation_project_2019/190911/2019-09-11_15-27-40',
        '/home/camp/warnert/working/Recordings/Correlation_project_2019/190912/2019-09-12_14-55-50',
        '/home/camp/warnert/working/Recordings/Correlation_project_2019/191008/2019-10-08_13-56-53',
        '/home/camp/warnert/working/Recordings/Correlation_project_2019/191009/2019-10-09_14-48-27',
        '/home/camp/warnert/working/Recordings/Correlation_project_2019/191010/2019-10-10_16-00-06']

tc = tr.Threshold_Recording(dirs[recording_index], channel_count=32)

tc.set()

tc.find_unique_spikes()
