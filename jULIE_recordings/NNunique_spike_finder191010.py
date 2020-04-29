# Unique spike finding for '/home/camp/warnert/working/Recordings/Correlation_project_2019/191010/2019-10-10_16-00-06'

import sys
sys.path.append('/home/camp/warnert/neurolytics')
import threshold_recording as tr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm




spike_slice = int(list(sys.argv)[1])
total_slices = int(list(sys.argv)[2])

start_time = spike_slice

tc = tr.Threshold_Recording('/home/camp/warnert/working/Recordings/Correlation_project_2019/191010/2019-10-10_16-00-06', channel_count=32)

tc.set()

slice_size = tc.rec_length/total_slices
start_time = spike_slice*slice_size
end_time = start_time+slice_size

tc.find_unique_spikes(start_time=start_time, end_time=end_time)
