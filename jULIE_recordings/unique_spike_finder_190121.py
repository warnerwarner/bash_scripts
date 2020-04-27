#setting for /home/camp/warnert/working/Recordings/2019_general/190121/2019-01-21_18-10-30

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

dir = "/home/camp/warnert/working/Recordings/2019_general/190121/2019-01-21_18-10-30"
chan_counts = [18, 18, 18, 18, 18, 32]

tc = tr.Threshold_Recording(dir, channel_count=18, dat_name='dat_for_jULIE_analysis.dat')

tc.set()

slice_size = tc.rec_length()/total_slices
start_time = spike_slice*slice_size
end_time = start_time+slice_size

tc.find_unique_spikes(start_time=start_time, end_time=end_time)
