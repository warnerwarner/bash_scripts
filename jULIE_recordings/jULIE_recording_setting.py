import sys
sys.path.append('/home/camp/warnert/neurolytics')
import threshold_recording as tr
import matplotlib.pyplot as plt
import numpy as np


dir_index = int(list(sys.argv)[1])
dirs = ["/home/camp/warnert/working/Recordings/2018_general/181205/2018-12-05_18-17-18",
        '/home/camp/warnert/working/Recordings/2019_general/190121/2019-01-21_18-10-30',
        "/home/camp/warnert/working/Recordings/2019_general/190207/2019-02-07_18-31-33",
        "/home/camp/warnert/working/Recordings/2019_general/190211/2019-02-11_16-35-46",
        "/home/camp/warnert/working/Recordings/2019_general/190704/2019-07-04_15-21-04",
        "/home/camp/warnert/working/Recordings/2019_general/190801/2019-08-01_16-38-19"]
chan_counts = [18, 18, 18, 18, 18, 32]

tc = tr.Threshold_Recording(dirs[dir_index], channel_count=chan_counts[dir_index])

bp_data = tc.set_threshold_crossings(return_bp=True, bp_indiv_chans=True)
tc.set_all_tcs_amplitudes(bp_data=bp_data)
