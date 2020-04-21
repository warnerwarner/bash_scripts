import sys
sys.path.append('/home/camp/warnert/neurolytics')
import threshold_recording as tr
import matplotlib.pyplot as plt
import numpy as np


tc = tr.Threshold_Recording("/camp/lab/schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191029/2019-10-29_15-20-16", channel_count=16)

bp_data = tc.set_threshold_crossings(return_bp=True, bp_indiv_chans=True)
tc.set_all_tcs_amplitudes(bp_data=bp_data)
