import sys
sys.path.append('/home/camp/warnert/neurolytics')
import binary_recording as br

home_dir = 'home/camp/warnert/working/Recordings/200309'

rec = br.Binary_recording(home_dir+'2020-03-09_16-20-42', 32, home_dir+'2020-03-09trial_name_joined.txt')

rec.set()
rec.set_threshold_crossings()
