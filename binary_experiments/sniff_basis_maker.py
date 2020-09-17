import sys
sys.path.append('/home/camp/warnert/neurolytics')
import binary_recording as br
import os

os.chdir('/camp/lab/schaefera/working/warnert/Recordings/binary_pulses')

rec1 = br.Binary_recording('200228/2020-02-28_19-56-29', 32, '200228/2020-02-28trial_names_ventral.txt', sniff_basis=True)
rec2 = br.Binary_recording('200228/2020-02-28_16-37-36/', 32, '200228/2020-02-28trial_names_dorsal.txt', sniff_basis=True)
rec3 = br.Binary_recording('200303/2020-03-03_16-44-23/', 32, '200303/2020-03-03trial_names_dorsal.txt', sniff_basis=True)
rec4 = br.Binary_recording('200303/2020-03-03_19-57-03/', 32, '200303/2020-03-03trial_names_ventral.txt', sniff_basis=True)
rec5 = br.Binary_recording('200309/2020-03-09_16-20-42/', 32, '200309/2020-03-09trial_name_joined.txt', sniff_basis=True)
rec6 = br.Binary_recording('200311/2020-03-11_16-51-10/', 32, '200311/2020-03-11trial_name_binary_joined.txt', sniff_basis=True)
rec7 = br.Binary_recording('200318/2020-03-18_15-24-43/', 32, '200318/2020-03-18trial_name.txt', sniff_basis=True)
rec8 = br.Binary_recording('200319/2020-03-19_16-08-45/', 32, '200319/2020-03-19_16-08-45_trial_names.txt', sniff_basis=True)
rec9 = br.Binary_recording('200320/2020-03-20_17-30-11/', 32, '200320/2020-03-20trial_name.txt', sniff_basis=True)
rec10 = br.Binary_recording("200625/2020-06-25_15-29-40/", 32, '200625/2020-06-25trial_name.txt', sniff_basis=True)
rec11 = br.Binary_recording('200626/2020-06-26_15-57-35/', 32, '200626/2020-06-26trial_name.txt', sniff_basis=True)
rec12 = br.Binary_recording('200708/2020-07-08_18-22-44/', 32, '200708/2020-07-08trial_name.txt', sniff_basis=True)

