import sys
sys.path.append('/home/camp/warnert/neurolytics/')
import binary_recording as br
import joined_recording as jr
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import psutil


def SCA(data, *, n_components='full'):
    scaler = StandardScaler()
    pcs = []
    scaled_data = scaler.fit_transform(data.T).T
    for j in tqdm(range(len(scaled_data)), leave=False):
        pca = PCA(n_components=1)
        comp1 = pca.fit_transform(scaled_data)
        comp1_project = pca.inverse_transform(comp1)
        scaled_data = scaled_data - comp1_project
        pcs.append(pca.components_)
        scaled_data = scaler.fit_transform(scaled_data.T).T
    return pcs, scaled_data

available_cpu_count = len(psutil.Process().cpu_affinity())
os.environ["MKL_NUM_THREADS"] = str(available_cpu_count)

home_dir = '/home/camp/warnert/working/Recordings/binary_pulses'

rec1 = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_19-56-29/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_ventral.txt'), sniff_basis=True)
rec2 = br.Binary_recording(os.path.join(home_dir, '200228/2020-02-28_16-37-36/'), 32, os.path.join(home_dir, '200228/2020-02-28trial_names_dorsal.txt'), sniff_basis=True)
rec3 = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_16-44-23/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_dorsal.txt'), sniff_basis=True)
rec4 = br.Binary_recording(os.path.join(home_dir, '200303/2020-03-03_19-57-03/'), 32, os.path.join(home_dir, '200303/2020-03-03trial_names_ventral.txt'), sniff_basis=True)
rec5 = br.Binary_recording(os.path.join(home_dir, '200309/2020-03-09_16-20-42/'), 32, os.path.join(home_dir, '200309/2020-03-09trial_name_joined.txt'), sniff_basis=True)
rec6 = br.Binary_recording(os.path.join(home_dir, '200311/2020-03-11_16-51-10/'), 32, os.path.join(home_dir, '200311/2020-03-11trial_name_binary_joined.txt'), sniff_basis=True)
rec7 = br.Binary_recording(os.path.join(home_dir, '200318/2020-03-18_15-24-43/'), 32, os.path.join(home_dir, '200318/2020-03-18trial_name.txt'), sniff_basis=True)
rec8 = br.Binary_recording(os.path.join(home_dir, '200319/2020-03-19_16-08-45/'), 32, os.path.join(home_dir, '200319/2020-03-19_16-08-45_trial_names.txt'), sniff_basis=True)

recs = jr.JoinedRecording(recordings=[rec1, rec2, rec3, rec4, rec5, rec6, rec7, rec8])

trial_names = ['%d_1' % i for i in range(32)]
resps = recs.get_multi_trial_response(trial_names, pre_trial_window=2, post_trial_window=2, w_bootstrap=True, bootstrap_limit=1000, bin_size=0.05)
resps = np.array(resps)
scaler = StandardScaler()
reshaped_resps = np.rollaxis(resps, axis=2).reshape(145, 32*1000*80)

pcs, scaled_data = SCA(reshaped_resps)
np.save(os.path.join(home_dir, 'SCA/200909_pcs.npy'), pcs)
np.save(os.path.join(home_dir, 'SCA/200909_data.npy'), scaled_data)
