import sys
sys.path.append('/camp/home/warnert/neurolytics')
import binary_recording as br
import matplotlib.pyplot as plt
import numpy as np


base_dir = '/camp/home/warnert/working/Recordings/binary_pulses/200228/2020-02-28_16-37-36'
rec = br.Binary_recording(base_dir, 32, base_dir+'/2020-02-28trial_name_dorsal.txt')

rec.set()
num_to_chem = ['IA', 'EB', 'EA', 'ET']
for i in rec.get_good_clusters():
    maxes = []
    cluster_num = i.cluster_num
    ys = []
    for i in range(4):
        trial_name = '31_%d' % int(2*i +1)
        trux, y = rec.get_binned_trial_response(trial_name, cluster_num, baselined=True, pre_trial_window=1, post_trial_window=1)
        maxes.append(max(np.max(y), abs(np.min(y))))
        ys.append(y)
    full_max = max(maxes)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        print(i%2, int(i/2))
        im = ax[int(i/2), i%2].imshow(ys[i], aspect='auto', vmin=-full_max, vmax=full_max, origin='lower', cmap='bwr')
        ax[int(i/2), i%2].set_title(num_to_chem[i])
        ax[int(i/2), i%2].set_xticks(np.arange(0, 240, 20))
        ax[int(i/2), i%2].set_xticklabels(np.round(np.arange(-1, 1.4, 1/5), 3))
        ax[int(i/2), i%2].axvspan(100, 120, color='gray', alpha=0.2)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (s)")
    plt.ylabel("Trials")
    plt.savefig(base_dir + '/unit_resp_heatmap/cluster_%d_full_odour_resp.png', dpi=300)
    plt.close()

for i in rec.get_good_clusters():
    maxes = []
    cluster_num = i.cluster_num
    ys = []
    for i in range(4):
        trial_name = '0_%d' % int(2*i +1)
        trux, y = rec.get_binned_trial_response(trial_name, cluster_num, baselined=True, pre_trial_window=1, post_trial_window=1)
        maxes.append(max(np.max(y), abs(np.min(y))))
        ys.append(y)
    full_max = max(maxes)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        print(i%2, int(i/2))
        im = ax[int(i/2), i%2].imshow(ys[i], aspect='auto', vmin=-full_max, vmax=full_max, origin='lower', cmap='bwr')
        ax[int(i/2), i%2].set_title(num_to_chem[i])
        ax[int(i/2), i%2].set_xticks(np.arange(0, 240, 20))
        ax[int(i/2), i%2].set_xticklabels(np.round(np.arange(-1, 1.4, 1/5), 3))
        ax[int(i/2), i%2].axvspan(100, 120, color='gray', alpha=0.2)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (s)")
    plt.ylabel("Trials")
    plt.savefig(base_dir + '/unit_resp_heatmap/cluster_%d_blank_resp.png', dpi=300)
    plt.close()
