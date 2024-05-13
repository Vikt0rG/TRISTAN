import struct
import os
from tqdm import tqdm

import numpy as np
from scipy.signal import lfilter

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
from matplotlib import cm

from readFile import readFile

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# =======================================================================================================
# Get data from directories
# =======================================================================================================

baseDir = "./data/run_1"
files = []
for filename in os.listdir(baseDir):
    files.append(os.path.join(baseDir, filename))

sorted_files = sorted(files, key=lambda x: int(x.split('_CH')[1].split('@')[0]))

# =======================================================================================================
# Plotting the waveforms
# =======================================================================================================

num_rows = 4
num_cols = 2
total_iterations = num_rows * num_cols  #  for tqdm

num_colors = 6  #  for cm
inferno = cm.get_cmap('inferno', num_colors)
viridis = cm.get_cmap('viridis', num_colors)
norm = Normalize(vmin=0, vmax=num_colors - 1)

# -------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7, 8), constrained_layout=True, sharex=True)
fig.supxlabel('Time [$\mu s$]')
fig.supylabel('Charge Signal [a.u.]')

# -------------------------------------------------------------------------------------------------------

for idx in tqdm(range(total_iterations), desc='Plotting waveforms', unit="%", unit_scale=True):
    i = idx // num_cols
    j = idx % num_cols
    if idx < 7:
        data = readFile(files[idx])
        waveforms = data['wave']
        for val, color_index in zip(np.arange(1, 12e4, 2e4), np.arange(num_colors)):
            val = int(val)
            ax[i, j].set_xlim(0, 20)
            #  ax[i, j].set_ylim(3000, 5700)
            ax[i, j].plot(np.arange(data['n_samples'][0]) / 250, waveforms[val],
                          color=viridis(norm(color_index)),
                          label=f'Sample No.: {val}')
        # -----------------------------------------------------------------------------------------------
        text_x = 0.95
        text_y = 0.12
        ax[i, j].text(text_x, text_y, f'Channel {idx}', transform=ax[i, j].transAxes, ha='right', va='top')
        # -----------------------------------------------------------------------------------------------
        ax[i, j].tick_params(direction='in', which='both', width=1, top=True, right=True)
        ax[i, j].grid(which='major', axis='both', alpha=0.5, color='grey')
        ax[i, j].xaxis.set_major_locator(MultipleLocator(4))
        ax[i, j].xaxis.set_minor_locator(MultipleLocator(1))
        ax[i, j].yaxis.set_major_locator(MultipleLocator(5e2))
        ax[i, j].yaxis.set_minor_locator(MultipleLocator(250))
    # ---------------------------------------------------------------------------------------------------
    else:
        for val, color_index in zip(np.arange(1, 12e4, 2e4), np.arange(num_colors)):
            val = int(val)
            ax[i, j].plot([], [],
                          color=viridis(norm(color_index)),
                          label=f'Sample Number.: {val}')
            ax[i, j].legend(loc = 'center')
            ax[i, j].axis('off')

#  fig.savefig('Plots/Waveform_run1.png', dpi=300)

# =======================================================================================================
# Baseline correction & Waveform deconvolution
# =======================================================================================================

tau = 15e-6  # Time constant in microseconds
fs = 250e6  # Sampling frequency in Hz
alpha = np.exp(-1 / (tau * fs)) 
a_coeff = [1,-1] # denominator coefficient
b_coeff = [1, -alpha] # numerator coefficien

fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7, 8), constrained_layout=True, sharex=True)
fig.supxlabel('Time [$\mu s$]')
fig.supylabel('Charge Signal [a.u.]')

# -------------------------------------------------------------------------------------------------------

for idx1 in tqdm(range(total_iterations), desc='Plotting baseline corrected waveforms', unit="%", unit_scale=True):
    i = idx1 // num_cols
    j = idx1 % num_cols
    if idx1 < 7:
        data = readFile(files[idx1])
        waveforms = data['wave']
        # -----------------------------------------------------------------------------------------------
        offset = np.mean(waveforms[:, :500], axis=1)
        baseline_corrected_waveforms = waveforms - offset[:, np.newaxis]
        # -----------------------------------------------------------------------------------------------
        """
        #  Reduce number of x-values in the rise region (around 4 microseconds or 1000 samples)
        max_indexes = np.argmax(baseline_corrected_waveforms, axis=1)  #  printing returns [ 971 1005  954 ...  953 1009 1074]
        
        for idx, idxm in enumerate(max_indexes):
            print(baseline_corrected_waveforms[idx, idxm - 100], baseline_corrected_waveforms[idx, idxm]) 
        => remove 100 samples before the rise
        num_samples_to_replace = 100
        for idx, max_idx in enumerate(max_indexes):
            baseline_corrected_waveforms[idx, max_idx - num_samples_to_replace:max_idx] = 0
        
        for idx, idxm in enumerate(max_indexes):
            print(baseline_corrected_waveforms[idx, idxm - 110], baseline_corrected_waveforms[idx, idxm - 1],
                  baseline_corrected_waveforms[idx, idxm])
        """
        # -----------------------------------------------------------------------------------------------
        pole_zero_corrected_waveforms = np.zeros_like(baseline_corrected_waveforms)
        for idx2, waveform in enumerate(baseline_corrected_waveforms):
            pole_zero_corrected_waveforms[idx2] = lfilter(b_coeff, a_coeff, waveform)
        # -----------------------------------------------------------------------------------------------
        ax[i, j].plot(np.arange(data['n_samples'][0]) / 250, baseline_corrected_waveforms[0],
                           color='grey', lw=1, alpha=1)
        for val, color_index in zip(np.arange(1, 117582, 2e4), np.arange(num_colors)):
            val = int(val)
            ax[i, j].set_xlim(0, 20)
            ax[i, j].set_ylim(-200, 1000)
            ax[i, j].plot(np.arange(data['n_samples'][0]) / 250, pole_zero_corrected_waveforms[val],
                          color=viridis(norm(color_index)), alpha=1)
        # -----------------------------------------------------------------------------------------------
        text_x = 0.95
        text_y = 0.12
        ax[i, j].text(text_x, text_y, f'Channel {idx1}', transform=ax[i, j].transAxes, ha='right', va='top')
        # -----------------------------------------------------------------------------------------------
        ax[i, j].tick_params(direction='in', which='both', width=1, top=True, right=True)
        ax[i, j].grid(which='major', axis='both', alpha=0.5, color='grey')
        ax[i, j].xaxis.set_major_locator(MultipleLocator(4))
        ax[i, j].xaxis.set_minor_locator(MultipleLocator(1))
        ax[i, j].yaxis.set_major_locator(MultipleLocator(200))
        ax[i, j].yaxis.set_minor_locator(MultipleLocator(100))
    # ---------------------------------------------------------------------------------------------------
    else:
        ax[i, j].plot([], [],
                          color='grey',
                          ls = 'solid', lw=1, alpha = 1,
                          label=f'Baseline correction')
        for val, color_index in zip(np.arange(1, 117582, 2e4), np.arange(num_colors)):
            val = int(val)
            ax[i, j].plot([], [],
                          color=viridis(norm(color_index)),
                          ls = 'solid', alpha=1,
                          label=f'Deconvolved Event No. {val}')
            ax[i, j].legend(loc = 'center', fontsize=10)
            ax[i, j].axis('off')

#  fig.savefig('Plots/Waveform_corrected.png', dpi=300)

plt.show()