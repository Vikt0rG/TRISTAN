import os
from joblib import Parallel, delayed
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

# -------------------------------------------------------------------------------------------------------
tau = 15e-6  # Time constant in microseconds
fs = 250e6  # Sampling frequency in Hz
alpha = np.exp(-1 / (tau * fs)) 
a_coeff = [1,-1] # denominator coefficient
b_coeff = [1, -alpha] # numerator coefficient
# -------------------------------------------------------------------------------------------------------
"""
data = readFile(files[0])  #  Take the zeroth channel
waveforms = data['wave']
offset = np.mean(waveforms[:, :500], axis=1)
baseline_corrected_waveforms = waveforms - offset[:, np.newaxis]

pole_zero_corrected_waveforms = np.zeros_like(baseline_corrected_waveforms)
for idx, waveform in enumerate(baseline_corrected_waveforms):
    pole_zero_corrected_waveforms[idx] = lfilter(b_coeff, a_coeff, waveform)

last_500_avg = np.mean(pole_zero_corrected_waveforms[:, -500:], axis=1)
first_500_avg = np.mean(pole_zero_corrected_waveforms[:, :500], axis=1)

energy = last_500_avg - first_500_avg

# =======================================================================================================
# Plotting energy differences histogram: 1 Channel
# =======================================================================================================

fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)

ax.hist(energy, bins=100, range=(0, 1000), color=plt.get_cmap('viridis')(0.75), alpha=0.70)
ax.set_xlabel('Energy [a.u.]')
ax.set_ylabel('Event counts')
ax.grid(which='major', color='grey', alpha=0.7)

ax.set_xlim(200, 1000)
ax.tick_params(direction='in', which='both', width=1, pad=5)
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(2000))
ax.yaxis.set_minor_locator(MultipleLocator(1000))

fig.savefig('Plots/Energy_distribution_uncal.png', dpi=300)
plt.show()
"""
# =======================================================================================================
# Plotting energy differences histogram: 1 Channel
# =======================================================================================================

fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)

ax.set_xlabel('Energy [a.u.]')
ax.set_ylabel('Event counts')
ax.grid(which='major', color='grey', alpha=0.7)
# -------------------------------------------------------------------------------------------------------
ax.set_xlim(300, 1000)
ax.tick_params(direction='in', which='both', width=1, pad=5)
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(2000))
ax.yaxis.set_minor_locator(MultipleLocator(1000))
# -------------------------------------------------------------------------------------------------------
num_colors = len(files)
viridis = cm.get_cmap('viridis', num_colors)
norm = Normalize(vmin=0, vmax=num_colors - 1)
# -------------------------------------------------------------------------------------------------------

for idx1, filename in enumerate(tqdm(files, desc='Plotting histograms', unit="%", unit_scale=True)):
    data = readFile(files[idx1])
    waveforms = data['wave']
    offset = np.mean(waveforms[:, :500], axis=1)
    baseline_corrected_waveforms = waveforms - offset[:, np.newaxis]

    pole_zero_corrected_waveforms = np.zeros_like(baseline_corrected_waveforms)
    for idx2, waveform in enumerate(baseline_corrected_waveforms):
        pole_zero_corrected_waveforms[idx2] = lfilter(b_coeff, a_coeff, waveform)

    last_500_avg = np.mean(pole_zero_corrected_waveforms[:, -500:], axis=1)
    first_500_avg = np.mean(pole_zero_corrected_waveforms[:, :500], axis=1)
    energy = last_500_avg - first_500_avg
    # ---------------------------------------------------------------------------------------------------
    ax.hist(energy, bins=100, range=(0, 1000), color=viridis(norm(idx1)), alpha=0.70)
    
fig.savefig('Plots/E_distribution_uncal_mult.png', dpi=300)
plt.show()