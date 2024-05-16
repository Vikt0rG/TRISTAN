import os
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from scipy.signal import lfilter

from lmfit.models import GaussianModel
from scipy.stats import linregress

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

data = readFile(sorted_files[0])  #  Take the zeroth channel
waveforms = data['wave']
offset = np.mean(waveforms[:, :500], axis=1)
baseline_corrected_waveforms = waveforms - offset[:, np.newaxis]

pole_zero_corrected_waveforms = np.zeros_like(baseline_corrected_waveforms)
for idx, waveform in enumerate(baseline_corrected_waveforms):
    pole_zero_corrected_waveforms[idx] = lfilter(b_coeff, a_coeff, waveform)

last_500_avg = np.mean(pole_zero_corrected_waveforms[:, -500:], axis=1)
first_500_avg = np.mean(pole_zero_corrected_waveforms[:, :500], axis=1)

energy = last_500_avg - first_500_avg

mask = energy > 1000
print(f'Counts above 1000: {np.sum(mask)}')

# =======================================================================================================
# Plotting energy differences histogram: 1 Channel
# =======================================================================================================

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.hist(energy, bins=1500, range=(250, 1750), color=plt.get_cmap('viridis')(0.75), alpha=0.70)
ax.set_xlabel('Energy [a.u.]')
ax.set_ylabel('Event counts')
ax.grid(which='major', color='grey', alpha=0.7)

#ax.set_xlim(250, 1750)

ax.tick_params(direction='in', which='both', width=1, pad=5)
ax.xaxis.set_major_locator(MultipleLocator(250))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.yaxis.set_minor_locator(MultipleLocator(50))

fig.savefig('Plots/Energy_distribution_uncal.png', dpi=300)
plt.show()

# =======================================================================================================
# Plotting energy differences histogram: All Channels
# =======================================================================================================

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)

ax.set_xlabel('Energy [a.u.]')
ax.set_ylabel('Event counts')
ax.grid(which='major', color='grey', alpha=0.7)
# -------------------------------------------------------------------------------------------------------
ax.set_xlim(1300, 1650)
ax.tick_params(direction='in', which='both', top=True, right=True, width=1, pad=5)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_minor_locator(MultipleLocator(100))
# -------------------------------------------------------------------------------------------------------
num_colors = len(sorted_files)
viridis = cm.get_cmap('viridis', num_colors)
norm = Normalize(vmin=0, vmax=num_colors - 1)
# -------------------------------------------------------------------------------------------------------

for idx1, filename in enumerate(tqdm(sorted_files, desc='Plotting histograms', unit="%", unit_scale=True)):
    data = readFile(filename)
    energy = data['energy']
    # ---------------------------------------------------------------------------------------------------
    ax.hist(energy, bins=350, range=(1300, 1650), color=viridis(norm(idx1)), alpha=0.70, label=f'Channel No. {idx1}')

ax.legend(loc='upper left')    
fig.savefig('Plots/E_distribution_mult.png', dpi=300)

# =======================================================================================================
# Fitting the data
# =======================================================================================================

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
# -------------------------------------------------------------------------------------------------------
num_colors = len(sorted_files)
viridis = cm.get_cmap('viridis', num_colors)
norm = Normalize(vmin=0, vmax=num_colors - 1)
# -------------------------------------------------------------------------------------------------------
data = readFile(sorted_files[0])
energy = data['energy']

gmodel_alpha = GaussianModel(prefix='alpha_')
gmodel_beta = GaussianModel(prefix='beta_')
# -------------------------------------------------------------------------------------------------------
initial_guesses_alpha = {'alpha_center': 1475, 'alpha_amplitude': 3000, 'alpha_sigma': 20}
counts_alpha, bin_edges_alpha, _ = ax.hist(energy, bins=200, range=(1325, 1525),
                                           color=plt.get_cmap('inferno')(0.85), alpha=0.7, label=f'Channel No. 0')
bin_centers_alpha = (bin_edges_alpha[:-1] + bin_edges_alpha[1:]) / 2
result_alpha = gmodel_alpha.fit(counts_alpha, x=bin_centers_alpha, method='leastsq', **initial_guesses_alpha)
# -------------------------------------------------------------------------------------------------------
initial_guesses_beta = {'beta_center': 1550, 'beta_amplitude': 500, 'beta_sigma': 20}
counts_beta, bin_edges_beta, _ = ax.hist(energy, bins=200, range=(1525, 1725),
                                         color=plt.get_cmap('plasma')(0.85), alpha=0.7)
bin_centers_beta = (bin_edges_beta[:-1] + bin_edges_beta[1:]) / 2
result_beta = gmodel_beta.fit(counts_beta, x=bin_centers_beta, method='leastsq', **initial_guesses_beta)
# -------------------------------------------------------------------------------------------------------
alpha_center = result_alpha.params['alpha_center'].value
alpha_amplitude = result_alpha.params['alpha_amplitude'].value
beta_center = result_beta.params['beta_center'].value
beta_amplitude = result_beta.params['beta_amplitude'].value
mean_alpha_lsb = alpha_center
mean_beta_lsb = beta_center
# -------------------------------------------------------------------------------------------------------
ax.plot(bin_centers_alpha, result_alpha.best_fit, linestyle='--', lw=1.5, label=r'Mn-$K_{\alpha}$ fit',
        color=plt.get_cmap('plasma')(0.30))
ax.plot(bin_centers_beta, result_beta.best_fit, linestyle='--', lw=1.5, label=r'Mn-$K_{\beta}$ fit',
        color=plt.get_cmap('plasma')(0.60))
ax.axvline(x=mean_alpha_lsb, label=fr'Mean energy $E_{{K_{{\alpha}}}}$ {mean_alpha_lsb:.2f} a.u.',
           ls='dashdot', lw=1, color=plt.get_cmap('viridis')(0.15))
ax.axvline(x=mean_beta_lsb, label=fr'Mean energy $E_{{K_{{\beta}}}}$ {mean_beta_lsb:.2f} a.u.',
           ls='dashdot', lw=1, color=plt.get_cmap('viridis')(0.15))
# -------------------------------------------------------------------------------------------------------
ax.set_xlim(1375, 1600)
ax.set_ylim(0, 3500)
ax.tick_params(direction='in', which='both', top=True, right=True, width=1, pad=5)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_minor_locator(MultipleLocator(100))
# -------------------------------------------------------------------------------------------------------
ax.set_xlabel('Energy [a.u]')
ax.set_ylabel('Event counts')

ax.legend()
fig.savefig('Plots/E_distribution_fit.png', dpi=300)
plt.show()
# -------------------------------------------------------------------------------------------------------

E_alpha = 5.90  # keV
E_beta = 6.49   # keV
# Linear regression to relate measured and true energies
slope, intercept, _, _, _ = linregress([mean_alpha_lsb, mean_beta_lsb], [E_alpha, E_beta])
energy_kev = slope * energy + intercept
print(energy_kev)