import os
from tqdm import tqdm

import numpy as np

from lmfit.models import GaussianModel
from scipy.stats import linregress

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rc

from readFile import readFileShortened

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# For plots focus on zeroth channel

# =======================================================================================================
# Get data from directories
# =======================================================================================================

baseDir = "./data/noise_ch0"
files = []
for filename in os.listdir(baseDir):
    files.append(os.path.join(baseDir, filename))

sorted_files = sorted(files, key=lambda x: int(x.split('_run_25_04_2024_')[-1].split('_')[0][0]))
trap_times = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]) # in microsecons
res = np.ones_like(trap_times)

# =======================================================================================================
# Rescaling
# =======================================================================================================

data = readFileShortened(sorted_files[0])
energy = data['energy']

gmodel_alpha = GaussianModel(prefix='alpha_')
gmodel_beta = GaussianModel(prefix='beta_')
# -------------------------------------------------------------------------------------------------------
initial_guesses_alpha = {'alpha_center': 1475, 'alpha_amplitude': 3000, 'alpha_sigma': 20}
counts_alpha, bin_edges_alpha = np.histogram(energy, bins=200, range=(1325, 1525))
bin_centers_alpha = (bin_edges_alpha[:-1] + bin_edges_alpha[1:]) / 2
result_alpha = gmodel_alpha.fit(counts_alpha, x=bin_centers_alpha, method='leastsq', **initial_guesses_alpha)
# -------------------------------------------------------------------------------------------------------
initial_guesses_beta = {'beta_center': 1550, 'beta_amplitude': 500, 'beta_sigma': 20}
counts_beta, bin_edges_beta = np.histogram(energy, bins=200, range=(1525, 1725))
bin_centers_beta = (bin_edges_beta[:-1] + bin_edges_beta[1:]) / 2
result_beta = gmodel_beta.fit(counts_beta, x=bin_centers_beta, method='leastsq', **initial_guesses_beta)
# -------------------------------------------------------------------------------------------------------
alpha_center = result_alpha.params['alpha_center'].value
beta_center = result_beta.params['beta_center'].value
mean_alpha_lsb = alpha_center
mean_beta_lsb = beta_center

E_alpha = 5.90  # keV
E_beta = 6.49   # keV
slope, intercept, _, _, _ = linregress([mean_alpha_lsb, mean_beta_lsb], [E_alpha, E_beta])

# =======================================================================================================
# Noise curve
# =======================================================================================================

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

# -------------------------------------------------------------------------------------------------------
gmodel_alpha = GaussianModel(prefix='alpha_')

ranges = [(2.6, 3.2), (2.4, 2.9), (2.8, 3.2), (3.7, 4.1), (5.8, 6.1), (11.9, 12.4), (22.35, 22.7)]
for idx, (trap_rise_time, filename, range) in enumerate(tqdm(zip(trap_times, sorted_files, ranges), desc='Analyzing rise times', total=len(trap_times))):
    initial_guesses_alpha = {'alpha_center': np.mean(range), 'alpha_amplitude': 2000, 'alpha_sigma': 0.1}

    data = readFileShortened(filename)  # Replace with your actual file reading function
    energy = slope * data['energy'] + intercept

    counts_alpha, bin_edges_alpha = np.histogram(energy, bins=200, range = range)
    bin_centers_alpha = (bin_edges_alpha[:-1] + bin_edges_alpha[1:]) / 2
    result_alpha = gmodel_alpha.fit(counts_alpha, x=bin_centers_alpha, method='leastsq', **initial_guesses_alpha)

    # plt.bar(bin_centers_alpha, counts_alpha, width=(bin_edges_alpha[1] - bin_edges_alpha[0]), color='blue', alpha=0.7)
    # plt.show()

    alpha_sigma = result_alpha.params['alpha_sigma'].value
    res[idx] = alpha_sigma
# -------------------------------------------------------------------------------------------------------
ax.plot(trap_times, 2.35*res, marker='o', linestyle='solid', color=plt.get_cmap('inferno')(0.15))
ax.set_xlabel(r'Trap. Rise Time [$\mu$s]')
ax.set_ylabel(r'Energy Resolution [FWHM $\cdot$ keV]')
# -------------------------------------------------------------------------------------------------------
ax.tick_params(direction='in', which='both', top=True, right=True, width=1, pad=5)

major_ticks = ax.xaxis.get_majorticklocs()
major_tick_interval = major_ticks[1] - major_ticks[0]
ax.xaxis.set_minor_locator(MultipleLocator(major_tick_interval / 5))

major_ticks = ax.yaxis.get_majorticklocs()
major_tick_interval = major_ticks[1] - major_ticks[0]
ax.yaxis.set_minor_locator(MultipleLocator(major_tick_interval / 5))
# -------------------------------------------------------------------------------------------------------
ax.grid(True, color='grey', lw=1, alpha=0.7)

plt.savefig('Plots/noise_curve.png', dpi=300)
plt.show()