import mne
import sys
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold

# T0 corresponds to rest
#
# T1 corresponds to onset of motion (real or imagined) of:
#   - the left fist (in runs 3, 4, 7, 8, 11, and 12)
#   - both fists (in runs 5, 6, 9, 10, 13, and 14)
#
# T2 corresponds to onset of motion (real or imagined) of
#   - the right fist (in runs 3, 4, 7, 8, 11, and 12)
#   - both feet (in runs 5, 6, 9, 10, 13, and 14)
#
# DATASET STRUCTURE:
# dataset/S{subject_nr}/S{subject_nr}{run_nr}.edf
#

baseline = [1, 2]
left_or_hands = [3, 4, 7, 8, 11, 12]
right_or_feet = [5, 6, 9, 10, 13, 14]


# Read from new .edf file
# raw = mne.io.read_raw_edf(sys.argv[1], preload=True)
# Read from saved .fif file
print(mne.channels.get_builtin_montages())
raw = mne.io.read_raw_fif(sys.argv[1] + '-filt-raw.fif', preload=True)
raw.set_montage('standard_1010')

# Read events from raw
events = mne.events_from_annotations(raw)

# Standard filter
# low_cut = 0.1
# hi_cut = 30
# raw_filt = raw.copy().filter(low_cut, hi_cut)

# ICA (Independent Component Analysis) filter
ica_low_cut = 1
hi_cut = 30
raw_ica = raw.copy().filter(ica_low_cut, hi_cut)

# Break raw data into 1 s epochs
tstep = 1.0
events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
epochs_ica = mne.Epochs(raw_ica, events_ica, tmin=0.0,
                        tmax=tstep, baseline=None, preload=True)

# Get rejection threshold (excessively noisy)
reject = get_rejection_threshold(epochs_ica)
print(reject)

# ICA parameters
random_state = 42   # ensures ICA is reproducable each time it's run
# Specify n_components as a decimal to set % explained variance
ica_n_components = .99

# Fit ICA
ica = mne.preprocessing.ICA(
    n_components=ica_n_components, random_state=random_state)
ica.fit(epochs_ica,
        reject=reject,
        tstep=tstep)

ica.plot_components()

# Create power spectral density plot
# raw_filt.plot_psd()

# Create time domain plot
# raw_filt.plot(block=True)

# Save .fif files
# raw_filt.save(sys.argv[1] + '-filt-raw.fif', overwrite=True)
# raw.save(sys.argv[1] + '-raw.fif', overwrite=True)
