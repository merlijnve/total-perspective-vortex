import mne
import sys
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold
from pathlib import Path
import argparse

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
left_or_right = [3, 4, 7, 8, 11, 12]
hands_or_feet = [5, 6, 9, 10, 13, 14]


def parse_args():
    parser = argparse.ArgumentParser(
                        prog = 'Epoching',
                        description = 'Segment data into epochs')
    parser.add_argument('-s', '--subject', type=int, required=True) 
    parser.add_argument('-r', '--run', type=int, required=True) 
    args = parser.parse_args()
    if args.subject > 109 or args.subject < 1:
        sys.exit("Invalid subject number")
    if args.run in left_or_right:
        event_mapping = {'Rest':1, 'Left fist':2, 'Right fist':3}
    elif args.run in hands_or_feet:
        event_mapping = {'Rest':1, 'Both hands':2, 'Both feet':3}
    else:
        sys.exit("Invalid run number")
    return args, event_mapping

def format_filename(args):
    subject = "S{:03d}".format(args.subject)
    run = "R{:02d}".format(args.run)
    filename = subject + "/" + subject + run + ".edf"
    return filename

args, event_mapping = parse_args()
filename = format_filename(args)
filepath = "/Users/mvan-eng/projects/total-perspective-vortex-git/dataset/" + filename

# Read from new .edf file
raw = mne.io.read_raw_edf(filepath, preload=True)
# Read from saved .fif file
# raw = mne.io.read_raw_fif(filepath + '-filt-raw.fif', preload=True)

# Set montage after standardizing the datasets ch_names
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005")

# Read events from raw
events, event_dict = mne.events_from_annotations(raw)

######## FILTERING

# Standard filter
low_cut = 0.1
hi_cut = 30
raw_filt = raw.copy().filter(low_cut, hi_cut)

# ICA (Independent Component Analysis) filter
ica_low_cut = 1
hi_cut = 30
raw_ica = raw.copy().filter(ica_low_cut, hi_cut)

######## EPOCHING

# Break raw data into 1 s epochs
# tstep = 1.0
# events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
# epochs_ica = mne.Epochs(raw_ica, events_ica, tmin=0.0,
#                         tmax=tstep, baseline=None, preload=True)


# ######## ICA

# # Get rejection threshold (excessively noisy)
# reject = get_rejection_threshold(epochs_ica)
# print(reject)

# # ICA parameters
# random_state = 42   # ensures ICA is reproducable each time it's run
# # Specify n_components as a decimal to set % explained variance
# ica_n_components = .99

# # Fit ICA
# ica = mne.preprocessing.ICA(
#     n_components=ica_n_components, random_state=random_state)
# ica.fit(epochs_ica,
#         reject=reject,
#         tstep=tstep)

# # Plot ICA components (set picks argument to show more than 5)
# ica.plot_properties(epochs_ica, psd_args={'fmax': hi_cut})

# # Find and exclude Electro Oculogram (EOG) components (blinks, eye movements)
# ica_z_thresh = 1.96
# eog_indices, eog_scores = ica.find_bads_eog(raw_ica,
#                                             ch_name=['Fp1', 'F8'],
#                                             threshold=ica_z_thresh)
# ica.exclude = eog_indices

# # Save ICA
# # ica.save(filepath + '-ica.fif',
# #          overwrite=True)

######## PLOTTING

# Plot EOG scores (red = bad)
# ica.plot_scores(eog_scores)

# Create power spectral density plot
# raw_filt.plot_psd()

# Create time domain plot
# raw_filt.plot(block=True)


######## SAVE
# Save .fif files
raw_filt.save(filepath + '-filt-raw.fif', overwrite=True)
# raw.save(filepath + '-raw.fif', overwrite=True)
