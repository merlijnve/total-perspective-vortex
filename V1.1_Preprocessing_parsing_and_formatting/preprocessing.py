import numpy as np
import mne
import sys


#T0 corresponds to rest
#
#T1 corresponds to onset of motion (real or imagined) of:
#   - the left fist (in runs 3, 4, 7, 8, 11, and 12)
#   - both fists (in runs 5, 6, 9, 10, 13, and 14)
#
#T2 corresponds to onset of motion (real or imagined) of
#   - the right fist (in runs 3, 4, 7, 8, 11, and 12)
#   - both feet (in runs 5, 6, 9, 10, 13, and 14)

baseline = [1, 2]
left_or_hands = [3, 4, 7, 8, 11, 12]
right_or_feet = [5, 6, 9, 10, 13, 14]

def info_raw_data():
    print(raw_data)
    print(raw_data.info)

def plot_raw_data():
    raw_data.plot(duration=60, use_opengl=True,block=True)

print(sys.argv)
raw_data = mne.io.read_raw_edf(sys.argv[1])
events = mne.events_from_annotations(raw_data)
info_raw_data()
plot_raw_data()
