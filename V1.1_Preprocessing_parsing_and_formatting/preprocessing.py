import numpy as np
import mne


def info_raw_data():
    print(raw_data)
    print(raw_data.info)

def plot_raw_data():
    raw_data.plot(block=True)

raw_data = mne.io.read_raw_edf('S001R01.edf')
info_raw_data()
plot_raw_data()
