"""
1. Left or right fist [R3, R7, R11]
2. Imagine left or right fist [R4, R8, R12]
3. Both fists or feet [R5, R9, R13]
4. Imagine both fists or feet [R6, R10, R14]
5. Baseline eyes open [R1]
6. Baseline eyes closed [R2]

# V.1.1 Preprocessing, parsing and formatting

First, you’ll need to parse and explore EEG data with MNE, from physionet.
You will have to write a script to visualize raw data and then filter it to
keep only useful frequency bands, and visualize again after this preprocessing.

This part is where you’ll decide which features you’ll extract from the
signals to feed them to your algorithm. So you’ll have to be thorough in
picking what matters for the desired output.

One example is to use the power of the signal by frequency and by channel to
the pipeline’s input. Most of the algorithms linked to filtering and obtaining
the signal’s specter use Fourier transform or wavelet transform (cf. bonus).

**Experiment structure**

T0 corresponds to rest

T1 corresponds to onset of motion (real or imagined) of:
  - the left fist (in runs 3, 4, 7, 8, 11, and 12)
  - both fists (in runs 5, 6, 9, 10, 13, and 14)

T2 corresponds to onset of motion (real or imagined) of:
  - the right fist (in runs 3, 4, 7, 8, 11, and 12)
  - both feet (in runs 5, 6, 9, 10, 13, and 14)

Therefore there are 5 targets:
  - Rest
  - Left fist
  - Right fist
  - Both fists
  - Both feet

**DATASET STRUCTURE:**

```
dataset/S{subject_nr}/S{subject_nr}{run_nr}.edf
```
"""

import joblib
import matplotlib.pyplot as plt
import math
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne import pick_types
import numpy as np
import mne

from MyCSP import MyCSP


experiments = [
    {
        "name": "Left_right_fist",
        "description": "open and close left or right fist",
        "runs": [3, 7, 11],
        "mapping": {
            0: "Rest",
            1: "Left fist",
            2: "Right fist"
        },
    },
    {
        "name": "Imagine_left_right_fist",
        "description": "imagine opening and closing left or right fist",
        "runs": [4, 8, 12],
        "mapping": {
            0: "Rest",
            1: "Imagine left fist",
            2: "Imagine right fist"
        },
    },
    {
        "name": "Fists_feet",
        "description": "open and close both fists or both feet",
        "runs": [5, 9, 13],
        "mapping": {
            0: "Rest",
            1: "Both fists",
            2: "Both feet"
        },
    },
    {
        "name": "Imagine_fists_feet",
        "description": "imagine opening and closing both fists or both feet",
        "runs": [6, 10, 14],
        "mapping": {
            0: "Rest",
            1: "Imagine both fists",
            2: "Imagine both feet"
        },
    },
    # {
    #     "name": "Movement_of_fists",
    #     "description": "movement (real or imagined) of fists",
    #     "runs": [3, 7, 11, 4, 8, 12],
    #     "mapping": {0: "Rest", 1: "Left fist", 2: "Right fist"},
    # },
    # {
    #     "name": "Movement_fists_feet",
    #     "description": "movement (real or imagined) of fists or feet",
    #     "runs": [5, 9, 13, 6, 10, 14],
    #     "mapping": {0: "Rest", 1: "Both fists", 2: "Both feet"},
    # },
]

PATH = "/Users/mvan-eng/goinfre/dataset/"
amount_of_subjects = 109
amount_of_runs = 14
batch_read = 50

plotting = False

good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]

f_low = 1.0
f_high = 15.0

csp_params = {
    "n_components": 4
}


def get_filepath(subject_nr=1, run_nr=1):
    subject = "S{:03d}".format(subject_nr)
    run = "R{:02d}".format(run_nr)
    filename = subject + "/" + subject + run + ".edf"
    filepath = PATH + filename
    return filepath


def make_runs():
    """**Creating a metadata array containing information about all runs**"""
    runs = []
    for subject_nr in range(1, amount_of_subjects + 1):
        for run_nr in range(3, amount_of_runs + 1):
            filepath = get_filepath(subject_nr=subject_nr, run_nr=run_nr)
            runs.append([run_nr, filepath])
    return runs


def get_mapping(run_nr):
    event_mapping = {}
    for e in experiments:
        if run_nr in e["runs"]:
            event_mapping = e["mapping"]
    return event_mapping


def read_dataset_batch(ex_nr, batch, start):
    raws = []

    batch_counter = batch
    start_counter = start
    for r in runs:
        if r[0] in experiments[ex_nr]["runs"]:
            if batch_counter == 0:
                break
            if start_counter == 0:
                raw = mne.io.read_raw_edf(r[1], preload=True)
                if raw.info['sfreq'] != 160.0:
                    raw.resample(sfreq=160.0)
                mne.datasets.eegbci.standardize(raw)
                raw.set_montage("standard_1005")

                events, _ = mne.events_from_annotations(
                                                    raw,
                                                    event_id=dict(T1=1, T2=2))
                mapping = get_mapping(r[0])
                ann = mne.annotations_from_events(
                                        events=events,
                                        event_desc=mapping,
                                        sfreq=raw.info["sfreq"]
                                        )
                raw.set_annotations(ann)
                raws.append(raw)
                batch_counter -= 1
            else:
                start_counter -= 1

    if len(raws) == 0:
        return None
    raw = mne.concatenate_raws(raws)
    raw = filter_raw(raw)

    channels = raw.info["ch_names"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw.drop_channels(bad_channels)

    return raw


def plot_raw(raw):
    """**Plot an instance of raw data**"""
    if plotting:
        raw.plot(scalings=dict(eeg=250e-6), events=events)


def plot_psd_raw(raw, fmin=0, fmax=np.inf):
    """**Plot the (average) Power Spectral Density of a raw instance**

    The power spectral density (PSD) of the signal describes the power
    present in the signal as a function of frequency, per unit frequency.
    """
    if plotting:
        psd = raw.compute_psd(fmin=fmin, fmax=fmax)
        psd.plot(average=True)


def filter_raw(raw):
    """**Filtering**

    - Simple bandpass
    - Notch filter to filter out 60hz electrical signals
    """
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(60, method="iir")
    raw_filtered.filter(f_low, f_high, fir_design="firwin",
                        skip_by_annotation="edge")
    return raw_filtered


def create_epochs(raw):
    """**Creating epochs**

    In the MNE-Python library, an "epoch" is a defined time window of EEG
    (Electroencephalography) or MEG (Magnetoencephalography) data that is
    extracted from continuous data based on specific events or triggers.
    """
    tmin = -.500  # start of each epoch (in sec)
    tmax = 2.000  # end of each epoch (in sec)
    baseline = (None, 0)
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")

    epochs = mne.Epochs(raw,
                        events=events,
                        event_id=event_dict,
                        tmin=tmin, tmax=tmax,
                        baseline=baseline,
                        picks=picks,
                        proj=True,
                        preload=True)

    return epochs


def balance_classes(epochs):
    print("Balancing classes...")
    event_id = epochs.event_id
    keys = list(event_id.keys())

    small, big = (keys[0], keys[1]) if len(epochs[keys[0]]) < len(epochs[keys[1]]) else (keys[1], keys[0])
    diff = len(epochs[big]) - len(epochs[small])

    indices = []
    for i in range(len(epochs.events[:, -1])):
        if len(indices) == diff:
            break
        if epochs.events[i, -1] == event_id[big]:
            indices.append(i)
    epochs.drop(indices)

    return epochs


def plot_evoked(evoked):
    """**Plotting an evoked**"""
    if plotting:
        evoked.plot(gfp=True)
        evoked.plot_topomap(times=[-0.2, 0.2, 0.4, 0.6, 0.8], average=0.05)


if plotting:
    plot_evoked(experiments[0]["epochs"]['Left fist'].average())

if plotting:
    experiments[0]["epochs"]['Left fist'].plot_image(picks=["Cz"])


def split_epochs_train_test(experiment):
    """ **Creating data and targets**"""
    E = experiment["epochs"]
    y = experiment["y"]
    test_amount = math.ceil(0.15 * len(E))

    E_test = E[:test_amount]
    y_test = y[:test_amount]

    E_train = E[test_amount:]
    y_train = y[test_amount:]

    return E_test, y_test, E_train, y_train


def average_over_epochs(ex):
    E_test, y_test, E_train, y_train = split_epochs_train_test(ex)
    new_x = []
    new_y = []

    # Optional: averaging over multiple sizes to increase dataset size
    sizes = [30]

    event_id = E_train.event_id
    keys = list(event_id.keys())

    if len(E_train[keys[0]]) > len(E_train[keys[1]]):
        max_len = len(E_train[keys[1]])
    else:
        max_len = len(E_train[keys[0]])

    for avg_size in sizes:
        print("Averaging epochs over size: ", avg_size, "...")
        i = 0
        while i < max_len:
            x_averaged = E_train[keys[0]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[0]])

            x_averaged = E_train[keys[1]][i:i+avg_size].average().get_data()
            new_x.append(x_averaged)
            new_y.append(event_id[keys[1]])

            if i + avg_size >= len(E_train):
                avg_size = len(E_train) - i
            i = i + avg_size

    return np.array(new_x), np.array(new_y)


def plot_csp_separation(X, y):
    """Plot a scatter_plot of X_train after CSP transformation"""
    if plotting:
        csp = MyCSP(csp_params["n_components"])
        X = csp.fit_transform(X, y)

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()


if plotting:
    plot_csp_separation(experiments[0]["X_avg"], experiments[0]["y_avg"])


"""# V.1.2 Treatment pipeline

Then the processing pipeline has to be set up:

• Dimensionality reduction algorithm (i.e.: PCA, ICA, CSP, CSSP...)

• Classification algorithm, there is plenty of choice among those available in
sklearn, to output the decision of what data chunk corresponds to what kind of
motion.

• "Playback" reading on the file to simulate a data stream.

It is advised to first test your program architecture with sklearn and MNE
algorithms, before implementing your own CSP or whatever algorithm you chose.

The program will have to contain a script for training and a script for
prediction.
The script predicting output will have to do it on a stream of data, and
within a delay of 2s after the data chunk was sent to the processing pipeline.
(You should not use mne-realtime)

You have to use the pipeline object from sklearn (use baseEstimator and
transformerMixin classes of sklearn)

"""

if plotting:
    mne.viz.plot_compare_evokeds(
        dict(left_fist=experiments[0]["epochs"]["Left fist"].average(),
             right_fist=experiments[0]["epochs"]["Right fist"].average()),
        legend="upper left"
    )

if plotting:
    experiments[0]["epochs"]["Left fist"].average().plot_joint(picks="eeg")
    _ = plot_evoked(experiments[0]["epochs"]["Left fist"].average())

if plotting:
    experiments[0]["epochs"]["Left fist"].average().compute_psd().plot()
    experiments[0]["epochs"]["Right fist"].average().compute_psd().plot()


"""# V.1.4 Train, Validation, and Test

• You have to use cross_val_score on the whole processing pipeline, to
evaluate your classification.

• You must choose how to split your data set between Train, Validation, and
Test set (Do not overfit, with different splits each time)

• You must have 60% mean accuracy on all subjects used in your Test Data
(corresponding to the six types of experiment runs and on never-learned data)

• You can train/predict on the subject and the task of your choice
"""
cv = ShuffleSplit(10, test_size=0.2, random_state=42)


def make_clf():
    csp = MyCSP(csp_params["n_components"])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')

    clf = Pipeline([
        ("CSP", csp),
        ("LDA", lda)
    ])
    return clf


def dump_model(ex):
    joblib.dump(ex["clf"], f'{ex["name"]}_{amount_of_subjects}_subjects.save')


def split_train_test(experiment):
    X = experiment["X"]
    y = experiment["y"]
    test_amount = math.ceil(0.15 * len(X))

    X_test = X[:test_amount]
    y_test = y[:test_amount]

    X_train = X[test_amount:]
    y_train = y[test_amount:]

    return X_test, y_test, X_train, y_train


for i, ex in enumerate(experiments):
    ex["epochs"] = []
    runs = make_runs()
    batch_start = 0
    buffer = read_dataset_batch(i, batch_read, batch_start)
    while buffer is not None:
        ex["raw"] = buffer
        events, event_dict = mne.events_from_annotations(ex["raw"])
        batch_start += batch_read

        ex["epochs"].append(create_epochs(ex["raw"]))
        del ex["raw"]
        buffer = read_dataset_batch(i, batch_read, batch_start)

    ex["epochs"] = mne.concatenate_epochs(ex["epochs"])
    ex["epochs"] = balance_classes(ex["epochs"])

    ex["X"] = ex["epochs"].get_data()
    ex['y'] = ex["epochs"].events[:, -1]

    ex["X_avg"], ex["y_avg"] = average_over_epochs(ex)
    plot_csp_separation(ex["X_avg"], ex["y_avg"])

    ex["clf"] = make_clf()

    ex["clf"].fit(ex["X_avg"], ex["y_avg"])
    dump_model(ex)

    ex["crossval_scores"] = cross_val_score(
        ex["clf"], ex["X_avg"], ex["y_avg"], cv=cv, error_score='raise')


test_scores = []
train_scores = []
crossval_scores = []
print()

for ex in experiments:
    X_test, y_test, X_train, y_train = split_train_test(ex)
    train_score = ex["clf"].score(ex["X_avg"], ex["y_avg"])
    test_score = ex["clf"].score(X_test, y_test)

    print(ex["name"])
    print("Train: ", train_score)
    train_scores.append(train_score)

    print("Test: ", test_score)
    test_scores.append(test_score)

    print("Crossval: %f" % (np.mean(ex["crossval_scores"])))
    crossval_scores.append(np.mean(ex["crossval_scores"]))

    print()

print("Mean scores")
print("Train: ", round(sum(train_scores) / len(train_scores), 2))
print("Test: ", round(sum(test_scores) / len(test_scores), 2))
print("Crossval: ", round(sum(crossval_scores) / len(crossval_scores), 2))
