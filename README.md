# Total Perspective Vortex
## Brain-computer interface with machine learning based on electroencephalographic data

This project aims to create a brain-computer interface based on electroencephalographic
data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG
reading, you’ll have to infer what he or she is thinking about or doing - (motion) A or B
in a t0 to tn timeframe.

Dataset and explanation of the experiment can be found here: https://physionet.org/content/eegmmidb/1.0.0/

The project has a few main steps:

### 1. Preprocessing, Parsing, and Formatting
- Parse and explore EEG data using MNE from PhysioNet.
- Visualize raw data and apply filtering to retain essential frequency bands.
- Choose relevant features for algorithm input

### 2. Treatment Pipeline Setup
Establish a Scikit-Learn processing pipeline:
- Utilize a dimensionality reduction algorithm (CSP)
- Choose a classification algorithm from sklearn to identify types of motion in data chunks.
- Simulate a data stream through "playback" reading from a file.

### 3. Implementation of Dimensionality Reduction
- Implement the Common Spatial Patterns algorithm to express data using meaningful features.
- Use BaseEstimator and TransformerMixin classes to be able to integrate it into the sklearn pipeline.

### 4. Train, Validation, and Test
- Use cross_val_score on the entire processing pipeline to assess classification performance.
- Split the dataset into Train, Validation, and Test sets, ensuring no overfitting with different splits each time.
- Achieve a minimum 60% mean accuracy on all subjects in the Test Data, covering six types of experimental runs and using never-learned data.

___________________________
### Results
|   |Left_right_fist|Imagine_left_right_fist|Fists_feet|Imagine_fists_feet|Fists|Fists_feet|
|---|---|---|---|---|---|---|
|Train|0.97|0.97|0.75|0.84|0.96|0.76
|Crossval|0.96|0.95|0.61|0.74|0.95|0.73
|Test|0.74|0.75|0.60|0.63|0.77|0.63

### Mean scores
#### Train:  0.87
#### Crossval:  0.82
#### Test:  0.69
