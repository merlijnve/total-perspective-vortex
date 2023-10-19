# total-perspective-vortex
## Brain computer interface with machine learning based on electroencephalographic data

This subject aims to create a brain computer interface based on electroencephalographic
data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG
reading, you’ll have to infer what he or she is thinking about or doing - (motion) A or B
in a t0 to tn timeframe.

Dataset and explanation of the experiment can be found here: https://physionet.org/content/eegmmidb/1.0.0/


|   |Left_right_fist|Imagine_left_right_fist|Fists_feet|Imagine_fists_feet|Fists|Fists_feet|
|---|---|---|---|---|---|---|
|Train|0.97|0.97|0.75|0.84|0.96|0.76
|Crossval|0.96|0.95|0.61|0.74|0.95|0.73
|Test|0.74|0.75|0.60|0.63|0.77|0.63

___________________________
### Mean scores
#### Train:  0.87
#### Crossval:  0.82
### Test:  0.69