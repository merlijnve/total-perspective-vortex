# Total Perspective Vortex 🧠

> A Brain-Computer Interface using Machine Learning on EEG Data

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange.svg)](https://scikit-learn.org/)
[![MNE](https://img.shields.io/badge/MNE-1.1.1-green.svg)](https://mne.tools/)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Dataset Information](#dataset-information)
- [Requirements](#requirements)

## Overview

This project implements a **brain-computer interface (BCI)** that classifies motor movements and motor imagery from electroencephalographic (EEG) data. Using machine learning algorithms, the system can infer what a subject is thinking about or doing based on their EEG signals.

The project processes 64-channel EEG recordings from 109 subjects performing various motor tasks (moving or imagining moving left/right fists, or both fists/feet) and achieves **69% mean accuracy** on test data across all experimental conditions.

**Key Technologies:**
- **Signal Processing:** MNE-Python for EEG data preprocessing and visualization
- **Dimensionality Reduction:** Custom implementation of Common Spatial Patterns (CSP)
- **Classification:** Linear Discriminant Analysis (LDA)
- **Pipeline:** Scikit-learn for end-to-end ML pipeline

## Features

- ✅ **Complete EEG Processing Pipeline:** From raw data to predictions
- ✅ **Custom CSP Implementation:** Dimensionality reduction optimized for EEG
- ✅ **Multi-task classification:** Supports 6 different experimental conditions
- ✅ **Real-time Prediction:** Classify motor tasks from streaming EEG data
- ✅ **Comprehensive Visualization:** Raw data, PSD, evoked responses, and CSP patterns
- ✅ **Model Persistence:** Save and load trained models for different tasks
- ✅ **Cross-Validation:** Robust evaluation with train/validation/test splits

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/merlijnve/total-perspective-vortex.git
cd total-perspective-vortex
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
   
   Download the EEG Motor Movement/Imagery Dataset from [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) and configure the path in `config.py`:
   ```python
   DATASET_PATH = "/path/to/your/dataset/"
   ```

## Usage

### Training Models

Train models on all subjects and experimental conditions:
```bash
python total_perspective_vortex.py
```

Train on a specific subject and run:
```bash
python total_perspective_vortex.py <subject_number> <run_number>
# Example: Train on subject 1, run 3
python total_perspective_vortex.py 1 3
```

### Making Predictions

Use a trained model to predict on new data:
```bash
python predict.py <model_name> <subject_number> <run_number>
# Example: Predict left/right fist movements for subject 5, run 3
python predict.py Left_right_fist 5 3
```

**Available Models:**
- `Left_right_fist` - Physical left or right fist movements
- `Imagine_left_right_fist` - Imagined left or right fist movements
- `Fists_feet` - Physical both fists or both feet movements
- `Imagine_fists_feet` - Imagined both fists or both feet movements
- `Movement_of_fists` - Combined real and imagined fist movements
- `Movement_fists_feet` - Combined real and imagined fists/feet movements

## Project Structure

```
.
├── total_perspective_vortex.py    # Main training script
├── predict.py                     # Prediction script for trained models
├── MyCSP.py                       # Custom CSP implementation
├── read_dataset.py                # Dataset loading and preprocessing utilities
├── plotting.py                    # Visualization functions
├── config.py                      # Configuration and experiment definitions
├── requirements.txt               # Python dependencies
├── Dataset_explanation.md         # Detailed dataset documentation
├── models/                        # Saved trained models (*.joblib)
└── README.md                      # This file
```

### Key Components

- **`MyCSP.py`**: Custom implementation of Common Spatial Patterns algorithm using scikit-learn's `BaseEstimator` and `TransformerMixin`
- **`total_perspective_vortex.py`**: End-to-end pipeline including data loading, preprocessing, training, and evaluation
- **`predict.py`**: Real-time prediction on streaming EEG data with <2s latency
- **`read_dataset.py`**: Functions for reading EDF files, creating epochs, and filtering signals
- **`config.py`**: Centralized configuration for experiments, parameters, and dataset paths

## Technical Details

### Pipeline Architecture

The system uses a scikit-learn `Pipeline` with two stages:

1. **CSP (Common Spatial Patterns)**: Dimensionality reduction that maximizes variance differences between classes
2. **LDA (Linear Discriminant Analysis)**: Classification with eigen solver and automatic shrinkage

```python
Pipeline([
    ("CSP", MyCSP(n_components=4)),
    ("LDA", LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto'))
])
```

### Signal Processing

- **Filtering**: Bandpass filter to retain essential EEG frequency bands
- **Epoching**: Segment continuous EEG into task-related epochs
- **Channel Selection**: Focus on motor cortex channels (C5, C3, C1, Cz, C2, C4, C6)
- **Class Balancing**: Equal representation of classes in training data
- **Averaging**: Optional epoch averaging to increase training data

### Experimental Conditions

The system classifies 6 different motor tasks:

1. **Left/Right Fist** (runs 3, 7, 11): Physical movement
2. **Imagine Left/Right Fist** (runs 4, 8, 12): Motor imagery
3. **Fists/Feet** (runs 5, 9, 13): Physical movement
4. **Imagine Fists/Feet** (runs 6, 10, 14): Motor imagery
5. **Movement of Fists** (combined runs): Real + imagined fist movements
6. **Movement Fists/Feet** (combined runs): Real + imagined fists/feet movements

## Results

### Classification Accuracy by Task

| Task | Left/Right Fist | Imagine Left/Right Fist | Fists/Feet | Imagine Fists/Feet | Movement of Fists | Movement Fists/Feet |
|------|----------------|------------------|------------|-------------------|----------------|-------------------|
| **Train** | 0.97 (97%) | 0.97 (97%) | 0.75 (75%) | 0.84 (84%) | 0.96 (96%) | 0.76 (76%) |
| **Cross-Val** | 0.96 (96%) | 0.95 (95%) | 0.61 (61%) | 0.74 (74%) | 0.95 (95%) | 0.73 (73%) |
| **Test** | 0.74 (74%) | 0.75 (75%) | 0.60 (60%) | 0.63 (63%) | 0.77 (77%) | 0.63 (63%) |

### Overall Performance

| Metric | Accuracy |
|--------|----------|
| **Mean Train** | 0.87 (87%) |
| **Mean Cross-Val** | 0.82 (82%) |
| **Mean Test** | 0.69 (69%) ✅ |

The system achieves the target **>60% accuracy** on test data across all subjects and experimental conditions, demonstrating robust generalization to unseen data.

### Key Observations

- **Physical movements** (left/right fist) achieve higher accuracy (74-77%) than **fists/feet** tasks (60-63%)
- **Motor imagery** tasks show comparable performance to physical movements
- Strong cross-validation scores (82%) indicate the model generalizes well
- Test accuracy of 69% exceeds the minimum requirement of 60%

## Dataset Information

The project uses the **EEG Motor Movement/Imagery Dataset** from PhysioNet:
- **109 subjects** performing motor and motor imagery tasks
- **64-channel EEG** recorded at 160 Hz using BCI2000
- **14 experimental runs** per subject (2 baseline, 12 task runs)
- Standardized **10-10 electrode placement** system

For detailed information about the dataset structure, experimental protocol, and data format, see [Dataset_explanation.md](Dataset_explanation.md).

**Dataset Citation:**
> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

**Dataset URL:** https://physionet.org/content/eegmmidb/1.0.0/

## Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies. Key packages:

- **mne==1.1.1** - EEG data processing and visualization
- **scikit-learn==1.2.0** - Machine learning pipeline and algorithms
- **numpy==1.23.2** - Numerical computing
- **pandas==2.0.3** - Data manipulation
- **matplotlib==3.5.3** - Plotting and visualization
- **joblib==1.2.0** - Model serialization
