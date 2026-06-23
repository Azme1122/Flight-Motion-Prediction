# Flight Motion Prediction

This repository contains ongoing work on flight motion prediction using drone trajectory data. The current focus is to compare a classical Constant Velocity Kalman Filter baseline with AI-based probabilistic trajectory prediction methods.

At this stage, the project is not completed yet. The currently implemented methods are:

- Constant Velocity Kalman Filter (CV-KF)
- LSTM + Mixture Density Network (LSTM+MDN)

Next, other possible AI+MDN methods will be investigated and compared against the current implementations.

## Project Overview

The goal of this project is to predict future motion from observed trajectory data and evaluate how reliable the predictions are. The work includes both deterministic and probabilistic approaches:

- **CV-KF** predicts future positions using a constant-velocity motion model.
- **LSTM+MDN** predicts a probability distribution over future positions, which can represent uncertainty and multiple possible future motions.

This makes it possible to compare prediction accuracy, uncertainty quality, and reliability across different methods.

## Dataset

This project uses drone trajectory data from:

[CenekAlbl/drone-tracking-datasets](https://github.com/CenekAlbl/drone-tracking-datasets.git)

The dataset contains drone trajectories that are used as the current source data for preprocessing, training, testing, and evaluation.

Other possible datasets may also be used later for additional training, testing, and comparison.

## Repository Structure

```text
Flight Motion Prediction/
├── CV-KF_code/
│   ├── CV_KF_12MAY/
│   ├── CV_KF_18MAY/
│   ├── CV_KF_22JUNE/
│   └── CV_KF_26MAY/
└── AI_MDN_METHOD/
    ├── DATA_NORMALIZATION/
    ├── Data_Normalization_01_06/
    └── LSTM_MDN/
        ├── LSTM_MDN_01/
        ├── LSTM_MDN_02/
        ├── LSTM_MDN_Trial/
        └── LSTM_MDN_Trial_2/
```

## Implemented Methods

### 1. Constant Velocity Kalman Filter

The updated CV-KF implementation is located in:

```text
CV-KF_code/CV_KF_22JUNE/
```

This method is used as the classical baseline. It estimates and predicts trajectory motion using a constant-velocity Kalman filter model.

The updated implementation includes:

- raw drone trajectory data handling
- trajectory preprocessing
- Kalman filter prediction
- evaluation and summary generation
- visualization scripts
- result storage

Main folder:

```text
CV-KF_code/CV_KF_22JUNE/base_CV_KF/
```

Important files include:

- `preprocess_3d.py`
- `kalman_filter_model.py`
- `train.py`
- `testing.py`
- `metrics.py`
- `generate_eval_summary.py`
- `visualization.py`

### 2. LSTM + Mixture Density Network

The updated LSTM+MDN implementations are located in:

```text
AI_MDN_METHOD/LSTM_MDN/LSTM_MDN_01/
AI_MDN_METHOD/LSTM_MDN/LSTM_MDN_02/
```

These two versions are based on two different trajectory normalization methods.

Each version contains a `base_MDN` workflow with:

- preprocessing
- LSTM model code
- MDN output layer
- training
- testing
- evaluation
- summary generation
- visualization

Important files include:

- `preprocess_3d.py`
- `base_lstm.py`
- `mdn.py`
- `train.py`
- `testing.py`
- `eval.py`
- `generate_eval_summary.py`
- `vis.py`

## Data Normalization

The AI workflow includes different normalization approaches for preparing drone trajectory data before model training.

The two updated LSTM+MDN folders represent the current comparison between two normalization methods:

- `LSTM_MDN_01`
- `LSTM_MDN_02`

The purpose is to study how the normalization method affects prediction performance and uncertainty quality.

## Current Project Status

This repository is still under development.

Completed or currently implemented:

- drone trajectory data integration
- CV-KF baseline implementation
- LSTM+MDN implementation
- two normalization-based LSTM+MDN variants
- preprocessing, training, testing, and evaluation scripts

Planned next steps:

- investigate additional AI+MDN methods
- include other possible datasets for future training and testing
- compare all implemented methods using the same drone trajectory data
- evaluate prediction accuracy and uncertainty reliability
- organize final results and documentation

## Expected Outputs

Depending on the selected method, the code can generate:

- processed trajectory files
- predicted future trajectories
- trained model files
- evaluation metrics
- summary tables
- visualizations of trajectory predictions
- comparison results between methods

## Purpose

The purpose of this repository is to build and compare flight motion prediction methods for drone trajectory data. The final objective is to understand which approach gives the most accurate and reliable future trajectory prediction.
