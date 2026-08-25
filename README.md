# Flight Motion Prediction

This repository contains selected implementations from my ongoing Master's thesis on probabilistic 3D motion prediction of flying objects for collision-free drone control.

> **Repository Notice:** This GitHub repository contains only a selected public subset of the ongoing thesis implementation. The primary development repository is maintained privately on the university GitLab infrastructure. Since the thesis is still in progress, newer experimental models, ongoing implementations, and intermediate research results are not currently published here.

The broader thesis investigates both physics-based and learning-based approaches, including a Constant Velocity Kalman Filter (CV-KF) baseline and sequence models such as LSTM, GRU, TCN, Transformer, and Mamba combined with a Mixture Density Network (MDN) output head for probabilistic trajectory prediction. It also investigates physics-enhanced residual learning approaches that combine model-based prediction with neural-network-based residual estimation.

The implementations currently available in this public repository are:

* Constant Velocity Kalman Filter (CV-KF)
* LSTM + Mixture Density Network (LSTM+MDN)

Additional methods currently being developed and evaluated as part of the thesis are maintained in the private university GitLab repository and may be added here later when appropriate.

## Project Overview

The goal of this project is to predict the future 3D motion of flying objects from observed trajectory data and evaluate both prediction accuracy and uncertainty quality.

The overall thesis investigates deterministic and probabilistic approaches:

* **CV-KF** predicts future positions using a constant-velocity motion model.
* **LSTM+MDN** predicts a probability distribution over future positions, allowing uncertainty and multiple possible future motions to be represented.
* **Additional sequence models** including GRU, TCN, Transformer, and Mamba are being investigated with MDN-based probabilistic output heads.
* **Physics-enhanced residual learning** is being investigated by combining model-based predictions with neural-network-based residual estimation.

Only the CV-KF and LSTM+MDN implementations are currently included in this public GitHub repository.

The methods are evaluated with respect to trajectory prediction accuracy as well as the quality and reliability of their probabilistic predictions.

## Dataset

This project currently uses drone trajectory data from:

[CenekAlbl/drone-tracking-datasets](https://github.com/CenekAlbl/drone-tracking-datasets.git)

The dataset contains drone trajectories used for preprocessing, training, testing, and evaluation.

Additional datasets may also be investigated during the thesis for further training, testing, and comparison.

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

## Publicly Available Implementations

### 1. Constant Velocity Kalman Filter

The updated CV-KF implementation is located in:

```text
CV-KF_code/CV_KF_22JUNE/
```

This method serves as the classical physics-based baseline. It estimates and predicts trajectory motion using a constant-velocity Kalman filter model.

The implementation includes:

* raw drone trajectory data handling
* trajectory preprocessing
* Kalman filter prediction
* evaluation and summary generation
* visualization scripts
* result storage

Main folder:

```text
CV-KF_code/CV_KF_22JUNE/base_CV_KF/
```

Important files include:

* `preprocess_3d.py`
* `kalman_filter_model.py`
* `train.py`
* `testing.py`
* `metrics.py`
* `generate_eval_summary.py`
* `visualization.py`

### 2. LSTM + Mixture Density Network

The updated LSTM+MDN implementations are located in:

```text
AI_MDN_METHOD/LSTM_MDN/LSTM_MDN_01/
AI_MDN_METHOD/LSTM_MDN/LSTM_MDN_02/
```

These versions use different trajectory normalization approaches.

Each version contains a `base_MDN` workflow covering:

* preprocessing
* LSTM sequence modeling
* MDN probabilistic output layer
* training
* testing
* evaluation
* summary generation
* visualization

Important files include:

* `preprocess_3d.py`
* `base_lstm.py`
* `mdn.py`
* `train.py`
* `testing.py`
* `eval.py`
* `generate_eval_summary.py`
* `vis.py`

## Data Normalization

Different normalization approaches are investigated for preparing trajectory data before training the learning-based models.

The two publicly available LSTM+MDN versions represent the current comparison between two normalization methods:

* `LSTM_MDN_01`
* `LSTM_MDN_02`

The purpose is to investigate how trajectory representation and normalization influence prediction accuracy and uncertainty quality.

## Ongoing Thesis Development

The complete thesis implementation is under active development in a private university GitLab repository.

Current research beyond the publicly available GitHub implementation includes:

* GRU + MDN
* TCN + MDN
* Transformer + MDN
* Mamba + MDN
* physics-enhanced residual learning
* comparison of physics-based and learning-based prediction approaches
* probabilistic uncertainty and reliability evaluation

These implementations and experimental results are intentionally not synchronized completely with this public GitHub repository while the thesis is ongoing.

## Evaluation

The prediction methods are evaluated using metrics including:

* Average Displacement Error (ADE)
* Final Displacement Error (FDE)
* probabilistic reliability
* coverage
* sharpness

This allows comparison of both trajectory accuracy and the quality of the predicted uncertainty distributions.

## Current Public Repository Status

Currently available on GitHub:

* drone trajectory data integration
* CV-KF baseline
* LSTM+MDN
* two normalization-based LSTM+MDN variants
* preprocessing, training, testing, and evaluation workflows
* evaluation and visualization utilities

Ongoing thesis work maintained privately includes additional sequence architectures, physics-enhanced prediction approaches, experiments, and intermediate results.

## Expected Outputs

Depending on the selected method, the code can generate:

* processed trajectory files
* predicted future trajectories
* trained model files
* evaluation metrics
* summary tables
* trajectory visualizations
* probabilistic prediction results
* method comparison results

## Purpose

The purpose of this repository is to provide selected implementations from an ongoing investigation into probabilistic motion prediction of flying objects.

The broader objective of the thesis is to compare physics-based, learning-based, and physics-enhanced approaches and determine which methods provide accurate, reliable, and useful future trajectory predictions for collision-free drone control.
