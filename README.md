# Beyond FA Challenge

This project is an attempt to solve the [Beyond FA (BFA) challenge](https://bfa.grand-challenge.org/), a diffusion MRI grand challenge focused on extracting meaningful features to better understand white matter integrity within Alzheimer's Disease (AD) cohorts.

## Background

Fractional anisotropy (FA) is a metric frequently used to interpret white matter integrity, but it exhibits high sensitivity and low specificity in pathological interpretation. The Beyond FA challenge encourages participants to explore alternative white matter models and metrics beyond FA to capture more meaningful biomarkers for white matter integrity analysis, especially in the context of lower-quality clinical data.

## Dataset

This project uses the **MASiVar dataset** ([ds003416](https://github.com/OpenNeuroDatasets/ds003416)), specifically the **PreQual v1.0.0 preprocessed derivatives**. The MASiVar dataset consists of 319 diffusion scans acquired across 14 healthy adults, 83 healthy children (5 to 8 years), three sites, and four scanners.

### Data Splitting

The dataset is split into train, validation, and evaluation sets using age-stratified splitting to ensure equal distribution across sets. The splitting strategy:

- **Age stratification**: Subjects are grouped into age bins (below 10 years and 10+ years) to ensure representation from different age groups in each split
- **Subject-level splitting**: Splits are performed at the subject level to prevent data leakage (multiple sessions from the same subject remain in the same split)
- **Distribution**: The splits maintain balanced age distributions across train/validation/evaluation sets

Example output from the splitting function:
```
Train set: 858 samples from 69 subjects
Validation set: 150 samples from 14 subjects
Evaluation set: 126 samples from 14 subjects

Statistics:
  Age range - Train: 5.4 to 47.0 years (mean: 18.1)
  Age range - Val: 6.3 to 30.0 years (mean: 11.7)
  Age range - Eval: 6.2 to 31.0 years (mean: 9.4)
```

## Feature Extraction

Feature extraction is the core component of this challenge. The pipeline extracts diffusion tensor imaging (DTI) metrics from white matter bundles identified using TractSeg.

### Pipeline Overview

1. **Preprocessing**: Brain mask creation using `dwi2mask` (MRtrix3)
2. **Response function estimation**: White matter response function estimation using `dwi2response`
3. **Fiber orientation distribution (FOD)**: Calculation of FODs using constrained spherical deconvolution (CSD) via `dwi2fod`
4. **Peak extraction**: Extraction of fiber orientation peaks using `sh2peaks`
5. **TractSeg**: White matter bundle segmentation using TractSeg to identify 72 white matter tracts
6. **DTI metric calculation**: Calculation of diffusion tensor metrics using `scil_dti_metrics.py` (scilpy)
7. **Feature aggregation**: Mean metric values are calculated within each bundle ROI using FSL's `fslstats`

### Baseline Implementation

The feature extraction pipeline is based on the [Beyond FA Microstructure Baseline](https://github.com/MASILab/beyond_fa_microstruct_baseline) and has been adapted and extended for this project.

### Extracted Features

The pipeline extracts **Fractional Anisotropy (FA)** which measures the degree of anisotropy in diffusion. The metric is averaged across 72 white matter bundles, resulting in a 128-dimensional feature vector (zero-padded to 128 elements).

### Required Tools

The following tools need to be installed for feature extraction:

- **MRtrix3**: For diffusion MRI processing (`dwi2mask`, `dwi2response`, `dwi2fod`, `sh2peaks`)
- **TractSeg**: For white matter bundle segmentation
- **FSL**: For statistical operations (`fslstats`)
- **scilpy**: For DTI metric calculation (`scil_dti_metrics.py`)
- **Python 3**: With packages: `numpy`, `pandas`, `scikit-learn`

## Evaluation

In the official Beyond FA challenge, the evaluation uses a shallow MLP (Multi-Layer Perceptron) for testing submissions. The MLP is provided by the challenge organizers and is used solely for evaluation purposes—it is not part of the challenge submission itself.

However, in this project, the evaluation model may be adjusted and fine-tuned. The evaluation code has been taken from the [BeyondFA_eval repository](https://github.com/MASILab/BeyondFA_eval/tree/main) and modified to suit this project's needs.

The evaluation framework supports three tasks:
- **Age prediction**: Regression task predicting subject age
- **Sex classification**: Binary classification task
- **Cognitive status classification**: Multi-class classification task

In the results only the Age prediction is being used to evaluate the resulting model.

## Improvements from Baseline

### Additional Features

Beyond the baseline implementation, this project extends the feature extraction pipeline to include additional DTI metrics:

- **Mean Diffusivity (MD)**: Provides complementary information to FA about overall diffusion magnitude
- **Axial Diffusivity (AD)**: Captures diffusion along the primary fiber direction
- **Radial Diffusivity (RD)**: Captures diffusion perpendicular to the primary fiber direction

These additional metrics provide a more comprehensive characterization of white matter microstructure, potentially improving the model's ability to capture meaningful biomarkers beyond what FA alone can provide.

## Project Structure

```
BeyondFA/
├── baseline/              # Baseline feature extraction scripts
├── evaluation/            # Evaluation framework and MLP training code
│   ├── bfa_dev/          # Main evaluation code
│   └── csv/              # Generated train/val/eval CSV files
├── output/               # Extracted feature JSON files
├── tmp/                  # Temporary processing files
├── extract_features.ipynb # Main feature extraction notebook
├── generate_data_csv.py  # Script for generating train/val/eval splits
└── README.md             # This file
```

## Usage

1. **Feature Extraction**: Run `extract_features.ipynb` to extract features from the MASiVar dataset
2. **Generate Splits**: Run `generate_data_csv.py` to create train/validation/evaluation CSV files
3. **Train Model**: Use the evaluation framework in `evaluation/bfa_dev/` to train and evaluate models

## Results

Using the baseline solution with extraction of only FA feature, the results of the best model are:

|            | Loss   | MAE    |
|------------|--------|--------|
| Training   | 7.3218 | 7.3314 |
| Validation | 4.3264 | 4.3264 |
| Evaluation |        | 3.3995 |



## References

- [Beyond FA Challenge](https://bfa.grand-challenge.org/)
- [MASiVar Dataset](https://github.com/OpenNeuroDatasets/ds003416)
- [Beyond FA Microstructure Baseline](https://github.com/MASILab/beyond_fa_microstruct_baseline)
- [BeyondFA Evaluation Framework](https://github.com/MASILab/BeyondFA_eval)

