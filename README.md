# Beyond FA Challenge

This project is an attempt to solve the [Beyond FA (BFA) challenge](https://bfa.grand-challenge.org/), a diffusion MRI grand challenge focused on extracting meaningful features to better understand white matter integrity within Alzheimer's Disease (AD) cohorts. The challenge was loosely followed. The main requirement was to extract a 128 feature set for the task, however, I explored three different approaches:
- Extracting 5 more other features across bundles and concatenating them into one feature set (432 features) for the evaluation model
- Using a 2D CNN model using extracted FA
- Taking the 6 extracted features and training a model where a linear layer projects the 432 into 128 feature set and ends with the evaluation model. The features are then extracted for the final evaluation. The main flaw with this approach is that the datasets for this and the evaluation models are for the same subjects, since there is not enough data to split and have the models be efficient. This approach was taken to try to comply with the challenge requirement.

## Background

Fractional anisotropy (FA) is a metric frequently used to interpret white matter integrity, but it exhibits high sensitivity and low specificity in pathological interpretation. The Beyond FA challenge encourages participants to explore alternative white matter models and metrics beyond FA to capture more meaningful biomarkers for white matter integrity analysis, especially in the context of lower-quality clinical data.

## Dataset

This project uses the **MASiVar dataset** ([ds003416](https://github.com/OpenNeuroDatasets/ds003416)), specifically the **PreQual v1.0.0 preprocessed derivatives**. The MASiVar dataset consists of 319 diffusion scans acquired across 14 healthy adults, 83 healthy children (5 to 8 years), three sites, and four scanners. The scans for each subject were acquired through multiple sessions using different scanners totaling 1134 scans. Some scans were taken using less than 6 directions, they are deemed invalid and the dataset is reduced to 754 images.

### Data Splitting

The dataset is split into train, validation, and evaluation sets using age-stratified splitting to ensure equal distribution across sets. It is split only after the features extraction process. The splitting strategy:

- **Age stratification**: Subjects are grouped into age bins (below 10 years and 10+ years) to ensure representation from different age groups in each split
- **Subject-level splitting**: Splits are performed at the subject level to prevent data leakage (multiple sessions from the same subject remain in the same split)
- **Distribution**: The splits maintain balanced age distributions across train/validation/evaluation sets

Example output from the splitting function:
```
Train set: 590 samples from 69 subjects
Validation set: 84 samples from 14 subjects
Evaluation set: 80 samples from 14 subjects

Statistics:
  Age range - Train: 5.4 to 47.0 years (mean: 18.1)
  Age range - Val: 6.0 to 21.0 years (mean: 7.8)
  Age range - Eval: 5.8 to 31.0 years (mean: 9.5)
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
- **AFD_total (Apparent Fiber Density total)**: Measures the total apparent fiber density derived from the fiber orientation distribution function (FODF), providing information about fiber density across all orientations
- **NuFO (Number of Fiber Orientations)**: Quantifies the number of distinct fiber orientations detected in each voxel, capturing the complexity of crossing fiber configurations

These additional metrics provide a more comprehensive characterization of white matter microstructure, potentially improving the model's ability to capture meaningful biomarkers beyond what FA alone can provide.

### Additional models

Beyond the baseline MLP evaluation model, this project explores alternative architectures:

- **2D CNN Model**: A convolutional neural network that processes 3D FA volumes as 2D slices. The model applies 2D convolutions to each slice independently, then aggregates features across all slices (63 slices) to predict age. This approach leverages spatial information in the FA maps directly from TractSeg output, rather than using aggregated bundle-level features. The architecture consists of four convolutional blocks with batch normalization and max pooling, followed by slice-level feature processing and a final regression head.

- **Feature Reduction Model**: A learned dimensionality reduction approach that takes the full 432-dimensional feature vector (6 metrics × 72 bundles) and reduces it to 128 dimensions using a trainable linear layer with LeakyReLU activation. The reduced features are then passed through the BFANet architecture for age prediction. This model allows the network to learn an optimal feature representation while maintaining compatibility with the standard 128-feature format. After training, the reduction layer can be used to extract reduced features from all 432-feature inputs for downstream analysis.

## Project Structure

```
BeyondFA/
├── baseline/                      # Baseline feature extraction scripts
│   ├── convert_json_to_bvalbvec.py
│   ├── convert_mha_to_nifti.py
│   ├── extract_metric.py
│   └── run_metric.sh
├── evaluation/                   # Evaluation framework and MLP training code
│   ├── bfa_dev/                  # Main evaluation code
│   │   ├── config.yaml          # Configuration file
│   │   ├── train.py             # Training script
│   │   ├── data/                # Dataset utilities
│   │   ├── models/              # Model definitions
│   │   └── utils/               # Training utilities
│   ├── convert_nifti_to_mha.py
│   └── generate_json.py
├── utils/                        # Utility modules for feature extraction
│   ├── dwi_data.py              # DWI data handling
│   ├── fodf.py                  # FOD (Fiber Orientation Distribution) utilities
│   ├── fslstats.py              # FSL statistics wrapper
│   ├── mrtrix.py                # MRtrix3 command wrappers
│   ├── run_cmd.py               # Command execution utilities
│   ├── scilpy.py                # scilpy command wrappers
│   └── tractseg.py              # TractSeg utilities
├── baseline_extract_features.ipynb  # Baseline feature extraction notebook
├── extract_features.ipynb        # Main feature extraction notebook
├── find_downloaded.py            # Script to find downloaded subjects
├── generate_data_csv.py          # Script for generating train/val/eval splits
├── plot_brain_scan.py           # Visualization utilities
├── run_tractseg.sh              # TractSeg execution script
├── train_2d_cnn.py              # 2D CNN training script
├── train_to_reduce_features.py  # Feature reduction training script
├── unify_scan_sizes.py          # Script to unify scan sizes
└── README.md                     # This file
```

## Usage

### Baseline and First Approach
1. **Feature Extraction**: Run `extract_features.ipynb` to extract features from the MASiVar dataset
2. **Generate Splits**: Run `generate_data_csv.py` to create train/validation/evaluation CSV files
3. **Train Model**: Use the evaluation framework in `evaluation/bfa_dev/` to train and evaluate models

### Second Approach
1. **Feature Extraction**: Run `extract_features.ipynb` to extract features from the MASiVar dataset
2. **Unify Scan Sizes**: Run `unify_scan_sizes.py` to standardize scan dimensions by centering brains and cropping/padding to a common size.
3. **Train 2D CNN**: Run `train_2d_cnn.py` to train the 2D CNN model on the unified FA scans. The script automatically splits data by subject with age-stratified splitting, normalizes the data, and trains the 2D CNN for age prediction. Model checkpoints and results are saved to the specified output directory.

### Third Approach
1. **Feature Extraction**: Run `extract_features.ipynb` to extract features from the MASiVar dataset
2. **Generate Splits**: Run `generate_data_csv.py` to create train/validation/evaluation CSV files
3. **Train Feature Reduction Model**: Run `train_to_reduce_features.py` to train and evaluate feature reduction model and eventually run it on the whole dataset to reduce the feature set.
4. **Generate Splits**: Run `generate_data_csv.py` to create train/validation/evaluation CSV files from the reduced data
5. **Train Model**: Use the evaluation framework in `evaluation/bfa_dev/` to train and evaluate models

## Results

### Baseline
Using the baseline solution with extraction of only FA feature, the results of the best model are:
|            | Loss    | MAE    |
|------------|---------|--------|
| Training   | 11.6099 | 1.1898 |
| Validation | 11.4805 | 1.1898 |
| Evaluation |         | 2.5183 |

### First Approach
Using all extracted features, the results are:
|            | Loss    | MAE    |
|------------|---------|--------|
| Training   | 11.6099 | 1.1898 |
| Validation | 11.4805 | 1.1898 |
| Evaluation |         | 2.5183 |

### Second Approach
Results after training 2D CNN using the FA feature:
|            | Loss    | MAE    |
|------------|---------|--------|
| Training   | 12.8195 | 2.4347 |
| Validation | 63.4710 | 4.8772 |
| Evaluation |         | 2.4050 |

## References

- [Beyond FA Challenge](https://bfa.grand-challenge.org/)
- [MASiVar Dataset](https://github.com/OpenNeuroDatasets/ds003416)
- [Beyond FA Microstructure Baseline](https://github.com/MASILab/beyond_fa_microstruct_baseline)
- [BeyondFA Evaluation Framework](https://github.com/MASILab/BeyondFA_eval)

