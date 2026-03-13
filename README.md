# Harmful Brain Activity Classification based on EEG Signals

This repository contains the notebooks, utilities, saved artifacts, and ensemble outputs used for the ECE247A final project on classifying harmful brain activity from EEG recordings. The work is built around the Kaggle HMS Harmful Brain Activity Classification dataset and explores multiple representation/model choices rather than a single packaged training pipeline.

## Project Scope

The repo includes experiments across several EEG modeling approaches:

- Baseline spectrogram classification with KerasCV / EfficientNetV2
- Spectrogram CNN models and CNN + Transformer variants
- Scalogram models built from wavelet transforms
- Raw EEG modeling with Conv1D + BiGRU + attention
- Vision-style spectrogram modeling with Swin Transformer
- PyTorch spectrogram + GRU pipelines with nnAudio
- Submission ensembling and KL-divergence-based comparison

## Repository Layout

| Path | Contents |
| --- | --- |
| `baselines/` | Starter KerasCV baseline notebook, baseline requirements, and baseline submission artifacts |
| `spectrogram_scalogram/` | Spectrogram CNN, scalogram, and spectrogram-GRU notebooks |
| `cnn-bigru_and_vit/` | Raw-EEG Conv1D+BiGRU pipeline, Swin-T spectrogram pipeline, utility modules, diagrams, saved results, and model checkpoints |
| `spectrogram-bigru/AL_model/` | PyTorch nnAudio + EfficientNet + GRU training notebook, requirements, saved predictions, and model weights |
| `ensemble/` | Notebook and CSVs used to average predictions and compare ensemble performance |

## Main Notebooks

- `baselines/hms-hbac-kerascv-starter-notebook_test.ipynb`
  - KerasCV starter baseline using spectrogram inputs and EfficientNetV2.
- `spectrogram_scalogram/spectrogram_cnn.ipynb`
  - Spectrogram-based CNN experiments, including an EfficientNet-style backbone and a CNN + Transformer variant.
- `spectrogram_scalogram/scalogram_w_transformer.ipynb`
  - Continuous-wavelet-transform scalogram pipeline with stronger augmentation and a Transformer-based model head.
- `spectrogram_scalogram/scalogram_w_out_transformer.ipynb`
  - Scalogram pipeline variant without the Transformer component.
- `spectrogram_scalogram/pytorch_spectrogram_gru_AL.ipynb`
  - PyTorch spectrogram pipeline using high-consensus training data and GRU-based temporal modeling.
- `cnn-bigru_and_vit/conv1d_bigru_raw_eeg_v1.ipynb`
  - Raw EEG classifier with bipolar montage features, filtering/downsampling, Conv1D layers, BiGRU, and attention pooling.
- `cnn-bigru_and_vit/vit_v1.ipynb`
  - Swin-T spectrogram vision pipeline built on Kaggle-provided spectrogram windows.
- `cnn-bigru_and_vit/architecture_diagrams.ipynb`
  - Utility notebook for regenerating architecture figures used in the report.
- `spectrogram-bigru/AL_model/pytorch_nnaudio_gru_pipeline_v14_usethis.ipynb`
  - Main nnAudio + EfficientNet + GRU notebook used for AL's PyTorch pipeline.
- `ensemble/ensemble_eval.ipynb`
  - Loads multiple submission CSVs, averages predictions, and evaluates KL divergence against a held-out confident set.

## Utilities and Saved Artifacts

- `cnn-bigru_and_vit/data_aug.py`
  - Shared augmentation helpers for raw EEG pipelines, including noise, scaling, time shift, left-right flips, MixUp, masking, and channel dropout.
- `cnn-bigru_and_vit/conv1d_bigru_utils.py`
  - Runtime setup, Kaggle download helpers, montage construction, caching, dataset logic, training loops, and artifact export for the raw EEG pipeline.
- `cnn-bigru_and_vit/vit_utils.py`
  - Shared code for spectrogram image generation, cached datasets, and Swin-T training.
- `cnn-bigru_and_vit/results/`, `spectrogram-bigru/AL_model/results/`, `ensemble/*.csv`
  - Example training histories, diagnostic plots, predictions, and ensemble outputs.

## Data Requirements

These notebooks are built around the Kaggle HMS Harmful Brain Activity Classification competition data. In practice, most notebooks expect some combination of:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `train_eegs/`
- `test_eegs/`
- `train_spectrograms/`
- `test_spectrograms/`

Several notebooks also expect prefiltered high-consensus splits such as:

- `confident_train.csv`
- `confident_test.csv`

Those files are referenced by the notebooks but are not stored in this repository.

## Environment Notes

There is not one fully unified environment for the entire repo. Different experiment tracks use different stacks:

- `baselines/requirements_starter.txt`
  - TensorFlow, Keras, KerasCV, librosa, PyWavelets
- `spectrogram-bigru/AL_model/requirements_AL.txt`
  - PyTorch, timm, nnAudio, pandas, pyarrow, scikit-learn
- `cnn-bigru_and_vit/`
  - PyTorch-based utilities that also rely on `timm`, `scipy`, `matplotlib`, `pandas`, `numpy`, `scikit-learn`, and Kaggle data access

Many notebooks were developed in Google Colab and still contain Colab- or Drive-specific paths. Expect to update path configuration before running locally.

## Typical Setup

1. Accept the Kaggle competition rules for HMS.
2. Place a valid `kaggle.json` on the machine or configure Kaggle credentials in the environment.
3. Create the environment that matches the notebook family you want to run.
4. Download the dataset and point notebook paths to your local data directory.
5. If you are running AL or some spectrogram/scalogram notebooks, also provide or regenerate `confident_train.csv` and `confident_test.csv`.

Example environment setup for the two pinned requirements files:

```bash
pip install -r baselines/requirements_starter.txt
pip install -r spectrogram-bigru/AL_model/requirements_AL.txt
```

## Reproducing Results

This repository is best treated as an experiment archive. To reproduce a given result:

1. Pick the relevant notebook from the sections above.
2. Match its expected framework stack and runtime environment.
3. Update any hard-coded paths for Kaggle data, Google Drive, checkpoints, and output directories.
4. Run the notebook cells in order.
5. Use `ensemble/ensemble_eval.ipynb` to compare or average prediction CSVs across models.

## Notes

- The repo mixes TensorFlow/Keras and PyTorch workflows.
- Some model checkpoints and result files are already included for reference.
