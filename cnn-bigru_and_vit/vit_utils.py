"""Utility module for the ViT/Swin-T spectrogram EEG classification pipeline.

Imports shared infrastructure from conv1d_bigru_utils; do not duplicate
any function or constant defined there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from conv1d_bigru_utils import (
    BIPOLAR_PAIRS,
    CHAIN_ORDER,
    build_confident_csvs,
    compute_global_prior,
    compute_prior_baseline_kl,
    ensure_dirs,
    kl_divergence_np,
    plot_history,
    prepare_cache,
    save_live_training_artifacts,
    save_run_config_artifact,
    set_seed,
    setup_runtime,
)

# Chain index slices in the 16-channel bipolar array (chain order: LL, RL, LP, RP).
CHAIN_INDICES: Dict[str, List[int]] = {
    'LL': [0, 1, 2, 3],
    'RL': [4, 5, 6, 7],
    'LP': [8, 9, 10, 11],
    'RP': [12, 13, 14, 15],
}

# Kaggle-provided spectrogram constants.
# Each parquet has 400 data columns: 100 freq bins per chain (LL, RL, LP, RP).
# Column names are like 'LL_0.59', 'LL_0.78', ..., 'RL_0.59', ..., 'RP_19.92'.
# Rows are 2-second time steps.  A 50-s label window maps to 300 time columns
# (at 2s/step, offset_seconds // 2 gives the start column index).
_PROVIDED_SPEC_CHAINS: Tuple[str, ...] = ('LL', 'RL', 'LP', 'RP')
_PROVIDED_SPEC_FREQS_PER_CHAIN: int = 100
_PROVIDED_SPEC_WINDOW_COLS: int = 300   # columns in one 50-s window


# ---------------------------------------------------------------------------
# Kaggle-provided spectrogram utilities
# ---------------------------------------------------------------------------

def read_provided_spectrogram(
    spec_path: Path,
    offset_seconds: float,
) -> np.ndarray:
    """Read and window a Kaggle-provided spectrogram parquet file.

    Parameters
    ----------
    spec_path : pathlib.Path
        Path to a ``train_spectrograms/{id}.parquet`` or
        ``test_spectrograms/{id}.parquet`` file.
    offset_seconds : float
        ``spectrogram_label_offset_seconds`` from the metadata CSV.
        Determines which 50-second window to extract.

    Returns
    -------
    numpy.ndarray
        Chain-separated power array of shape ``(4, 100, 300)``, dtype float32.
        Axis 0 indexes chains in order LL, RL, LP, RP.
        Axis 1 indexes frequency bins (0.59 Hz low to 19.92 Hz high).
        Axis 2 indexes time steps within the 50-second window.

    Notes
    -----
    The parquet ``time`` column is dropped.  Each remaining column is one
    (chain, frequency) pair.  After transposing the frame to
    ``(400, n_time_steps)``, the 300-column label window is extracted starting
    at ``int(offset_seconds) // 2``.  Columns that fall outside the recording
    are zero-padded on the right.
    """
    spec_path = Path(spec_path)
    df_spec = pd.read_parquet(spec_path)

    # Drop the time index column; result shape: (n_time, 400)
    values = df_spec.drop(columns=['time'], errors='ignore').values.astype(np.float32)
    # Transpose to (400, n_time)
    values = values.T

    n_freq, n_time = values.shape
    start = int(offset_seconds) // 2
    end = start + _PROVIDED_SPEC_WINDOW_COLS

    if end <= n_time:
        window = values[:, start:end]
    else:
        # Zero-pad the right edge for windows near the end of the recording.
        avail = max(0, n_time - start)
        window = np.zeros((n_freq, _PROVIDED_SPEC_WINDOW_COLS), dtype=np.float32)
        if avail > 0:
            window[:, :avail] = values[:, start:start + avail]

    # Split by chain. Column order in the parquet is LL×100, RL×100, LP×100, RP×100.
    chains = np.stack([
        window[i * _PROVIDED_SPEC_FREQS_PER_CHAIN:(i + 1) * _PROVIDED_SPEC_FREQS_PER_CHAIN, :]
        for i in range(len(_PROVIDED_SPEC_CHAINS))
    ], axis=0)   # (4, 100, 300)
    return chains


def provided_spec_to_image(
    spec_path: Path,
    offset_seconds: float,
    cfg: 'SpectrogramVisionCFG',
) -> np.ndarray:
    """Convert a Kaggle-provided spectrogram window to a 3-channel model input.

    Parameters
    ----------
    spec_path : pathlib.Path
        Path to the spectrogram parquet file.
    offset_seconds : float
        ``spectrogram_label_offset_seconds`` for this label window.
    cfg : SpectrogramVisionCFG
        Pipeline configuration; uses ``cfg.img_size``.

    Returns
    -------
    numpy.ndarray
        Float32 image of shape ``(3, cfg.img_size, cfg.img_size)``,
        values normalised to approximately ``[-3, 3]`` (z-score in log space).

    Notes
    -----
    Processing steps:

    1. Read chains via :func:`read_provided_spectrogram` → ``(4, 100, 300)``.
    2. Assemble a 2x2 spatial montage::

           [[LL (top-left),  RL (top-right)],
            [LP (bottom-left), RP (bottom-right)]]

       producing a ``(200, 600)`` power image.
    3. Clip power to ``[exp(-4), exp(8)]`` to prevent log(0).
    4. Apply natural log: ``log(clipped)``.
    5. Z-score normalise: subtract mean, divide by std (+ 1e-6).
    6. Tile the single log-spectrogram channel to 3 identical RGB channels
       so ImageNet-pretrained weights can be used without modification.
    7. Bilinear-resize to ``(3, cfg.img_size, cfg.img_size)``.
    """
    chains = read_provided_spectrogram(spec_path, offset_seconds)  # (4, 100, 300)

    ll, rl, lp, rp = chains[0], chains[1], chains[2], chains[3]

    # 2x2 montage: top row = LL|RL, bottom row = LP|RP.
    top = np.concatenate([ll, rl], axis=1)   # (100, 600)
    bot = np.concatenate([lp, rp], axis=1)   # (100, 600)
    montage = np.concatenate([top, bot], axis=0).astype(np.float32)  # (200, 600)

    # Kaggle parquets contain NaN for sparse frequency bins.
    # Replace with 0.0 so clip maps them to the safe floor exp(-4).
    np.nan_to_num(montage, nan=0.0, copy=False)

    # clip → log → z-score  (matches starter notebook normalisation).
    montage = np.clip(montage, np.exp(-4.0), np.exp(8.0))
    montage = np.log(montage)
    montage = (montage - montage.mean()) / (montage.std() + 1e-6)

    # Tile to 3 identical channels: (3, 200, 600).
    image = np.stack([montage, montage, montage], axis=0)

    # Bilinear resize to (3, img_size, img_size).
    t = torch.from_numpy(image).unsqueeze(0).float()
    t = F.interpolate(t, size=(cfg.img_size, cfg.img_size), mode='bilinear', align_corners=False)
    return t.squeeze(0).numpy()


def _kaggle_spec_cache_key(row: 'pd.Series') -> str:
    """Derive a stable filesystem-safe cache key from a metadata row.

    Parameters
    ----------
    row : pandas.Series
        One row from a metadata CSV (e.g. ``confident_train.csv``).
        Must contain either ``label_id`` or both ``spectrogram_id`` and
        ``spectrogram_label_offset_seconds``.

    Returns
    -------
    str
        Cache key string suitable for use as a filename stem.
    """
    if 'label_id' in row.index and not pd.isna(row['label_id']):
        return f"label_{int(row['label_id'])}"
    off = int(round(float(row.get('spectrogram_label_offset_seconds', 0))))
    return f"spec_{int(row['spectrogram_id'])}_off_{off}"


def precompute_kaggle_spec_cache(
    df: 'pd.DataFrame',
    output_dir: Path,
    cfg: 'SpectrogramVisionCFG',
) -> List[Path]:
    """Pre-generate and cache spectrogram images from Kaggle-provided parquets.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata frame.  Must contain columns ``spec_path`` (path to the
        ``.parquet`` file) and ``spectrogram_label_offset_seconds``.
    output_dir : pathlib.Path
        Destination directory for ``.npy`` image files.  Created if absent.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.

    Returns
    -------
    list of pathlib.Path
        Cache file path for each row in ``df``, in row order.
        Already-existing files are skipped (idempotent).

    Notes
    -----
    Each ``.npy`` file stores a float32 array of shape
    ``(3, cfg.img_size, cfg.img_size)``.  The filename stem is derived from
    :func:`_kaggle_spec_cache_key`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_paths: List[Path] = []
    created = skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Caching specs → {output_dir.name}'):
        key = _kaggle_spec_cache_key(row)
        out_path = output_dir / f'{key}.npy'
        cache_paths.append(out_path)

        if out_path.exists():
            skipped += 1
            continue

        image = provided_spec_to_image(row['spec_path'], row['spectrogram_label_offset_seconds'], cfg)
        np.save(out_path, image.astype(np.float32))
        created += 1

    print(f'Spectrogram cache: created={created}, skipped={skipped}, dir={output_dir}')
    return cache_paths


# ---------------------------------------------------------------------------
# Dataset — Kaggle provided spectrograms
# ---------------------------------------------------------------------------

class KaggleSpectrogramDataset(Dataset):
    """Dataset backed by Kaggle-provided spectrogram parquet files.

    Reads pre-cached ``.npy`` images when available; falls back to computing
    from the parquet on-the-fly.  Applies the same SpecAugment-style
    augmentations as :class:`SpectrogramImageDataset` during training.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata frame with one row per sample.  Test rows do not need
        ``soft_labels`` or ``total_votes`` columns.
    split : str
        One of ``'train'``, ``'valid'``, or ``'test'``.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.
    augment : bool, optional
        Apply SpecAugment-style augmentations (default ``False``).
    precomputed_dir : pathlib.Path or None, optional
        Directory of pre-generated ``.npy`` files produced by
        :func:`precompute_kaggle_spec_cache`.  If ``None`` (or if a file is
        absent), the image is computed on-the-fly from ``row['spec_path']``.

    Attributes
    ----------
    df : pandas.DataFrame
    split : str
    cfg : SpectrogramVisionCFG
    augment : bool
    precomputed_dir : pathlib.Path or None
    cache_paths : list of pathlib.Path or None
        Resolved per-row ``.npy`` paths when ``precomputed_dir`` is given.
    """

    def __init__(
        self,
        df: 'pd.DataFrame',
        split: str,
        cfg: 'SpectrogramVisionCFG',
        augment: bool = False,
        precomputed_dir: Optional[Path] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.split = split
        self.cfg = cfg
        self.augment = augment
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir is not None else None

        if self.precomputed_dir is not None:
            self.cache_paths: Optional[List[Path]] = [
                self.precomputed_dir / f'{_kaggle_spec_cache_key(row)}.npy'
                for _, row in self.df.iterrows()
            ]
        else:
            self.cache_paths = None

    def __len__(self) -> int:
        """Return the number of samples.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple of torch.Tensor
            ``(image, soft_label, total_votes)`` where ``image`` has shape
            ``(3, img_size, img_size)``.  For the test split, ``soft_label``
            and ``total_votes`` are zero tensors.
        """
        row = self.df.iloc[idx]

        # Load image from cache or compute on-the-fly.
        if self.cache_paths is not None:
            npy_path = self.cache_paths[idx]
            if npy_path.exists():
                image = np.load(npy_path)
            else:
                image = provided_spec_to_image(row['spec_path'], row['spectrogram_label_offset_seconds'], self.cfg)
        else:
            image = provided_spec_to_image(row['spec_path'], row['spectrogram_label_offset_seconds'], self.cfg)

        x = torch.from_numpy(image).float()

        if self.augment:
            x = self._augment(x)

        if self.split == 'test':
            return x, torch.zeros(self.cfg.num_classes), torch.tensor(0.0)

        y = torch.from_numpy(np.array(row['soft_labels'], dtype=np.float32))
        votes = torch.tensor(float(row['total_votes']))
        return x, y, votes

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment-style image augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape ``(3, H, W)``.

        Returns
        -------
        torch.Tensor
            Augmented image tensor of the same shape.

        Notes
        -----
        Three independent transforms, each with probability 0.5:

        - Time masking: zero a contiguous 10 % strip along the time axis.
        - Frequency masking: zero a contiguous 10 % strip along the freq axis.
        - Horizontal flip: reverse the time axis.
        """
        _, H, W = x.shape

        if torch.rand(1).item() < 0.5:
            mask_len = max(1, int(W * 0.10))
            start = torch.randint(0, max(W - mask_len, 1), (1,)).item()
            x = x.clone()
            x[:, :, start: start + mask_len] = 0.0

        if torch.rand(1).item() < 0.5:
            mask_len = max(1, int(H * 0.10))
            start = torch.randint(0, max(H - mask_len, 1), (1,)).item()
            x = x.clone()
            x[:, start: start + mask_len, :] = 0.0

        if torch.rand(1).item() < 0.5:
            x = x.flip(dims=[2])

        return x


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SpectrogramVisionCFG:
    """Configuration for the Swin-T spectrogram vision pipeline.

    Attributes
    ----------
    BASE_PATH : pathlib.Path
        Root directory containing HMS competition data.
    WORK_DIR : pathlib.Path
        Project working directory.
    CACHE_DIR : pathlib.Path
        Base directory for cached EEG NPZ files (VIT-specific subtree).
    MODELS_DIR : pathlib.Path
        Directory for saved model checkpoints.
    RESULTS_DIR : pathlib.Path
        Directory for training artifacts (plots, CSVs, JSON configs).
    PLOTS_DIR : pathlib.Path
        Directory for standalone plot outputs.
    backbone_name : str
        timm model identifier.
    img_size : int
        Square image size expected by the backbone.
    n_fft : int
        FFT window size for per-chain STFT (applied at target_sample_rate).
    hop_length : int
        STFT hop length in samples.
    freq_crop_hz : float
        Maximum frequency (Hz) retained after STFT crop.
    target_sample_rate : int
        EEG sampling rate after resampling (Hz).
    window_seconds : int
        EEG window duration in seconds.
    src_sample_rate : int
        Raw EEG sampling rate before resampling (Hz).
    num_bipolar_channels : int
        Number of bipolar EEG channels.
    bandpass_low_hz : float
        Bandpass lower cutoff (Hz).
    bandpass_high_hz : float
        Bandpass upper cutoff (Hz).
    bandpass_order : int
        Butterworth filter order.
    apply_notch : bool
        Whether to apply a 60 Hz notch filter.
    notch_freq_hz : float
        Notch filter center frequency (Hz).
    notch_q : float
        Notch filter quality factor.
    batch_size : int
        Mini-batch size.
    num_epochs_warmup : int
        Epochs for Stage 1 (frozen backbone, head-only warmup).
    num_epochs_partial : int
        Epochs for Stage 2 (last 2 Swin stages unfrozen).
    num_epochs_full : int
        Epochs for Stage 3 (full unfreeze). Set to 0 to skip (sanity mode).
    lr : float
        Base learning rate for trainable backbone parameters.
    head_lr : float
        Learning rate for the classification head during warmup.
    dropout : float
        Dropout probability in the classification head.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Gradient norm clip value.
    use_amp : bool
        Enable automatic mixed precision (fp16).
    early_stopping_patience : int
        Patience epochs for early stopping in Stages 2 and 3.
    pin_memory : bool
        Pin DataLoader memory for faster GPU transfers.
    fold : int
        Active cross-validation fold index.
    num_classes : int
        Number of output classes.
    num_workers : int
        DataLoader worker processes.
    seed : int
        Global random seed.
    force_rebuild_cache : bool
        If True, rebuild EEG NPZ caches even when files already exist.
    sanity_mode : bool
        When True, override epoch counts and sample sizes for fast testing.
    sanity_train_samples : int
        Max training samples used when sanity_mode is True.
    sanity_valid_samples : int
        Max validation samples used when sanity_mode is True.
    sanity_epochs_warmup : int
        Warmup epoch count in sanity mode.
    sanity_epochs_partial : int
        Partial-unfreeze epoch count in sanity mode.
    sanity_epochs_full : int
        Full-unfreeze epoch count in sanity mode (0 skips Stage 3).
    """

    # Paths — populated by setup_runtime() in the notebook.
    BASE_PATH: Path = Path('/home/littl/data/data')
    WORK_DIR: Path = Path('/home/littl/ECE247A_Final_Project/JZ')
    CACHE_DIR: Path = Path('/home/littl/ECE247A_Final_Project/JZ/cache/vit')
    MODELS_DIR: Path = Path('/home/littl/ECE247A_Final_Project/JZ/models')
    RESULTS_DIR: Path = Path('/home/littl/ECE247A_Final_Project/JZ/results')
    PLOTS_DIR: Path = Path('/home/littl/ECE247A_Final_Project/JZ/results/plots')

    # Backbone.
    backbone_name: str = 'swin_tiny_patch4_window7_224'
    img_size: int = 224

    # STFT parameters (applied at target_sample_rate = 100 Hz).
    n_fft: int = 128
    hop_length: int = 4
    freq_crop_hz: float = 25.0
    target_sample_rate: int = 100

    # EEG preprocessing attributes required by prepare_cache / preprocess_row_to_array.
    window_seconds: int = 50
    src_sample_rate: int = 200
    num_bipolar_channels: int = 16
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 49.9
    bandpass_order: int = 4
    apply_notch: bool = True
    notch_freq_hz: float = 60.0
    notch_q: float = 30.0

    # Training hyperparameters.
    batch_size: int = 64
    num_epochs_warmup: int = 3
    num_epochs_partial: int = 25
    num_epochs_full: int = 8
    lr: float = 2e-4
    head_lr: float = 1e-3
    partial_unfreeze_stages: Tuple[int, ...] = (2, 3)
    dropout: float = 0.3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 5
    pin_memory: bool = True
    mixup_alpha: float = 0.4
    cosine_t_max: int = 0  # 0 = use num_epochs_partial (default); set to shorter value to decay LR faster

    # Run control.
    fold: int = 0
    num_classes: int = 6
    num_workers: int = 2
    seed: int = 42
    force_rebuild_cache: bool = False
    use_compile: bool = False

    # Sanity mode — overrides for fast pipeline verification.
    sanity_mode: bool = False
    sanity_train_samples: int = 500
    sanity_valid_samples: int = 150
    sanity_epochs_warmup: int = 1
    sanity_epochs_partial: int = 2
    sanity_epochs_full: int = 0


# ---------------------------------------------------------------------------
# Spectrogram generation
# ---------------------------------------------------------------------------

def eeg_to_spectrogram_image(
    eeg: np.ndarray,
    cfg: SpectrogramVisionCFG,
) -> np.ndarray:
    """Convert a 16-channel bipolar EEG array to a 3-channel spectrogram image.

    Parameters
    ----------
    eeg : numpy.ndarray
        Bipolar EEG of shape ``(16, n_samples)`` at ``cfg.target_sample_rate``.
        Channel order: LL[0:4], RL[4:8], LP[8:12], RP[12:16].
    cfg : SpectrogramVisionCFG
        Pipeline configuration.

    Returns
    -------
    numpy.ndarray
        Spectrogram image of shape ``(3, cfg.img_size, cfg.img_size)``,
        dtype float32, values in ``[0, 1]``.

    Notes
    -----
    RGB channels encode:
      R : log1p(power)
      G : temporal gradient of log1p(power)
      B : mean-subtracted log1p(power)

    A 2x2 montage is assembled as::

        [[LL, RL],
         [LP, RP]]

    before building the 3-channel representation.
    """
    fs = cfg.target_sample_rate
    freq_max_bin = int(cfg.freq_crop_hz * cfg.n_fft / fs) + 1

    panels: List[np.ndarray] = []
    for chain in ['LL', 'RL', 'LP', 'RP']:
        ch_idx = CHAIN_INDICES[chain]
        # Average the 4 bipolar pairs in this chain.
        chain_signal = eeg[ch_idx].mean(axis=0).astype(np.float64)

        _, _, Zxx = stft(
            chain_signal,
            fs=fs,
            nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length,
            nfft=cfg.n_fft,
        )
        power = np.abs(Zxx[:freq_max_bin, :]) ** 2
        panels.append(power)

    # Assemble 2x2 montage: [[LL, RL], [LP, RP]].
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bot = np.concatenate([panels[2], panels[3]], axis=1)
    montage = np.concatenate([top, bot], axis=0)      # (2*freq_bins, 2*T)

    log_power = np.log1p(montage)

    r_channel = log_power
    g_channel = np.gradient(log_power, axis=1)
    b_channel = log_power - log_power.mean(axis=1, keepdims=True)

    image = np.stack([r_channel, g_channel, b_channel], axis=0).astype(np.float32)

    # Per-channel normalize to [0, 1].
    for c in range(3):
        cmin, cmax = float(image[c].min()), float(image[c].max())
        if cmax - cmin > 1e-8:
            image[c] = (image[c] - cmin) / (cmax - cmin)
        else:
            image[c] = 0.0

    # Bilinear resize to (3, img_size, img_size).
    t = torch.from_numpy(image).unsqueeze(0).float()
    t = F.interpolate(t, size=(cfg.img_size, cfg.img_size), mode='bilinear', align_corners=False)
    return t.squeeze(0).numpy()


def precompute_spectrogram_cache(
    cache_files: List[Path],
    output_dir: Path,
    cfg: SpectrogramVisionCFG,
) -> None:
    """Pre-generate spectrogram images from EEG NPZ files and save as .npy.

    Parameters
    ----------
    cache_files : list of pathlib.Path
        Paths to preprocessed EEG ``.npz`` cache files.  Each file must
        contain an array stored under the key ``'x'``.
    output_dir : pathlib.Path
        Destination directory for ``.npy`` spectrogram image files.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.

    Returns
    -------
    None
        Writes ``.npy`` files to ``output_dir``.  Already-existing files are
        skipped (idempotent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    created = 0
    for fp in tqdm(cache_files, desc='Precomputing spectrograms'):
        out_path = output_dir / (fp.stem + '.npy')
        if out_path.exists():
            skipped += 1
            continue
        data = np.load(fp)
        image = eeg_to_spectrogram_image(data['x'], cfg)
        np.save(out_path, image.astype(np.float32))
        created += 1
    print(f'Spectrogram cache: created={created}, skipped={skipped}, dir={output_dir}')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpectrogramImageDataset(Dataset):
    """Dataset that loads bipolar EEG NPZ caches and converts to spectrogram images.

    Parameters
    ----------
    cache_files : list of pathlib.Path
        Paths to ``.npz`` EEG cache files produced by ``prepare_cache``.
    split : str
        One of ``'train'``, ``'valid'``, or ``'test'``.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.
    augment : bool, optional
        If True, apply SpecAugment-style image augmentation (default False).
    precomputed_dir : pathlib.Path or None, optional
        Directory of pre-generated ``.npy`` spectrogram images.  If given and
        the matching file exists, it is loaded directly; otherwise the
        spectrogram is computed on-the-fly from the NPZ.

    Attributes
    ----------
    cache_files : list of pathlib.Path
    split : str
    cfg : SpectrogramVisionCFG
    augment : bool
    precomputed_dir : pathlib.Path or None
    """

    def __init__(
        self,
        cache_files: List[Path],
        split: str,
        cfg: SpectrogramVisionCFG,
        augment: bool = False,
        precomputed_dir: Optional[Path] = None,
    ) -> None:
        self.cache_files = [Path(p) for p in cache_files]
        self.split = split
        self.cfg = cfg
        self.augment = augment
        self.precomputed_dir = precomputed_dir

    def __len__(self) -> int:
        """Return number of samples.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.cache_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple of torch.Tensor
            ``(image, soft_label, total_votes)`` where ``image`` has shape
            ``(3, img_size, img_size)``.  For the test split, ``soft_label``
            and ``total_votes`` are zero tensors.
        """
        fp = self.cache_files[idx]
        data = np.load(fp)

        # Resolve spectrogram image.
        if self.precomputed_dir is not None:
            img_path = self.precomputed_dir / (fp.stem + '.npy')
            if img_path.exists():
                image = np.load(img_path)
            else:
                image = eeg_to_spectrogram_image(data['x'], self.cfg)
        else:
            image = eeg_to_spectrogram_image(data['x'], self.cfg)

        x = torch.from_numpy(image).float()

        if self.augment:
            x = self._augment(x)

        if self.split == 'test':
            return x, torch.zeros(self.cfg.num_classes), torch.tensor(0.0)

        y = torch.from_numpy(data['y'].astype(np.float32))
        votes = torch.tensor(float(data['votes']))
        return x, y, votes

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment-style image augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape ``(3, H, W)``.

        Returns
        -------
        torch.Tensor
            Augmented image tensor of the same shape.

        Notes
        -----
        Three independent transforms, each with probability 0.5:
        - Time masking: zero a contiguous 10% strip along the time axis.
        - Frequency masking: zero a contiguous 10% strip along the freq axis.
        - Horizontal flip: reverse the time axis (left-right).
        """
        _, H, W = x.shape

        # Time masking (width dimension).
        if torch.rand(1).item() < 0.5:
            mask_len = max(1, int(W * 0.10))
            start = torch.randint(0, max(W - mask_len, 1), (1,)).item()
            x = x.clone()
            x[:, :, start: start + mask_len] = 0.0

        # Frequency masking (height dimension).
        if torch.rand(1).item() < 0.5:
            mask_len = max(1, int(H * 0.10))
            start = torch.randint(0, max(H - mask_len, 1), (1,)).item()
            x = x.clone()
            x[:, start: start + mask_len, :] = 0.0

        # Time reversal (horizontal flip).
        if torch.rand(1).item() < 0.5:
            x = x.flip(dims=[2])

        return x


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_vit_dataloaders(
    df_train,
    df_valid,
    cfg: SpectrogramVisionCFG,
) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders for the ViT spectrogram pipeline.

    Uses Kaggle-provided spectrogram parquets (``train_spectrograms/``) as the
    image source.  EEG NPZ files are **not** required.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training split metadata.  Must contain ``spec_path``,
        ``spectrogram_label_offset_seconds``, ``soft_labels``,
        ``total_votes``.
    df_valid : pandas.DataFrame
        Validation split metadata (same schema as ``df_train``).
    cfg : SpectrogramVisionCFG
        Pipeline configuration.  ``cfg.CACHE_DIR`` must point to a VIT-specific
        directory (e.g. ``cache/vit/``) so caches remain independent from the
        conv1d_bigru pipeline.

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        ``(train_loader, valid_loader)``.

    Notes
    -----
    Spectrogram image ``.npy`` files are written to
    ``cfg.CACHE_DIR / 'spectrograms' / 'train_fold{N}/'`` and the matching
    valid path.  Existing files are skipped (idempotent).  On subsequent runs
    the caching step is near-instant.
    """
    spec_train_dir = cfg.CACHE_DIR / 'spectrograms' / f'train_fold{cfg.fold}'
    spec_valid_dir = cfg.CACHE_DIR / 'spectrograms' / f'valid_fold{cfg.fold}'

    precompute_kaggle_spec_cache(df_train, spec_train_dir, cfg)
    precompute_kaggle_spec_cache(df_valid, spec_valid_dir, cfg)

    train_ds = KaggleSpectrogramDataset(
        df_train, split='train', cfg=cfg, augment=True, precomputed_dir=spec_train_dir,
    )
    valid_ds = KaggleSpectrogramDataset(
        df_valid, split='valid', cfg=cfg, augment=False, precomputed_dir=spec_valid_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    print(f'Train loader: {len(train_ds)} samples, {len(train_loader)} batches')
    print(f'Valid loader: {len(valid_ds)} samples, {len(valid_loader)} batches')
    return train_loader, valid_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SwinSpectrogramClassifier(nn.Module):
    """Swin Transformer backbone with a lightweight EEG classification head.

    Parameters
    ----------
    cfg : SpectrogramVisionCFG
        Pipeline configuration.
    pretrained : bool, optional
        Load ImageNet-1K pretrained weights via timm (default True).

    Attributes
    ----------
    backbone : torch.nn.Module
        Swin-T feature extractor (timm, ``num_classes=0``).
    head : torch.nn.Sequential
        ``Linear(feat_dim, 256) → SiLU → Dropout → Linear(256, num_classes)``.
    """

    def __init__(self, cfg: SpectrogramVisionCFG, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim: int = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        features = self.backbone(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: SwinSpectrogramClassifier) -> None:
    """Freeze all backbone parameters; head remains trainable.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Model instance.

    Returns
    -------
    None
        Modifies ``model.backbone`` parameters in place.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def unfreeze_last_stages(
    model: SwinSpectrogramClassifier,
    stages: Tuple[int, ...] = (2, 3),
) -> None:
    """Freeze the full backbone, then selectively unfreeze requested stages.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Model instance.
    stages : tuple of int, optional
        Indices of ``backbone.layers`` to unfreeze (default: stages 2 and 3).

    Returns
    -------
    None
        Modifies model parameters in place.

    Notes
    -----
    ``backbone.norm`` (final layer norm) and the classification head are always
    left trainable regardless of ``stages``.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    for stage_idx in stages:
        for param in model.backbone.layers[stage_idx].parameters():
            param.requires_grad = True

    if hasattr(model.backbone, 'norm'):
        for param in model.backbone.norm.parameters():
            param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True


def unfreeze_all(model: SwinSpectrogramClassifier) -> None:
    """Make all model parameters trainable.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Model instance.

    Returns
    -------
    None
        Modifies model parameters in place.
    """
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_vit_optimizer(
    model: SwinSpectrogramClassifier,
    lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Build AdamW with separate learning rates for backbone and head.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Model instance.
    lr : float
        Base learning rate applied to trainable backbone parameters.
        The head always receives ``lr * 5``.
    weight_decay : float, optional
        AdamW weight decay (default 0.01).

    Returns
    -------
    torch.optim.AdamW
        Configured two-group optimizer.
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    param_groups = [
        {'params': backbone_params, 'lr': lr},
        {'params': head_params, 'lr': lr * 5},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def kl_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence loss from raw logits against soft-label targets.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output of shape ``(B, C)``.
    targets : torch.Tensor
        Soft label probabilities of shape ``(B, C)``.

    Returns
    -------
    torch.Tensor
        Scalar batch-mean KL divergence.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, targets, reduction='batchmean')


# ---------------------------------------------------------------------------
# Training / validation / inference loops
# ---------------------------------------------------------------------------

def train_one_epoch_vit(
    model: SwinSpectrogramClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: SpectrogramVisionCFG,
) -> float:
    """Run one AMP-enabled training epoch.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Model in training mode (managed by this function).
    loader : torch.utils.data.DataLoader
        Training loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scaler : torch.amp.GradScaler
        AMP gradient scaler.
    device : torch.device
        Training device.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.

    Returns
    -------
    float
        Mean KL divergence over the epoch.
    """
    model.train()
    losses: List[float] = []
    use_mixup = cfg.mixup_alpha > 0
    for x, y, _votes in tqdm(loader, desc='Train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Mixup: blend pairs of samples within the batch.
        if use_mixup:
            lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(x)
            loss = kl_loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def validate_vit(
    model: SwinSpectrogramClassifier,
    loader: DataLoader,
    device: torch.device,
    cfg: SpectrogramVisionCFG,
) -> float:
    """Evaluate mean KL loss on a loader without gradient computation.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Model instance.
    loader : torch.utils.data.DataLoader
        Validation or test loader.
    device : torch.device
        Evaluation device.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.

    Returns
    -------
    float
        Mean KL divergence over all batches.
    """
    model.eval()
    losses: List[float] = []
    for x, y, _votes in tqdm(loader, desc='Valid', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(x)
            loss = kl_loss_fn(logits, y)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def predict_vit(
    model: SwinSpectrogramClassifier,
    loader: DataLoader,
    device: torch.device,
    cfg: SpectrogramVisionCFG,
) -> np.ndarray:
    """Run inference and return softmax class probabilities.

    Parameters
    ----------
    model : SwinSpectrogramClassifier
        Trained model instance.
    loader : torch.utils.data.DataLoader
        Loader for the target split.
    device : torch.device
        Inference device.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.

    Returns
    -------
    numpy.ndarray
        Class probabilities of shape ``(N, num_classes)``, dtype float32.
    """
    model.eval()
    outputs: List[np.ndarray] = []
    for batch in tqdm(loader, desc='Predict', leave=False):
        x = batch[0].to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        outputs.append(probs.detach().cpu().numpy())
    if not outputs:
        return np.empty((0, cfg.num_classes), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

def run_vit_training(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cfg: SpectrogramVisionCFG,
    baseline_kl: float,
    checkpoint_name: str,
    resume: bool = False,
) -> Dict[str, List[float]]:
    """Three-stage Swin-T training with frozen-backbone warmup and progressive unfreeze.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    valid_loader : torch.utils.data.DataLoader
        Validation data loader.
    cfg : SpectrogramVisionCFG
        Pipeline configuration.
    baseline_kl : float
        Global-prior baseline KL for reference logging.
    checkpoint_name : str
        Filename for the checkpoint (e.g. ``'vit_fold0.pth'``).
        Artifacts are written to ``cfg.RESULTS_DIR / stem /``.
    resume : bool, optional
        If ``True``, resume training from ``checkpoint_name`` when a compatible
        checkpoint exists. The checkpoint must include optimizer, scheduler,
        scaler, and stage metadata produced by this function.

    Returns
    -------
    dict
        History dictionary with keys ``'train_kl'``, ``'valid_kl'``, ``'lr'``.

    Notes
    -----
    Stage 1 — Head warmup:
        Frozen backbone; only head trained at ``cfg.head_lr`` for
        ``cfg.num_epochs_warmup`` epochs.

    Stage 2 — Partial unfreeze:
        Swin stages 2-3 + norm unfrozen; backbone stages 0-1 stay frozen.
        ``CosineAnnealingLR(T_max=num_epochs_partial, eta_min=1e-6)``.
        Early stopping with ``cfg.early_stopping_patience``.

    Stage 3 — Full unfreeze:
        All parameters trainable; ``CosineAnnealingLR(T_max=num_epochs_full,
        eta_min=1e-7)``.  Skipped entirely when ``cfg.num_epochs_full == 0``.
        Early stopping patience = 3.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinSpectrogramClassifier(cfg, pretrained=True).to(device)
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    history: Dict[str, List[float]] = {'train_kl': [], 'valid_kl': [], 'lr': []}
    best_val = float('inf')
    patience2 = 0
    patience3 = 0
    training_complete = False
    current_stage = 'stage1'
    current_stage_epoch = 0
    optimizer = None
    scheduler = None
    resume_state = None

    ckpt_path = cfg.MODELS_DIR / checkpoint_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    stem = Path(checkpoint_name).stem
    best_ckpt_path = ckpt_path.parent / f'{stem}_best.pth'

    # Run dir for artifacts.
    run_dir = cfg.RESULTS_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    def _save_state(is_best: bool = False) -> None:
        state = {
            'epoch': len(history['train_kl']),
            'stage': current_stage,
            'stage_epoch': current_stage_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict(),
            'best_valid_kl': best_val,
            'patience2': patience2,
            'patience3': patience3,
            'training_complete': training_complete,
            'resume_supported': True,
            'history': history,
        }
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, best_ckpt_path)

    def _record(tr_kl: float, va_kl: float, lr_val: float, stage: str) -> bool:
        """Append epoch stats, save artifacts, return True if improved."""
        nonlocal best_val
        history['train_kl'].append(tr_kl)
        history['valid_kl'].append(va_kl)
        history['lr'].append(lr_val)
        epoch_num = len(history['train_kl'])
        print(
            f'[{stage}] Epoch {epoch_num} | '
            f'train_kl={tr_kl:.5f} | valid_kl={va_kl:.5f} | '
            f'lr={lr_val:.2e} | baseline={baseline_kl:.5f}'
        )
        improved = va_kl < best_val
        if improved:
            best_val = va_kl
        _save_state(is_best=improved)
        try:
            save_live_training_artifacts(history, cfg, checkpoint_name)
        except Exception as exc:
            print(f'Warning: live artifact save failed ({exc})')
        return improved

    def _restore_stage_state(stage_name: str) -> int:
        """Load optimizer/scheduler state when resuming the active stage."""
        nonlocal optimizer, scheduler, scaler, patience2, patience3
        if resume_state is None or resume_state.get('stage') != stage_name:
            return 0

        opt_state = resume_state.get('optimizer_state_dict')
        if opt_state is not None and optimizer is not None:
            optimizer.load_state_dict(opt_state)

        sched_state = resume_state.get('scheduler_state_dict')
        if sched_state is not None and scheduler is not None:
            scheduler.load_state_dict(sched_state)

        scaler_state = resume_state.get('scaler_state_dict')
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        if stage_name == 'stage2':
            patience2 = int(resume_state.get('patience2', 0))
        elif stage_name == 'stage3':
            patience3 = int(resume_state.get('patience3', 0))

        restored_epoch = int(resume_state.get('stage_epoch', 0))
        print(f"Resuming {stage_name} from completed stage epochs: {restored_epoch}")
        return restored_epoch

    if resume and ckpt_path.exists():
        loaded = torch.load(ckpt_path, map_location=device)
        if not loaded.get('resume_supported', False):
            print('Warning: checkpoint exists but does not support full resume; starting a fresh run.')
        else:
            state_dict = loaded['model_state_dict']
            # torch.compile() prefixes keys with '_orig_mod.' — strip it so the
            # checkpoint is loadable into a plain (not-yet-compiled) model.
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            history = loaded.get('history', history)
            best_val = float(loaded.get('best_valid_kl', best_val))
            patience2 = int(loaded.get('patience2', 0))
            patience3 = int(loaded.get('patience3', 0))
            training_complete = bool(loaded.get('training_complete', False))
            current_stage = loaded.get('stage', 'stage1')
            current_stage_epoch = int(loaded.get('stage_epoch', 0))
            resume_state = loaded
            print(
                f"Loaded resume checkpoint: epoch={loaded.get('epoch', 0)} "
                f"stage={current_stage} stage_epoch={current_stage_epoch} "
                f"best_valid_kl={best_val:.5f}"
            )
            if training_complete:
                print('Checkpoint already marked complete; returning saved history.')
                return history

    # Optionally JIT-compile the model for faster forward/backward (PyTorch 2.x+).
    # The first epoch will be slower due to compilation; subsequent epochs are faster.
    if getattr(cfg, 'use_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print('torch.compile() applied — first epoch will be slower (JIT compilation).')

    # ── Stage 1: Head warmup ───────────────────────────────────────────────
    current_stage = 'stage1'
    freeze_backbone(model)
    optimizer = build_vit_optimizer(model, cfg.head_lr, cfg.weight_decay)
    scheduler = None

    try:
        save_run_config_artifact(
            cfg, checkpoint_name, 'stage1', model,
            {'baseline_kl': float(baseline_kl), 'num_epochs': cfg.num_epochs_warmup},
        )
    except Exception as exc:
        print(f'Warning: stage1 config artifact failed ({exc})')

    print(f'\n=== Stage 1: Head warmup ({cfg.num_epochs_warmup} epochs) ===')
    start_epoch = _restore_stage_state('stage1')
    # If the checkpoint is already past Stage 1, skip it entirely.
    if resume_state is not None and resume_state.get('stage') in ('stage2', 'stage3'):
        start_epoch = cfg.num_epochs_warmup
        print('Resuming: Stage 1 already completed, skipping.')
    current_stage_epoch = start_epoch
    for stage_epoch_idx in range(start_epoch, cfg.num_epochs_warmup):
        tr_kl = train_one_epoch_vit(model, train_loader, optimizer, scaler, device, cfg)
        va_kl = validate_vit(model, valid_loader, device, cfg)
        lr_val = float(optimizer.param_groups[0]['lr'])
        current_stage_epoch = stage_epoch_idx + 1
        _record(tr_kl, va_kl, lr_val, 'stage1')

    # ── Stage 2: Partial unfreeze ──────────────────────────────────────────
    current_stage = 'stage2'
    unfreeze_last_stages(model, stages=cfg.partial_unfreeze_stages)
    optimizer = build_vit_optimizer(model, cfg.lr, cfg.weight_decay)
    _t_max2 = cfg.cosine_t_max if getattr(cfg, 'cosine_t_max', 0) > 0 else cfg.num_epochs_partial
    scheduler2 = CosineAnnealingLR(optimizer, T_max=_t_max2, eta_min=1e-6)
    scheduler = scheduler2

    try:
        save_run_config_artifact(
            cfg, checkpoint_name, 'stage2', model,
            {'baseline_kl': float(baseline_kl), 'num_epochs': cfg.num_epochs_partial},
        )
    except Exception as exc:
        print(f'Warning: stage2 config artifact failed ({exc})')

    print(f'\n=== Stage 2: Partial unfreeze ({cfg.num_epochs_partial} epochs, '
          f'patience={cfg.early_stopping_patience}) ===')
    start_epoch = _restore_stage_state('stage2')
    # If the checkpoint is already past Stage 2, skip it entirely.
    if resume_state is not None and resume_state.get('stage') == 'stage3':
        start_epoch = cfg.num_epochs_partial
        print('Resuming: Stage 2 already completed, skipping.')
    current_stage_epoch = start_epoch
    if start_epoch == 0:
        patience2 = 0
    for stage_epoch_idx in range(start_epoch, cfg.num_epochs_partial):
        tr_kl = train_one_epoch_vit(model, train_loader, optimizer, scaler, device, cfg)
        va_kl = validate_vit(model, valid_loader, device, cfg)
        scheduler2.step()
        lr_val = float(optimizer.param_groups[0]['lr'])
        current_stage_epoch = stage_epoch_idx + 1
        improved = _record(tr_kl, va_kl, lr_val, 'stage2')
        if improved:
            patience2 = 0
        else:
            patience2 += 1
        if patience2 >= cfg.early_stopping_patience:
            print('Early stopping triggered (stage2).')
            break

    # ── Stage 3: Full fine-tune ────────────────────────────────────────────
    if cfg.num_epochs_full == 0:
        print('\nStage 3 skipped (num_epochs_full=0).')
    else:
        current_stage = 'stage3'
        unfreeze_all(model)
        # Stage-3 LR: ~0.25x of the Stage-2 base LR → 5e-5 when cfg.lr = 2e-4.
        optimizer = build_vit_optimizer(model, cfg.lr * 0.25, cfg.weight_decay)
        scheduler3 = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs_full, eta_min=1e-7)
        scheduler = scheduler3

        try:
            save_run_config_artifact(
                cfg, checkpoint_name, 'stage3', model,
                {'baseline_kl': float(baseline_kl), 'num_epochs': cfg.num_epochs_full},
            )
        except Exception as exc:
            print(f'Warning: stage3 config artifact failed ({exc})')

        print(f'\n=== Stage 3: Full fine-tune ({cfg.num_epochs_full} epochs, patience=3) ===')
        start_epoch = _restore_stage_state('stage3')
        current_stage_epoch = start_epoch
        if start_epoch == 0:
            patience3 = 0
        for stage_epoch_idx in range(start_epoch, cfg.num_epochs_full):
            tr_kl = train_one_epoch_vit(model, train_loader, optimizer, scaler, device, cfg)
            va_kl = validate_vit(model, valid_loader, device, cfg)
            scheduler3.step()
            lr_val = float(optimizer.param_groups[0]['lr'])
            current_stage_epoch = stage_epoch_idx + 1
            improved = _record(tr_kl, va_kl, lr_val, 'stage3')
            if improved:
                patience3 = 0
            else:
                patience3 += 1
            if patience3 >= 3:
                print('Early stopping triggered (stage3).')
                break

    training_complete = True
    _save_state(is_best=False)
    print(f'\nTraining complete. Best valid KL: {best_val:.5f}')
    return history


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def assert_no_test_leakage(df_train, df_valid, df_test) -> None:
    """Assert zero patient_id overlap across all three data splits.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training metadata with a ``patient_id`` column.
    df_valid : pandas.DataFrame
        Validation metadata with a ``patient_id`` column.
    df_test : pandas.DataFrame
        Held-out test metadata (``confident_test.csv``) with a
        ``patient_id`` column.

    Returns
    -------
    None
        Prints a confirmation message if all assertions pass.

    Raises
    ------
    AssertionError
        If any ``patient_id`` appears in more than one split.
    """
    train_pts = set(df_train['patient_id'].unique())
    valid_pts = set(df_valid['patient_id'].unique())
    test_pts = set(df_test['patient_id'].unique())

    overlap_tv = train_pts & valid_pts
    overlap_tt = train_pts & test_pts
    overlap_vt = valid_pts & test_pts

    assert len(overlap_tt) == 0, (
        f'LEAKAGE: {len(overlap_tt)} patient(s) shared between train and test: '
        f'{sorted(overlap_tt)[:5]}'
    )
    assert len(overlap_vt) == 0, (
        f'LEAKAGE: {len(overlap_vt)} patient(s) shared between valid and test: '
        f'{sorted(overlap_vt)[:5]}'
    )
    assert len(overlap_tv) == 0, (
        f'LEAKAGE: {len(overlap_tv)} patient(s) shared between train and valid: '
        f'{sorted(overlap_tv)[:5]}'
    )
    print('Split integrity verified: zero patient_id overlap across train / valid / test.')
