# DISCLAIMER: docstrings created by ChatGPT and may require review for accuracy and completeness.

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

AUG_COUNTER_KEYS = (
    'noise',
    'scale',
    'time_shift_nonzero',
    'left_right_flip',
    'mixup',
    'time_mask',
    'channel_drop',
)

_AUG_COUNTERS: Dict[str, int] = {k: 0 for k in AUG_COUNTER_KEYS}

# Channel remap for 16 bipolar channels in chain order:
# [LL(0:4), RL(4:8), LP(8:12), RP(12:16)] -> swap left/right chains.
_LR_FLIP_INDEX = np.array(
    [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
    dtype=np.int64,
)


def _inc_aug_counter(name: str, n: int = 1) -> None:
    """Increment one augmentation counter in-process.

    Parameters
    ----------
    name : str
        Counter key.
    n : int, default=1
        Increment amount.

    Returns
    -------
    None
        Updates module-level counters.
    """
    if name not in _AUG_COUNTERS:
        return
    _AUG_COUNTERS[name] += int(max(0, n))


def reset_augmentation_counters() -> None:
    """Reset all augmentation counters to zero.

    Returns
    -------
    None
        Updates module-level counters in place.
    """
    for k in _AUG_COUNTERS:
        _AUG_COUNTERS[k] = 0


def get_augmentation_counters() -> Dict[str, int]:
    """Return a snapshot of augmentation counters.

    Returns
    -------
    dict
        Counter dictionary keyed by augmentation name.
    """
    return {k: int(v) for k, v in _AUG_COUNTERS.items()}


def _cfg_float(cfg, name: str, default: float) -> float:
    """Read float config values with a fallback.

    Parameters
    ----------
    cfg : object
        Configuration object with optional attribute ``name``.
    name : str
        Attribute name to read.
    default : float
        Fallback value when ``name`` is missing.

    Returns
    -------
    float
        Parsed float value.
    """
    return float(getattr(cfg, name, default))


def _cfg_int(cfg, name: str, default: int) -> int:
    """Read integer config values with a fallback.

    Parameters
    ----------
    cfg : object
        Configuration object with optional attribute ``name``.
    name : str
        Attribute name to read.
    default : int
        Fallback value when ``name`` is missing.

    Returns
    -------
    int
        Parsed integer value.
    """
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg, name: str, default: bool) -> bool:
    """Read boolean config values with a fallback.

    Parameters
    ----------
    cfg : object
        Configuration object with optional attribute ``name``.
    name : str
        Attribute name to read.
    default : bool
        Fallback value when ``name`` is missing.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    return bool(getattr(cfg, name, default))


def apply_time_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Apply zero-padded temporal shift to a multi-channel signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal array of shape ``(channels, time)``.
    shift : int
        Signed temporal shift in samples. Positive values shift right.

    Returns
    -------
    numpy.ndarray
        Shifted signal array with the same shape as ``x``.
    """
    if shift == 0:
        return x
    out = np.zeros_like(x)
    if shift > 0:
        out[:, shift:] = x[:, :-shift]
    else:
        out[:, :shift] = x[:, -shift:]
    return out


def apply_left_right_flip(x: np.ndarray) -> np.ndarray:
    """Swap left and right bipolar chains for 16-channel inputs.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal array of shape ``(channels, time)``.

    Returns
    -------
    numpy.ndarray
        Left-right flipped signal with unchanged shape.
    """
    if x.ndim != 2:
        return x
    channels = int(x.shape[0])
    if channels < int(_LR_FLIP_INDEX.shape[0]):
        return x
    if channels == int(_LR_FLIP_INDEX.shape[0]):
        return x[_LR_FLIP_INDEX, :]

    # Preserve any extra channels beyond the 16 bipolar channels.
    out = x.copy()
    out[:16, :] = x[_LR_FLIP_INDEX, :]
    return out


def augment_sample_np(x: np.ndarray, cfg, target_sample_rate: int) -> np.ndarray:
    """Apply per-sample raw EEG augmentations in numpy space.

    Parameters
    ----------
    x : numpy.ndarray
        Input EEG sample of shape ``(channels, time)``.
    cfg : object
        Runtime configuration with augmentation ranges.
    target_sample_rate : int
        Sampling rate used to convert shift seconds to samples.

    Returns
    -------
    numpy.ndarray
        Augmented EEG sample with unchanged shape.
    """
    noise_std_min = _cfg_float(cfg, 'aug_noise_std_min', 0.0015)
    noise_std_max = _cfg_float(cfg, 'aug_noise_std_max', 0.0060)
    scale_min = _cfg_float(cfg, 'aug_scale_min', 0.93)
    scale_max = _cfg_float(cfg, 'aug_scale_max', 1.07)
    max_shift_seconds = _cfg_float(cfg, 'aug_max_shift_seconds', 0.25)
    left_right_flip_prob = _cfg_float(cfg, 'left_right_flip_prob', 0.15)

    if noise_std_max > 0:
        noise_std = np.random.uniform(noise_std_min, noise_std_max)
        x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        _inc_aug_counter('noise', 1)

    if scale_max > 0:
        scale = np.random.uniform(scale_min, scale_max)
        x = x * scale
        _inc_aug_counter('scale', 1)

    max_shift = max(0, int(round(max_shift_seconds * float(target_sample_rate))))
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        x = apply_time_shift(x, shift)
        if shift != 0:
            _inc_aug_counter('time_shift_nonzero', 1)

    if left_right_flip_prob > 0.0 and np.random.rand() <= left_right_flip_prob:
        x = apply_left_right_flip(x)
        _inc_aug_counter('left_right_flip', 1)

    return x.astype(np.float32)


def apply_mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    votes: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MixUp to a training batch.

    Parameters
    ----------
    x : torch.Tensor
        Batch tensor of shape ``(batch, channels, time)``.
    y : torch.Tensor
        Soft label tensor of shape ``(batch, classes)``.
    votes : torch.Tensor
        Vote-count tensor of shape ``(batch,)``.
    cfg : object
        Runtime configuration containing MixUp settings.

    Returns
    -------
    tuple of torch.Tensor
        Mixed tensors ``(x, y, votes)`` with unchanged shapes.
    """
    use_mixup = _cfg_bool(cfg, 'use_mixup', True)
    mixup_prob = _cfg_float(cfg, 'mixup_prob', 0.60)
    mixup_alpha = _cfg_float(cfg, 'mixup_alpha', 0.40)

    bsz = int(x.shape[0])
    if not use_mixup or bsz < 2 or mixup_prob <= 0.0:
        return x, y, votes
    if np.random.rand() > mixup_prob:
        return x, y, votes

    alpha = max(mixup_alpha, 1e-6)
    lam_np = np.random.beta(alpha, alpha, size=bsz).astype(np.float32)
    perm = torch.randperm(bsz, device=x.device)
    lam = torch.tensor(lam_np, dtype=x.dtype, device=x.device)

    lam_x = lam.view(bsz, 1, 1)
    lam_y = lam.view(bsz, 1)

    x_mix = lam_x * x + (1.0 - lam_x) * x[perm]
    y_mix = lam_y * y + (1.0 - lam_y) * y[perm]

    votes_float = votes.to(dtype=x.dtype)
    votes_mix = lam * votes_float + (1.0 - lam) * votes_float[perm]
    votes_mix = votes_mix.to(dtype=votes.dtype)
    _inc_aug_counter('mixup', bsz)

    return x_mix, y_mix, votes_mix


def apply_time_mask_batch(x: torch.Tensor, cfg) -> torch.Tensor:
    """Apply random temporal masking to batch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Batch tensor of shape ``(batch, channels, time)``.
    cfg : object
        Runtime configuration containing time-mask settings.

    Returns
    -------
    torch.Tensor
        Time-masked batch tensor with unchanged shape.
    """
    prob = _cfg_float(cfg, 'time_mask_prob', 0.35)
    frac_min = _cfg_float(cfg, 'time_mask_frac_min', 0.03)
    frac_max = _cfg_float(cfg, 'time_mask_frac_max', 0.10)

    if prob <= 0.0:
        return x

    bsz, _channels, seq_len = x.shape
    min_len = max(1, int(round(frac_min * seq_len)))
    max_len = max(min_len, int(round(frac_max * seq_len)))

    masked = 0
    for i in range(bsz):
        if np.random.rand() > prob:
            continue
        span = np.random.randint(min_len, max_len + 1)
        if span >= seq_len:
            x[i, :, :] = 0
            masked += 1
            continue
        start = np.random.randint(0, seq_len - span + 1)
        x[i, :, start:start + span] = 0
        masked += 1
    if masked > 0:
        _inc_aug_counter('time_mask', masked)
    return x


def apply_channel_dropout_batch(x: torch.Tensor, cfg) -> torch.Tensor:
    """Drop random channels in batch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Batch tensor of shape ``(batch, channels, time)``.
    cfg : object
        Runtime configuration containing channel-drop settings.

    Returns
    -------
    torch.Tensor
        Channel-dropped batch tensor with unchanged shape.
    """
    prob = _cfg_float(cfg, 'channel_drop_prob', 0.25)
    max_drop = _cfg_int(cfg, 'channel_drop_max', 2)

    if prob <= 0.0 or max_drop <= 0:
        return x

    bsz, channels, _seq_len = x.shape
    max_drop = min(max_drop, channels)
    if max_drop <= 0:
        return x

    dropped = 0
    for i in range(bsz):
        if np.random.rand() > prob:
            continue
        n_drop = np.random.randint(1, max_drop + 1)
        drop_idx = np.random.choice(channels, size=n_drop, replace=False)
        x[i, drop_idx, :] = 0
        dropped += 1
    if dropped > 0:
        _inc_aug_counter('channel_drop', dropped)
    return x


def build_train_collate_fn(
    cfg,
) -> Callable[
    [List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Create a collate function with batch-level train augmentations.

    Parameters
    ----------
    cfg : object
        Runtime configuration with MixUp and masking settings.

    Returns
    -------
    callable
        Collate function for train DataLoader batches.
    """

    def collate_train(batch):
        """Collate and augment one train batch.

        Parameters
        ----------
        batch : list of tuple
            List of ``(x, y, votes)`` samples from the dataset.

        Returns
        -------
        tuple of torch.Tensor
            Augmented batch tensors ``(x, y, votes)``.
        """
        x_list, y_list, v_list = zip(*batch)
        x = torch.stack(x_list, dim=0).float()
        y = torch.stack(y_list, dim=0).float()
        votes = torch.stack(v_list, dim=0)

        x, y, votes = apply_mixup_batch(x, y, votes, cfg)
        x = apply_time_mask_batch(x, cfg)
        x = apply_channel_dropout_batch(x, cfg)
        return x, y, votes

    return collate_train
