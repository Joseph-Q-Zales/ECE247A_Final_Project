# DISCLAIMER: docstrings created by ChatGPT and may require review for accuracy and completeness.

from __future__ import annotations

import json
import os
import random
import subprocess
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_aug import (
    apply_time_shift,
    augment_sample_np,
    build_train_collate_fn,
    get_augmentation_counters,
    reset_augmentation_counters,
)
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Default bipolar montage definition (16 channels).
BIPOLAR_MONTAGE = {
    'LL': [('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1')],
    'RL': [('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2')],
    'LP': [('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1')],
    'RP': [('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')],
}

CHAIN_ORDER = ['LL', 'RL', 'LP', 'RP']
BIPOLAR_PAIRS: List[Tuple[str, str]] = []
for _chain in CHAIN_ORDER:
    BIPOLAR_PAIRS.extend(BIPOLAR_MONTAGE[_chain])


def ensure_dirs(paths: List[Path]) -> None:
    """Create directories if they do not already exist.

    Parameters
    ----------
    paths : list of pathlib.Path
        Directory paths to create.

    Returns
    -------
    None
        This function updates the filesystem in place.
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _to_jsonable(value: Any) -> Any:
    """Convert runtime values into JSON-serializable representations.

    Parameters
    ----------
    value : Any
        Arbitrary Python object.

    Returns
    -------
    Any
        JSON-safe representation.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cfg_to_dict(cfg) -> Dict[str, Any]:
    """Extract public config attributes into a plain dictionary.

    Parameters
    ----------
    cfg : object
        Runtime config object.

    Returns
    -------
    dict
        Serializable config dictionary.
    """
    out: Dict[str, Any] = {}
    for name in dir(cfg):
        if name.startswith('_'):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        out[name] = _to_jsonable(value)
    return out


def save_run_config_artifact(
    cfg,
    checkpoint_name: str,
    stage_name: str,
    model: nn.Module,
    stage_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write stage-level run configuration and architecture metadata.

    Parameters
    ----------
    cfg : object
        Runtime configuration object.
    checkpoint_name : str
        Checkpoint filename used to derive run directory.
    stage_name : str
        Stage identifier, e.g. ``'stage1'`` or ``'stage2'``.
    model : torch.nn.Module
        Model instance used for parameter counting.
    stage_metadata : dict, optional
        Additional stage-specific metadata.

    Returns
    -------
    pathlib.Path
        Path to the written JSON artifact.
    """
    stem = Path(checkpoint_name).stem
    run_dir = cfg.RESULTS_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    n_total = int(sum(p.numel() for p in model.parameters()))
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    payload = {
        'stage_name': stage_name,
        'checkpoint_name': checkpoint_name,
        'model_architecture': {
            'model_class': model.__class__.__name__,
            'conv_channels': list(getattr(cfg, 'conv_channels', [])),
            'conv_kernels': list(getattr(cfg, 'conv_kernels', [])),
            'conv_strides': list(getattr(cfg, 'conv_strides', [])),
            'gru_hidden': int(getattr(cfg, 'gru_hidden', 0)),
            'gru_layers': int(getattr(cfg, 'gru_layers', 0)),
            'dropout': float(getattr(cfg, 'dropout', 0.0)),
            'use_multiscale_conv': bool(getattr(cfg, 'use_multiscale_conv', False)),
            'multiscale_kernels': list(getattr(cfg, 'multiscale_kernels', (3, 15, 31))),
            'num_parameters_total': n_total,
            'num_parameters_trainable': n_trainable,
        },
        'runtime': {
            'base_path': str(getattr(cfg, 'BASE_PATH', '')),
            'work_dir': str(getattr(cfg, 'WORK_DIR', '')),
            'cache_dir': str(getattr(cfg, 'CACHE_DIR', '')),
            'models_dir': str(getattr(cfg, 'MODELS_DIR', '')),
            'results_dir': str(getattr(cfg, 'RESULTS_DIR', '')),
        },
        'config': _cfg_to_dict(cfg),
        'stage_metadata': _to_jsonable(stage_metadata or {}),
    }

    out_path = run_dir / f'run_config_{stage_name}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return out_path


def save_augmentation_artifacts(
    augmentation_history: List[Dict[str, Any]],
    cfg,
    checkpoint_name: str,
) -> None:
    """Save per-epoch augmentation counters to CSV and JSON artifacts.

    Parameters
    ----------
    augmentation_history : list of dict
        Per-epoch counter snapshots.
    cfg : object
        Runtime configuration.
    checkpoint_name : str
        Checkpoint filename used to derive run directory.

    Returns
    -------
    None
        Writes augmentation artifacts to disk.
    """
    if not augmentation_history:
        return

    stem = Path(checkpoint_name).stem
    run_dir = cfg.RESULTS_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    aug_csv = run_dir / 'augmentation_epoch_stats.csv'
    aug_json = run_dir / 'augmentation_epoch_stats.json'

    aug_df = pd.DataFrame(augmentation_history)
    aug_df.to_csv(aug_csv, index=False)
    aug_json.write_text(
        json.dumps([_to_jsonable(r) for r in augmentation_history], indent=2),
        encoding='utf-8',
    )


def _get_colab_secret(userdata_obj, names: List[str]) -> str:
    """Read first available non-empty secret value from Colab userdata.

    Parameters
    ----------
    userdata_obj : object
        ``google.colab.userdata`` module.
    names : list of str
        Secret names to try in order.

    Returns
    -------
    str
        Secret value.
    """
    for name in names:
        try:
            value = userdata_obj.get(name)
        except Exception:
            value = None

        if isinstance(value, str):
            value = value.strip().strip('"').strip("'")

        if value:
            return value

    return ''


def setup_colab_auth_from_secrets() -> None:
    """Load Kaggle credentials from Colab secrets into env vars and kaggle.json.

    Parameters
    ----------
    None
        This function reads Colab secrets from runtime context.

    Expected keys
    -------------
    kaggle_uname
    kaggle_key

    Returns
    -------
    None
        This function sets process environment variables and writes ``kaggle.json``.
    """
    try:
        from google.colab import userdata
    except Exception as exc:
        raise RuntimeError('google.colab.userdata is unavailable. Set COLAB=False or run in Colab.') from exc

    kaggle_uname = _get_colab_secret(userdata, ['kaggle_uname', 'kaggle_username', 'KAGGLE_USERNAME'])
    kaggle_key = _get_colab_secret(userdata, ['kaggle_key', 'KAGGLE_KEY'])

    if not kaggle_uname or not kaggle_key:
        raise RuntimeError(
            'Missing Kaggle secrets in Colab. Add secrets named "kaggle_uname" and "kaggle_key".'
        )

    # Same approach as prior notebooks: export env vars for Kaggle CLI.
    os.environ['KAGGLE_USERNAME'] = kaggle_uname
    os.environ['KAGGLE_KEY'] = kaggle_key
    print('got kaggle creds')

    # Also persist kaggle.json for CLI compatibility.
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / 'kaggle.json'
    kaggle_json.write_text(
        json.dumps({'username': kaggle_uname, 'key': kaggle_key}),
        encoding='utf-8',
    )
    os.chmod(kaggle_json, 0o600)


def _run_command(cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the result.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    cwd : pathlib.Path, optional
        Working directory.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process object.
    """
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def build_confident_csvs(
    base_path: Path,
    agreement_threshold: float = 0.70,
    test_fraction: float = 0.10,
    seed: int = 42,
    force_rebuild: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Create confidence-filtered metadata CSV files.

    This function filters ``train.csv`` rows by confidence and then creates
    deterministic train/test splits from the filtered subset.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory containing HMS metadata files.
    agreement_threshold : float, default=0.70
        Minimum top-class vote fraction required to keep a training row.
    test_fraction : float, default=0.10
        Target fraction of confidence-filtered rows assigned to holdout test.
    seed : int, default=42
        Random seed used for deterministic grouped splitting.
    force_rebuild : bool, default=False
        If True, regenerate outputs even if they already exist.
    output_dir : pathlib.Path, optional
        Destination directory for ``confident_train.csv`` and ``confident_test.csv``.
        If None, files are written under ``base_path``.

    Returns
    -------
    dict
        Dictionary with keys ``train_csv`` and ``test_csv`` pointing to the generated files.
    """
    base_path = Path(base_path)
    test_fraction = float(test_fraction)
    seed = int(seed)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f'test_fraction must be in (0, 1), got {test_fraction}')

    if output_dir is None:
        output_dir = base_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = base_path / 'train.csv'
    out_train = output_dir / 'confident_train.csv'
    out_test = output_dir / 'confident_test.csv'

    if not train_csv.exists():
        raise FileNotFoundError(f'Missing train.csv at {train_csv}')

    vote_cols = [
        'seizure_vote',
        'lpd_vote',
        'gpd_vote',
        'lrda_vote',
        'grda_vote',
        'other_vote',
    ]

    if out_train.exists() and out_test.exists() and not force_rebuild:
        try:
            old_train = pd.read_csv(out_train, nrows=5)
            old_test = pd.read_csv(out_test, nrows=5)
            has_votes_train = all(c in old_train.columns for c in vote_cols)
            has_votes_test = all(c in old_test.columns for c in vote_cols)
            if has_votes_train and has_votes_test:
                print(f'Confidence CSVs already exist. Using: {out_train} and {out_test}')
                return {'train_csv': out_train, 'test_csv': out_test}
            print('Existing confident CSVs are legacy/incompatible; rebuilding with train-derived holdout test split.')
        except Exception:
            print('Could not validate existing confidence CSVs; rebuilding.')

    df_train = pd.read_csv(train_csv)
    missing_vote_cols = [c for c in vote_cols if c not in df_train.columns]
    if missing_vote_cols:
        raise ValueError(f'train.csv is missing vote columns: {missing_vote_cols}')

    votes = df_train[vote_cols].astype(np.float32).values
    total_votes = votes.sum(axis=1)
    max_votes = votes.max(axis=1)
    agreement = np.divide(
        max_votes,
        np.clip(total_votes, 1e-6, None),
        out=np.zeros_like(max_votes, dtype=np.float32),
        where=total_votes > 0,
    )
    keep_mask = agreement >= float(agreement_threshold)

    df_conf = df_train.loc[keep_mask].copy().reset_index(drop=True)
    if len(df_conf) < 2:
        raise ValueError('Not enough confidence-filtered rows to build train/test split.')

    target_test_rows = max(1, int(round(len(df_conf) * test_fraction)))
    group_col = 'patient_id' if 'patient_id' in df_conf.columns else 'eeg_id'
    if group_col not in df_conf.columns:
        raise ValueError('Neither patient_id nor eeg_id exists for grouped split.')

    if 'expert_consensus' in df_conf.columns and df_conf['expert_consensus'].notna().all():
        split_y = df_conf['expert_consensus'].astype(str).values
    else:
        split_y = df_conf[vote_cols].values.argmax(axis=1)

    best_valid_idx = None
    best_meta = None

    ideal_splits = max(2, int(round(1.0 / test_fraction)))
    candidate_splits = sorted(
        {
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            int(np.floor(1.0 / test_fraction)),
            int(np.ceil(1.0 / test_fraction)),
            ideal_splits,
        },
        reverse=True,
    )
    n_groups = int(df_conf[group_col].nunique())
    class_counts = pd.Series(split_y).value_counts()

    for n_splits in candidate_splits:
        if n_splits < 2:
            continue
        if n_splits > n_groups:
            continue
        if int(class_counts.min()) < n_splits:
            continue

        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx, (_train_idx, valid_idx) in enumerate(
                sgkf.split(df_conf, y=split_y, groups=df_conf[group_col].values)
            ):
                fold_rows = int(len(valid_idx))
                fold_frac = fold_rows / max(len(df_conf), 1)
                diff_rows = abs(fold_rows - target_test_rows)
                score = (diff_rows, abs(fold_frac - test_fraction), -n_splits, fold_idx)
                if best_meta is None or score < best_meta:
                    best_meta = score
                    best_valid_idx = valid_idx
        except Exception:
            continue

    if best_valid_idx is not None:
        test_mask = np.zeros(len(df_conf), dtype=bool)
        test_mask[np.asarray(best_valid_idx, dtype=int)] = True
    else:
        # Fallback: deterministic grouped sampling when SGKF constraints fail.
        group_sizes = df_conf.groupby(group_col).size().sort_index()
        group_sizes = group_sizes.sample(frac=1.0, random_state=seed)
        cum = group_sizes.cumsum().values
        pick = int(np.argmin(np.abs(cum - target_test_rows)))
        chosen_groups = set(group_sizes.index[:pick + 1].tolist())
        test_mask = df_conf[group_col].isin(chosen_groups).values

    if test_mask.sum() == 0 or test_mask.sum() == len(df_conf):
        raise RuntimeError('Failed to construct non-empty confidence train/test split.')

    df_conf_test = df_conf.loc[test_mask].copy()
    df_conf_train = df_conf.loc[~test_mask].copy()
    df_conf_train.to_csv(out_train, index=False)
    df_conf_test.to_csv(out_test, index=False)

    kept = int(len(df_conf))
    total = int(len(df_train))
    actual_test_fraction = len(df_conf_test) / max(len(df_conf), 1)
    overlap_groups = set(df_conf_train[group_col]).intersection(set(df_conf_test[group_col]))
    print(
        f'Built confidence pool: kept {kept}/{total} rows '
        f'({(100.0 * kept / max(total, 1)):.2f}%) at threshold={agreement_threshold:.2f}'
    )
    print(
        f'Built split with seed={seed}, target_test_fraction={test_fraction:.3f}, '
        f'actual_test_fraction={actual_test_fraction:.3f}'
    )
    print(f'Built {out_train.name}: {len(df_conf_train)} rows')
    print(f'Built {out_test.name}: {len(df_conf_test)} rows')
    print(f'Group overlap count ({group_col}): {len(overlap_groups)}')

    return {'train_csv': out_train, 'test_csv': out_test}


def ensure_hms_data_colab(base_path: Path) -> None:
    """Ensure HMS competition data exists locally in Colab.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory where competition files should exist or be downloaded.

    Returns
    -------
    None
        This function downloads/extracts data when required.
    """
    required = [
        base_path / 'train.csv',
        base_path / 'test.csv',
        base_path / 'sample_submission.csv',
        base_path / 'train_eegs',
        base_path / 'test_eegs',
    ]

    if all(p.exists() for p in required):
        print(f'HMS data already present at {base_path}. Skipping download.')
        return

    base_path.mkdir(parents=True, exist_ok=True)

    # Ensure kaggle CLI exists.
    kaggle_check = _run_command(['bash', '-lc', 'command -v kaggle'])
    if kaggle_check.returncode != 0:
        install_res = _run_command(['python', '-m', 'pip', 'install', '-q', 'kaggle'])
        if install_res.returncode != 0:
            raise RuntimeError(f'Failed to install kaggle CLI: {install_res.stderr}')

    dl_cmd = [
        'kaggle', 'competitions', 'download',
        '-c', 'hms-harmful-brain-activity-classification',
        '-p', str(base_path),
    ]
    dl_res = _run_command(dl_cmd)
    if dl_res.returncode != 0:
        msg = dl_res.stderr or dl_res.stdout
        if '403' in msg or 'forbidden' in msg.lower():
            raise RuntimeError(
                'Kaggle download returned 403. Accept HMS competition rules in Kaggle browser first, then rerun.'
            )
        raise RuntimeError(f'Kaggle download failed: {msg}')

    while True:
        zip_files = sorted(base_path.rglob('*.zip'))
        if not zip_files:
            break
        for zf in zip_files:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(zf.parent)
            zf.unlink(missing_ok=True)

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f'HMS data setup incomplete. Missing: {missing}')

    print(f'HMS data ready at {base_path}.')


def setup_runtime(
    colab: bool,
    colab_use_drive: bool = True,
    colab_drive_root: str = '/content/drive/MyDrive/HMS_JZ',
    colab_content_root: str = '/content/hms',
    local_work_dir: str = '/home/littl/ECE247A_Final_Project/JZ',
    confident: bool = False,
    confident_threshold: float = 0.70,
    confident_test_fraction: float = 0.10,
    confident_seed: int = 42,
    confident_force_rebuild: bool = False,
    confident_output_subdir: Optional[str] = None,
) -> Dict[str, object]:
    """Build runtime settings for local VM or Colab.

    Parameters
    ----------
    colab : bool
        Manual gate that controls whether Colab setup is executed.
    confident : bool, default=False
        If True, build and use confidence-filtered metadata CSV files.
    confident_threshold : float, default=0.70
        Minimum top-class vote share used for confidence filtering.
    confident_test_fraction : float, default=0.10
        Target fraction of confidence-filtered rows assigned to holdout test.
    confident_seed : int, default=42
        Seed for deterministic confidence train/test splitting.
    confident_force_rebuild : bool, default=False
        If True, rebuild confidence CSVs even when existing files are present.
    confident_output_subdir : str or None, default=None
        Subdirectory under ``work_dir`` where confidence CSVs are saved.
        If None, confidence CSVs are written directly under ``base_path``.

    Returns
    -------
    dict
        Runtime settings dictionary consumed by CFG.
    """
    if colab:
        try:
            from google.colab import drive
        except Exception as exc:
            raise RuntimeError('COLAB=True but google.colab is unavailable. Set COLAB=False in this environment.') from exc

        use_drive = bool(colab_use_drive)
        drive_root = Path(colab_drive_root)

        if use_drive:
            try:
                drive.mount('/content/drive', force_remount=False)
            except Exception as exc:
                warnings.warn(f'Drive mount failed ({exc}). Falling back to /content for models/results.')
                use_drive = False

        setup_colab_auth_from_secrets()

        base_path = Path(colab_content_root)
        ensure_hms_data_colab(base_path)

        if use_drive:
            work_root = drive_root
        else:
            work_root = Path('/content/HMS_JZ')

        settings = {
            'colab': True,
            'base_path': base_path,
            'work_dir': work_root,
            'cache_dir': Path('/content/hms_cache'),
            'models_dir': work_root / 'models',
            'results_dir': work_root / 'results',
            'plots_dir': work_root / 'results' / 'plots',
            'num_workers': 0,
        }
    else:
        work_dir = Path(local_work_dir)
        settings = {
            'colab': False,
            'base_path': Path('/home/littl/data/data'),
            'work_dir': work_dir,
            'cache_dir': work_dir / 'cache',
            'models_dir': work_dir / 'models',
            'results_dir': work_dir / 'results',
            'plots_dir': work_dir / 'results' / 'plots',
            'num_workers': 2,
        }

    # Resolve which metadata CSVs to use for this run.
    if confident:
        if confident_output_subdir is None:
            confident_dir = settings['base_path']
        else:
            confident_dir = settings['work_dir'] / confident_output_subdir
        conf_paths = build_confident_csvs(
            settings['base_path'],
            agreement_threshold=confident_threshold,
            test_fraction=confident_test_fraction,
            seed=confident_seed,
            force_rebuild=confident_force_rebuild,
            output_dir=confident_dir,
        )
        settings['train_csv_path'] = conf_paths['train_csv']
        settings['test_csv_path'] = conf_paths['test_csv']
        settings['confident_csv_dir'] = confident_dir
    else:
        settings['train_csv_path'] = settings['base_path'] / 'train.csv'
        settings['test_csv_path'] = settings['base_path'] / 'test.csv'
        settings['confident_csv_dir'] = None

    settings['confident'] = bool(confident)
    settings['confident_threshold'] = float(confident_threshold)
    settings['confident_test_fraction'] = float(confident_test_fraction)
    settings['confident_seed'] = int(confident_seed)

    ensure_dirs([
        settings['work_dir'],
        settings['cache_dir'],
        settings['models_dir'],
        settings['results_dir'],
        settings['plots_dir'],
    ])

    return settings

# Reproducibility helpers.
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int, default=42
        Seed value used for Python, NumPy, and PyTorch.

    Returns
    -------
    None
        This function updates global random-number-generator state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism can reduce speed; keep benchmark disabled for stability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# Signal processing utilities.

def build_bipolar(window_df: pd.DataFrame, pairs: List[Tuple[str, str]], target_len: int) -> np.ndarray:
    """Build bipolar montage channels from a raw EEG window.

    Parameters
    ----------
    window_df : pandas.DataFrame
        EEG data frame for one extracted window.
    pairs : list of tuple of str
        Electrode pairs used to create bipolar channels.
    target_len : int
        Number of expected time samples in the returned signal.

    Returns
    -------
    numpy.ndarray
        Bipolar signal array of shape ``(len(pairs), target_len)``.
    """
    cols = set(window_df.columns)
    out = []
    for a, b in pairs:
        if a in cols and b in cols:
            sig = window_df[a].values.astype(np.float32) - window_df[b].values.astype(np.float32)
        else:
            sig = np.zeros(target_len, dtype=np.float32)
        out.append(sig)
    return np.stack(out, axis=0)


def bandpass_filter(data: np.ndarray, low_hz: float, high_hz: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filtering channel-wise.

    Parameters
    ----------
    data : numpy.ndarray
        Input array of shape ``(channels, time)``.
    low_hz : float
        Lower cutoff frequency in Hz.
    high_hz : float
        Upper cutoff frequency in Hz.
    fs : int
        Sampling rate in Hz.
    order : int, default=4
        Butterworth filter order.

    Returns
    -------
    numpy.ndarray
        Filtered array with same shape as input.
    """
    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    out = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        if np.std(data[i]) < 1e-8:
            continue
        try:
            out[i] = sosfiltfilt(sos, data[i]).astype(np.float32)
        except ValueError:
            out[i] = data[i].astype(np.float32)
    return out


def notch_filter(data: np.ndarray, fs: int, notch_hz: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Apply notch filtering channel-wise.

    Parameters
    ----------
    data : numpy.ndarray
        Input array of shape ``(channels, time)``.
    fs : int
        Sampling rate in Hz.
    notch_hz : float, default=60.0
        Center frequency of the notch filter.
    q : float, default=30.0
        Quality factor of the notch filter.

    Returns
    -------
    numpy.ndarray
        Notch-filtered array with same shape as input.
    """
    b, a = iirnotch(notch_hz, q, fs)
    out = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        if np.std(data[i]) < 1e-8:
            continue
        try:
            out[i] = filtfilt(b, a, data[i]).astype(np.float32)
        except ValueError:
            out[i] = data[i].astype(np.float32)
    return out


def resample_signal(data: np.ndarray, src_fs: int, target_fs: int) -> np.ndarray:
    """Resample multi-channel signals using polyphase resampling.

    Parameters
    ----------
    data : numpy.ndarray
        Input array of shape ``(channels, time)``.
    src_fs : int
        Original sampling rate.
    target_fs : int
        Target sampling rate.

    Returns
    -------
    numpy.ndarray
        Resampled array of shape ``(channels, new_time)``.
    """
    if src_fs == target_fs:
        return data.astype(np.float32)
    return resample_poly(data, up=target_fs, down=src_fs, axis=1).astype(np.float32)


def extract_50s_window_by_offset(eeg_df: pd.DataFrame, offset_seconds: float, cfg: CFG) -> pd.DataFrame:
    """Extract a fixed-length window from an EEG dataframe using offset.

    Parameters
    ----------
    eeg_df : pandas.DataFrame
        Full EEG recording dataframe.
    offset_seconds : float
        Offset from start in seconds.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    pandas.DataFrame
        Data frame containing exactly ``cfg.window_seconds * cfg.src_sample_rate`` rows,
        padded with zeros if needed.
    """
    full_len = cfg.window_seconds * cfg.src_sample_rate
    start = int(max(0, round(float(offset_seconds) * cfg.src_sample_rate)))
    end = start + full_len
    window = eeg_df.iloc[start:end].copy()

    if len(window) < full_len:
        pad = pd.DataFrame(
            np.zeros((full_len - len(window), eeg_df.shape[1]), dtype=np.float32),
            columns=eeg_df.columns,
        )
        window = pd.concat([window, pad], ignore_index=True)
    elif len(window) > full_len:
        window = window.iloc[:full_len].copy()

    return window


def preprocess_row_to_array(
    row: pd.Series,
    cfg,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> np.ndarray:
    """Convert one metadata row into a normalized bipolar EEG tensor.

    Parameters
    ----------
    row : pandas.Series
        Metadata row containing at least ``eeg_path`` and optionally offset fields.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    numpy.ndarray
        Preprocessed signal array of shape ``(cfg.num_bipolar_channels, cfg.window_seconds * cfg.target_sample_rate)``.
    """
    eeg_df = pd.read_parquet(row['eeg_path'])

    offset = float(row['eeg_label_offset_seconds']) if 'eeg_label_offset_seconds' in row and not pd.isna(row['eeg_label_offset_seconds']) else 0.0
    window = extract_50s_window_by_offset(eeg_df, offset, cfg)

    use_pairs = pairs if pairs is not None else BIPOLAR_PAIRS
    x = build_bipolar(window, use_pairs, cfg.window_seconds * cfg.src_sample_rate)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1024.0, 1024.0)

    x = bandpass_filter(x, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.src_sample_rate, cfg.bandpass_order)
    if cfg.apply_notch:
        x = notch_filter(x, cfg.src_sample_rate, cfg.notch_freq_hz, cfg.notch_q)

    x = resample_signal(x, cfg.src_sample_rate, cfg.target_sample_rate)

    # Robust channel-wise normalization.
    med = np.median(x, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    iqr = q75 - q25
    x = (x - med) / (iqr + 1e-6)
    x = np.clip(x, -8.0, 8.0).astype(np.float32)

    return x



# Cached preprocessing utilities.

def row_cache_key(row: pd.Series, split: str) -> str:
    """Create a stable cache key for one row.

    Parameters
    ----------
    row : pandas.Series
        Input metadata row.
    split : str
        One of ``'train'``, ``'valid'``, ``'test'``.

    Returns
    -------
    str
        Stable cache key string.
    """
    if split == 'test':
        return f"eeg_{int(row['eeg_id'])}"

    if 'label_id' in row and not pd.isna(row['label_id']):
        return f"label_{int(row['label_id'])}"

    eeg_id = int(row['eeg_id'])
    sub_id = int(row['eeg_sub_id']) if 'eeg_sub_id' in row and not pd.isna(row['eeg_sub_id']) else -1
    off = int(round(float(row['eeg_label_offset_seconds']))) if 'eeg_label_offset_seconds' in row and not pd.isna(row['eeg_label_offset_seconds']) else 0
    return f"eeg_{eeg_id}_sub_{sub_id}_off_{off}"


def cache_split_dir(split: str, cfg: CFG) -> Path:
    """Return split-specific cache directory path.

    Parameters
    ----------
    split : str
        One of ``'train'``, ``'valid'``, ``'test'``.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    pathlib.Path
        Absolute cache directory path.
    """
    if split in {'train', 'valid'}:
        return cfg.CACHE_DIR / f"{split}_fold{cfg.fold}"
    return cfg.CACHE_DIR / 'test'


def prepare_cache(df: pd.DataFrame, split: str, cfg: CFG, force_rebuild: bool = False) -> List[Path]:
    """Preprocess and cache one dataframe split into NPZ files.

    Parameters
    ----------
    df : pandas.DataFrame
        Input metadata table for one split.
    split : str
        One of ``'train'``, ``'valid'``, ``'test'``.
    cfg : CFG
        Runtime configuration.
    force_rebuild : bool, default=False
        If True, rebuild cache files even when they already exist.

    Returns
    -------
    list of pathlib.Path
        Paths to cache files corresponding to rows in the input dataframe order.
    """
    split_dir = cache_split_dir(split, cfg)
    split_dir.mkdir(parents=True, exist_ok=True)

    cache_files: List[Path] = []
    created = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Caching {split}'):
        key = row_cache_key(row, split)
        fp = split_dir / f"{key}.npz"
        cache_files.append(fp)

        if fp.exists() and not force_rebuild:
            skipped += 1
            continue

        x = preprocess_row_to_array(row, cfg)

        payload = {
            'x': x.astype(np.float32),
            'eeg_id': np.int64(row['eeg_id']),
        }

        if split != 'test':
            y = np.array(row['soft_labels'], dtype=np.float32)
            votes = np.float32(row['total_votes'])
            payload['y'] = y
            payload['votes'] = votes

        np.savez_compressed(fp, **payload)
        created += 1

    print(f'{split}: total={len(df)}, created={created}, skipped={skipped}, dir={split_dir}')
    return cache_files


class NPZEEGDataset(Dataset):
    """Dataset backed by cached NPZ files."""
    def __init__(self, cache_files: List[Path], split: str, augment: bool = False, cfg=None):
        """Initialize dataset state.

        Parameters
        ----------
        cache_files : list of pathlib.Path
            Paths to NPZ samples.
        split : str
            Split name, one of ``'train'``, ``'valid'``, or ``'test'``.
        augment : bool, default=False
            Whether to apply sample-level augmentation.
        cfg : object, optional
            Runtime configuration object.

        Returns
        -------
        None
            This initializer stores dataset metadata for indexed access.
        """
        self.cache_files = [Path(p) for p in cache_files]
        self.split = split
        self.augment = augment
        self.cfg = cfg
        self.target_sample_rate = (
            int(getattr(cfg, 'target_sample_rate', 100)) if cfg is not None else 100
        )
        self.eeg_ids = []
        # Only test split requires stable eeg_ids for submission/prediction merge.
        if self.split == 'test':
            for fp in self.cache_files:
                eeg_id = self._resolve_eeg_id(fp)
                self.eeg_ids.append(int(eeg_id))

    @staticmethod
    def _infer_eeg_id_from_filename(fp: Path) -> Optional[int]:
        """Infer EEG ID from cache filename when possible.

        Parameters
        ----------
        fp : pathlib.Path
            NPZ cache file path.

        Returns
        -------
        int or None
            Parsed EEG ID when filename follows ``eeg_<id>`` convention, else None.
        """
        stem = fp.stem
        if stem.startswith('eeg_'):
            token = stem.split('_')[1]
            if token.isdigit():
                return int(token)
        return None

    def _resolve_eeg_id(self, fp: Path) -> int:
        """Resolve EEG ID from NPZ payload or filename fallback.

        Parameters
        ----------
        fp : pathlib.Path
            NPZ cache file path.

        Returns
        -------
        int
            EEG identifier.
        """
        try:
            with np.load(fp) as z:
                if 'eeg_id' in z:
                    return int(z['eeg_id'])
        except Exception:
            pass

        inferred = self._infer_eeg_id_from_filename(fp)
        if inferred is not None:
            return int(inferred)
        return -1

    def __len__(self) -> int:
        """Return dataset length.

        Parameters
        ----------
        None
            Length is computed from cached file paths stored on the instance.

        Returns
        -------
        int
            Number of cached files available in this dataset.
        """
        return len(self.cache_files)

    def _augment_x(self, x: np.ndarray) -> np.ndarray:
        """Apply lightweight augmentations to normalized EEG.

        Parameters
        ----------
        x : numpy.ndarray
            Input EEG array of shape ``(channels, time)``.

        Returns
        -------
        numpy.ndarray
            Augmented array with same shape.
        """
        return augment_sample_np(x, self.cfg, self.target_sample_rate)

    def __getitem__(self, idx: int):
        """Load one sample from cache.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            For train/valid: ``(x, y, votes)`` tensors.
            For test: ``(x, eeg_id)``.
        """
        fp = self.cache_files[idx]
        with np.load(fp) as z:
            x = z['x'].astype(np.float32)
            if 'eeg_id' in z:
                eeg_id = int(z['eeg_id'])
            else:
                eeg_id = self._resolve_eeg_id(fp) if self.split == 'test' else -1
            if self.augment:
                x = self._augment_x(x)
            x_t = torch.tensor(x, dtype=torch.float32)
            if self.split == 'test':
                return x_t, eeg_id
            y = z['y'].astype(np.float32)
            votes = np.float32(z['votes'])
            y_t = torch.tensor(y, dtype=torch.float32)
            v_t = torch.tensor(votes, dtype=torch.float32)
            return x_t, y_t, v_t


def build_dataloaders(df_train: pd.DataFrame, df_valid: pd.DataFrame, cfg) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders for train and validation splits.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training split metadata.
    df_valid : pandas.DataFrame
        Validation split metadata.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        ``(train_loader, valid_loader)``.
    """
    train_files = prepare_cache(df_train, split='train', cfg=cfg, force_rebuild=cfg.force_rebuild_cache)
    valid_files = prepare_cache(df_valid, split='valid', cfg=cfg, force_rebuild=cfg.force_rebuild_cache)

    train_ds = NPZEEGDataset(train_files, split='train', augment=True, cfg=cfg)
    valid_ds = NPZEEGDataset(valid_files, split='valid', augment=False, cfg=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=build_train_collate_fn(cfg),
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
    return train_loader, valid_loader


# Model definitions: Conv1D front-end + BiGRU + attention pooling.
class ConvBlock1D(nn.Module):
    """Conv1D block used for temporal reduction."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, dropout: float):
        """Initialize one temporal Conv1D block.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        kernel : int
            Convolution kernel size.
        stride : int
            Convolution stride.
        dropout : float
            Dropout probability.

        Returns
        -------
        None
            This initializer creates learnable layers.
        """
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass for one Conv1D block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, normalization, activation, and dropout.
        """
        return self.block(x)


class MultiScaleConvBlock1D(nn.Module):
    """Inception-style multi-scale temporal Conv1D block.

    Runs parallel Conv1D branches with different kernel sizes and concatenates
    their outputs. Large-kernel branches use depthwise-separable convolution to
    keep the parameter count modest (avoids the t11 regression caused by a
    large standard Conv1d in the first layer).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        dropout: float,
        kernels: tuple = (3, 15, 31),
    ):
        """Initialize multi-scale Conv1D block.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Total number of output channels (split evenly across branches;
            remainder absorbed into the last branch).
        stride : int
            Stride applied by every branch.
        dropout : float
            Dropout probability applied after concatenation.
        kernels : tuple of int, default=(3, 15, 31)
            Kernel size for each branch.  Kernels > 7 use depthwise-separable
            conv to avoid parameter explosion.

        Returns
        -------
        None
        """
        super().__init__()
        n_branches = len(kernels)
        branch_channels = [out_ch // n_branches] * n_branches
        # Absorb the integer remainder into the last branch so the total is
        # exactly out_ch regardless of divisibility.
        branch_channels[-1] = out_ch - sum(branch_channels[:-1])

        self.branches = nn.ModuleList()
        for k, bc in zip(kernels, branch_channels):
            if k <= 7:
                # Small kernel: standard Conv1d.
                branch = nn.Sequential(
                    nn.Conv1d(in_ch, bc, kernel_size=k, stride=stride,
                              padding=k // 2, bias=False),
                    nn.BatchNorm1d(bc),
                    nn.SiLU(inplace=True),
                )
            else:
                # Large kernel: depthwise (groups=in_ch) then pointwise 1x1.
                branch = nn.Sequential(
                    nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride,
                              padding=k // 2, groups=in_ch, bias=False),
                    nn.Conv1d(in_ch, bc, kernel_size=1, bias=False),
                    nn.BatchNorm1d(bc),
                    nn.SiLU(inplace=True),
                )
            self.branches.append(branch)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass for multi-scale Conv1D block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Concatenated branch outputs after dropout,
            shape ``(batch, out_ch, time // stride)``.
        """
        outs = [branch(x) for branch in self.branches]
        return self.drop(torch.cat(outs, dim=1))


class ConvBiGRUAttention(nn.Module):
    """Raw EEG classifier with temporal convolution, BiGRU, and attention pooling."""
    def __init__(self, cfg: CFG):
        """Initialize Conv1D + BiGRU + attention classifier.

        Parameters
        ----------
        cfg : CFG
            Runtime/model configuration.

        Returns
        -------
        None
            This initializer constructs model submodules.
        """
        super().__init__()
        conv_channels = list(cfg.conv_channels)
        conv_kernels = list(cfg.conv_kernels)
        conv_strides = list(cfg.conv_strides)

        if not (len(conv_channels) == len(conv_kernels) == len(conv_strides)):
            raise ValueError(
                'conv_channels, conv_kernels, and conv_strides must have the same length.'
            )

        use_multiscale = bool(getattr(cfg, 'use_multiscale_conv', False))
        ms_kernels = tuple(getattr(cfg, 'multiscale_kernels', (3, 15, 31)))

        conv_blocks = []
        in_ch = cfg.num_bipolar_channels
        for out_ch, kernel, stride in zip(conv_channels, conv_kernels, conv_strides):
            if use_multiscale:
                conv_blocks.append(
                    MultiScaleConvBlock1D(in_ch, out_ch, stride, cfg.dropout, kernels=ms_kernels)
                )
            else:
                conv_blocks.append(ConvBlock1D(in_ch, out_ch, kernel, stride, cfg.dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)

        gru_input_size = conv_channels[-1]

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )

        self.attn = nn.Linear(cfg.gru_hidden * 2, 1)

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gru_hidden * 2, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape ``(batch, num_classes)``.
        """
        z = self.conv(x)
        z = z.transpose(1, 2)
        z, _ = self.gru(z)
        w = torch.softmax(self.attn(z), dim=1)
        pooled = (z * w).sum(dim=1)
        logits = self.head(pooled)
        return logits


def build_confident_subset(
    df: pd.DataFrame,
    vote_cols: List[str],
    agreement_threshold: float = 0.70,
) -> pd.DataFrame:
    """Filter rows by top-class vote agreement threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing vote columns.
    vote_cols : list of str
        Vote column names.
    agreement_threshold : float, default=0.70
        Minimum ``max_vote / total_votes`` ratio.

    Returns
    -------
    pandas.DataFrame
        Confidence-filtered dataframe.
    """
    missing = [c for c in vote_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Input dataframe is missing vote columns: {missing}')

    votes = df[vote_cols].astype(np.float32).values
    total_votes = votes.sum(axis=1)
    max_votes = votes.max(axis=1)
    agreement = np.divide(
        max_votes,
        np.clip(total_votes, 1e-6, None),
        out=np.zeros_like(max_votes, dtype=np.float32),
        where=total_votes > 0,
    )
    keep = agreement >= float(agreement_threshold)
    out = df.loc[keep].copy().reset_index(drop=True)
    print(
        f'Confident subset built at threshold={agreement_threshold:.2f}: '
        f'{len(out)}/{len(df)} rows ({100.0 * len(out) / max(len(df), 1):.2f}%).'
    )
    return out


def cap_rows_per_eeg(
    df: pd.DataFrame,
    max_rows_per_eeg: int = 2,
    rule: str = 'top_total_votes',
) -> pd.DataFrame:
    """Cap training rows per EEG ID using a deterministic selection rule.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least ``eeg_id`` and ``total_votes``.
    max_rows_per_eeg : int, default=2
        Maximum number of rows retained per EEG.
    rule : str, default='top_total_votes'
        Selection rule. Currently supports only ``'top_total_votes'``.

    Returns
    -------
    pandas.DataFrame
        Deduplicated dataframe with stable row selection.
    """
    if int(max_rows_per_eeg) <= 0:
        raise ValueError(f'max_rows_per_eeg must be >= 1, got {max_rows_per_eeg}')
    if rule != 'top_total_votes':
        raise ValueError(f'Unsupported dedup rule: {rule}')

    required = ['eeg_id', 'total_votes']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Input dataframe is missing required columns: {missing}')

    work = df.reset_index(drop=False).rename(columns={'index': '_orig_row_order'}).copy()
    work = work.sort_values(
        by=['eeg_id', 'total_votes', '_orig_row_order'],
        ascending=[True, False, True],
        kind='mergesort',
    )

    capped = (
        work.groupby('eeg_id', sort=False, group_keys=False)
        .head(int(max_rows_per_eeg))
        .sort_values(by='_orig_row_order', kind='mergesort')
        .drop(columns=['_orig_row_order'])
        .reset_index(drop=True)
    )
    return capped


# Baseline comparator and initial model sanity checks.
def compute_global_prior(df_train: pd.DataFrame) -> np.ndarray:
    """Compute class prior from mean soft-label distribution.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataframe containing ``soft_labels``.

    Returns
    -------
    numpy.ndarray
        Prior probabilities with shape ``(num_classes,)``.
    """
    ys = np.stack(df_train['soft_labels'].values)
    prior = ys.mean(axis=0)
    prior = prior / np.clip(prior.sum(), 1e-8, None)
    return prior.astype(np.float32)


def kl_divergence_np(targets: np.ndarray, probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute per-sample KL divergence ``KL(target || probs)``.

    Parameters
    ----------
    targets : numpy.ndarray
        Target distribution array of shape ``(N, C)``.
    probs : numpy.ndarray
        Predicted distribution array of shape ``(N, C)``.
    eps : float, default=1e-8
        Numerical stability floor.

    Returns
    -------
    numpy.ndarray
        KL values of shape ``(N,)``.
    """
    t = np.clip(targets, eps, 1.0)
    p = np.clip(probs, eps, 1.0)
    return np.sum(t * (np.log(t) - np.log(p)), axis=1)


def compute_prior_baseline_kl(df_valid: pd.DataFrame, prior: np.ndarray) -> float:
    """Evaluate global-prior baseline KL on validation rows.

    Parameters
    ----------
    df_valid : pandas.DataFrame
        Validation dataframe containing ``soft_labels``.
    prior : numpy.ndarray
        Class prior probabilities of shape ``(C,)``.

    Returns
    -------
    float
        Mean validation KL divergence.
    """
    targets = np.stack(df_valid['soft_labels'].values)
    preds = np.repeat(prior[None, :], repeats=len(df_valid), axis=0)
    return float(kl_divergence_np(targets, preds).mean())



# Training, validation, and prediction loops.

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: CFG,
) -> Tuple[float, Dict[str, int]]:
    """Run one training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance.
    loader : torch.utils.data.DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scaler : torch.amp.GradScaler
        AMP scaler.
    device : torch.device
        Training device.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    tuple
        ``(mean_train_kl, augmentation_counters)`` for the epoch.
    """
    model.train()
    losses = []
    reset_augmentation_counters()

    use_vote_weighting = bool(getattr(cfg, 'run_a_use_vote_weighting', False))
    vote_weight_mode = str(getattr(cfg, 'run_a_vote_weight_mode', 'sqrt_norm'))

    for x, y, _votes in tqdm(loader, desc='Train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        votes = _votes.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            if use_vote_weighting:
                per_sample_kl = F.kl_div(log_probs, y, reduction='none').sum(dim=1)
                votes = votes.to(dtype=per_sample_kl.dtype)
                vote_base = torch.clamp(votes, min=1.0)

                if vote_weight_mode == 'sqrt_norm':
                    weights = torch.sqrt(vote_base)
                elif vote_weight_mode == 'linear_norm':
                    weights = vote_base
                else:
                    raise ValueError(
                        f'Unsupported vote weighting mode: {vote_weight_mode}. '
                        "Expected 'sqrt_norm' or 'linear_norm'."
                    )

                weights = weights / torch.clamp(weights.mean(), min=1e-6)
                loss = (weights * per_sample_kl).mean()
            else:
                loss = F.kl_div(log_probs, y, reduction='batchmean')

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))

    mean_loss = float(np.mean(losses)) if losses else float('nan')
    aug_stats = get_augmentation_counters()
    return mean_loss, aug_stats


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: CFG) -> float:
    """Evaluate model KL loss on validation data.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance.
    loader : torch.utils.data.DataLoader
        Validation loader.
    device : torch.device
        Evaluation device.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    float
        Mean validation KL loss.
    """
    model.eval()
    losses = []

    for x, y, _votes in tqdm(loader, desc='Valid', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, y, reduction='batchmean')

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, cfg: CFG) -> np.ndarray:
    """Run probabilistic inference.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : torch.utils.data.DataLoader
        Inference loader.
    device : torch.device
        Inference device.
    cfg : CFG
        Runtime configuration.

    Returns
    -------
    numpy.ndarray
        Predicted class probabilities with shape ``(N, C)``.
    """
    model.eval()
    outputs = []

    for batch in tqdm(loader, desc='Predict', leave=False):
        x = batch[0].to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        outputs.append(probs.detach().cpu().numpy())

    if not outputs:
        return np.empty((0, cfg.num_classes), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def save_live_training_artifacts(history: Dict[str, List[float]], cfg: CFG, checkpoint_name: str) -> None:
    """Persist live training artifacts after each epoch.

    Parameters
    ----------
    history : dict
        Training history dictionary with keys ``train_kl``, ``valid_kl``, and ``lr``.
    cfg : CFG
        Runtime configuration.
    checkpoint_name : str
        Checkpoint filename used to derive output artifact names.

    Returns
    -------
    None
        This function writes plot and CSV artifacts to disk.
    """
    stem = Path(checkpoint_name).stem
    run_dir = cfg.RESULTS_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    history_csv = run_dir / 'history.csv'
    live_plot = run_dir / 'live_training_curves.png'

    hist_df = pd.DataFrame({
        'epoch': np.arange(1, len(history['train_kl']) + 1),
        'train_kl': history['train_kl'],
        'valid_kl': history['valid_kl'],
        'lr': history['lr'],
    })
    hist_df.to_csv(history_csv, index=False)

    epochs = hist_df['epoch'].values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, history['train_kl'], marker='o', label='Train KL')
    ax[0].plot(epochs, history['valid_kl'], marker='o', label='Valid KL')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('KL Divergence')
    ax[0].set_title('Train vs Valid KL')
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, history['lr'], marker='o', color='tab:orange', label='Learning Rate')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Learning Rate')
    ax[1].set_title('LR Schedule')
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.suptitle(f'Live Training Curves: {stem}')
    fig.tight_layout()
    fig.savefig(live_plot, dpi=160, bbox_inches='tight')
    plt.close(fig)


def run_training(
    train_loader,
    valid_loader,
    cfg,
    baseline_kl,
    epochs,
    checkpoint_name,
    resume=True,
    resume_from: Optional[Path] = None,
    weights_only: bool = False,
    override_lr: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    stage_name: str = 'stage1',
    stage_metadata: Optional[Dict[str, Any]] = None,
    reset_tracking: bool = False,
):
    """Train model with validation, checkpointing, and optional resume.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    valid_loader : torch.utils.data.DataLoader
        Validation data loader.
    cfg : CFG
        Runtime configuration.
    baseline_kl : float
        Baseline KL value for reference/logging.
    epochs : int
        Maximum number of epochs to run.
    checkpoint_name : str
        Checkpoint filename under ``cfg.MODELS_DIR``.
    resume : bool, default=True
        Whether to resume from existing checkpoint.
    resume_from : pathlib.Path, optional
        Path to checkpoint used for warm-starting weights.
    weights_only : bool, default=False
        If True, only model weights are restored and optimizer state is reset.
    override_lr : float, optional
        If set, overrides ``cfg.lr`` for this call.
    early_stopping_patience : int, optional
        If set, overrides ``cfg.early_stopping_patience`` for this training call.
    stage_name : str, default='stage1'
        Stage tag used in artifact filenames and logs.
    stage_metadata : dict, optional
        Extra stage metadata written into run-config artifact.
    reset_tracking : bool, default=False
        If True (and ``weights_only=True``), reset history, best_val, and epoch
        numbering so the stage trains with a completely fresh tracking state.
        Model weights are still loaded from the checkpoint.

    Returns
    -------
    dict
        History dictionary with ``train_kl``, ``valid_kl``, and ``lr`` lists.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvBiGRUAttention(cfg).to(device)

    run_lr = float(override_lr) if override_lr is not None else float(cfg.lr)
    run_patience = (
        int(early_stopping_patience)
        if early_stopping_patience is not None
        else int(cfg.early_stopping_patience)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=run_lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    ckpt_path = cfg.MODELS_DIR / checkpoint_name

    history = {'train_kl': [], 'valid_kl': [], 'lr': []}
    augmentation_history: List[Dict[str, Any]] = []
    best_val = float('inf')
    patience = 0
    start_epoch = 0

    total_epochs = int(epochs)

    # Resume block
    resume_path = Path(resume_from) if resume_from is not None else ckpt_path
    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt['model_state_dict']
        # torch.compile() prefixes keys with '_orig_mod.' — strip it so the
        # checkpoint is loadable into a plain (not-yet-compiled) model.
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if not weights_only:
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if cfg.use_amp and ckpt.get('scaler_state_dict') is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])

            history = ckpt.get('history', history)
            augmentation_history = ckpt.get('augmentation_history', augmentation_history)
            best_val = ckpt.get('best_valid_kl', best_val)
            start_epoch = ckpt.get('epoch', -1) + 1
            print(f"Resuming full state from epoch {start_epoch}/{total_epochs}")
        else:
            if reset_tracking:
                # True fine-tune: fresh tracking, model weights only
                start_epoch = 0
                total_epochs = int(epochs)
                print(
                    f'Loaded weights-only with tracking reset from {resume_path} | '
                    f'total_epochs={total_epochs}'
                )
            else:
                # Legacy warm-start: inherit history, append epochs
                history = ckpt.get('history', history)
                augmentation_history = ckpt.get('augmentation_history', augmentation_history)
                best_val = ckpt.get('best_valid_kl', best_val)
                start_epoch = len(history.get('train_kl', []))
                total_epochs = start_epoch + int(epochs)
                print(
                    f'Loaded weights-only warm start from {resume_path} | '
                    f'start_epoch={start_epoch} | total_target_epochs={total_epochs}'
                )

    # Optionally JIT-compile the model for faster forward/backward (PyTorch 2.x+).
    if getattr(cfg, 'use_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print('torch.compile() applied — first epoch will be slower (JIT compilation).')

    print(
        f"Stage={stage_name} | epochs={total_epochs} | lr={run_lr:.2e} | "
        f"baseline_kl={baseline_kl:.5f} | early_stopping_patience={run_patience}"
    )
    try:
        config_path = save_run_config_artifact(
            cfg=cfg,
            checkpoint_name=checkpoint_name,
            stage_name=stage_name,
            model=model,
            stage_metadata={
                'baseline_kl': float(baseline_kl),
                'epochs': int(total_epochs),
                'epochs_requested': int(epochs),
                'resume': bool(resume),
                'resume_from': str(resume_path) if resume else '',
                'weights_only': bool(weights_only),
                'override_lr': float(run_lr),
                'early_stopping_patience': int(run_patience),
                **(stage_metadata or {}),
            },
        )
        print('Saved run config:', config_path)
    except Exception as exc:
        print(f'Warning: run-config artifact save failed ({exc})')

    for ep in range(start_epoch, total_epochs):
        tr_kl, aug_counts = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        va_kl = validate(model, valid_loader, device, cfg)
        lr = optimizer.param_groups[0]['lr']

        history['train_kl'].append(tr_kl)
        history['valid_kl'].append(va_kl)
        history['lr'].append(lr)
        augmentation_history.append({
            'stage': stage_name,
            'epoch': int(ep + 1),
            **{k: int(v) for k, v in aug_counts.items()},
        })

        scheduler.step()

        print(
            f"Epoch {ep+1}/{total_epochs} | train_kl={tr_kl:.5f} | valid_kl={va_kl:.5f} | lr={lr:.2e} | "
            f"aug={aug_counts}"
        )

        improved = va_kl < best_val
        if improved:
            best_val = va_kl
            patience = 0
        else:
            patience += 1

        # Save full state every epoch (important for resume)
        state = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if cfg.use_amp else None,
            'best_valid_kl': best_val,
            'history': history,
            'augmentation_history': augmentation_history,
            'stage_name': stage_name,
        }
        torch.save(state, ckpt_path)

        # Save a separate best-epoch checkpoint whenever validation improves.
        if improved:
            best_ckpt_path = ckpt_path.parent / (ckpt_path.stem + '_best' + ckpt_path.suffix)
            torch.save(state, best_ckpt_path)

        # Persist a CSV + plot each epoch so long runs are easy to monitor.
        try:
            save_live_training_artifacts(history, cfg, checkpoint_name)
            save_augmentation_artifacts(augmentation_history, cfg, checkpoint_name)
        except Exception as exc:
            print(f'Warning: live artifact save failed ({exc})')

        if patience >= run_patience:
            print('Early stopping triggered.')
            break

    return history


# Loss and LR plotting.
def plot_history(history: Dict[str, List[float]], title: str, cfg: CFG, tag: str) -> None:
    """Plot and save training curves.

    Parameters
    ----------
    history : dict
        Training history containing ``train_kl``, ``valid_kl``, and ``lr``.
    title : str
        Figure title.
    cfg : CFG
        Runtime configuration.
    tag : str
        File tag for plot image name.

    Returns
    -------
    None
        This function saves and displays the training-curve figure.
    """
    epochs = np.arange(1, len(history['train_kl']) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, history['train_kl'], marker='o', label='Train KL')
    ax[0].plot(epochs, history['valid_kl'], marker='o', label='Valid KL')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('KL Divergence')
    ax[0].set_title('Train vs Valid KL')
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, history['lr'], marker='o', color='tab:orange', label='Learning Rate')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Learning Rate')
    ax[1].set_title('LR Schedule')
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.suptitle(title)
    fig.tight_layout()

    out_file = cfg.PLOTS_DIR / f'{tag}_training_curves.png'
    fig.savefig(out_file, dpi=160, bbox_inches='tight')
    plt.show()
    print('Saved plot:', out_file)
