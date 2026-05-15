import numpy as np
import pandas as pd
import librosa

from .groups import get_feature_list
from .features import FEATURE_FUNCTIONS, band_energies


def extract_privacy_features(
    audio,
    sr=None,
    feature_list=None,
    window_size=0.5,
    hop_size=0.25,
    target_sr=16000,
    label=None,
    file_id=None,
):
    """
    Extract FeatureSense privacy-aware audio features.

    Parameters
    ----------
    audio : str or np.ndarray
        Audio file path or raw waveform.
    sr : int, optional
        Sampling rate if audio is a waveform.
    feature_list : list[str], optional
        Features to extract. If None, uses privacy-safe default set.
    window_size : float
        Window size in seconds.
    hop_size : float
        Hop size in seconds.
    target_sr : int
        Sampling rate for loading/resampling.
    label : optional
        Optional label added to output rows.
    file_id : optional
        Optional file identifier added to output rows.

    Returns
    -------
    pd.DataFrame
        Window-level feature table.
    """

    if isinstance(audio, str):
        y, sr = librosa.load(audio, sr=target_sr, mono=True)
        if file_id is None:
            file_id = audio
    else:
        if sr is None:
            raise ValueError("sr must be provided when audio is a waveform.")
        y = np.asarray(audio, dtype=np.float32)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    feature_list = feature_list or get_feature_list(include_gray_zone=False)

    window_length = int(window_size * sr)
    hop_length = int(hop_size * sr)

    if window_length <= 0 or hop_length <= 0:
        raise ValueError("window_size and hop_size must be positive.")

    rows = []

    for window_index, start in enumerate(range(0, len(y) - window_length + 1, hop_length)):
        segment = y[start:start + window_length]
        row = {
            "file_id": file_id,
            "window_index": window_index,
            "start_time": start / sr,
            "end_time": (start + window_length) / sr,
        }

        band_features_needed = {
            "low_band_energy", "mid_band_energy", "high_band_energy", "lh1000"
        }

        if any(f in feature_list for f in band_features_needed):
            row.update(band_energies(segment, sr))

        for name in feature_list:
            if name in band_features_needed:
                continue

            if name not in FEATURE_FUNCTIONS:
                raise ValueError(f"Unknown or unsupported feature: {name}")

            row[name] = FEATURE_FUNCTIONS[name](segment, sr)

        if label is not None:
            row["label"] = label

        rows.append(row)

    return pd.DataFrame(rows)
