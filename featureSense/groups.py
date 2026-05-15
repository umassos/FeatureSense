PRIVACY_SAFE_FEATURES = {
    "time_domain": [
        "amplitude_envelope", "rms", "zcr", "short_term_energy",
        "temporal_centroid", "envelope_modulation_rate", "silence_ratio"
    ],
    "spectral": [
        "spectral_centroid", "spectral_flatness", "spectral_contrast",
        "spectral_spread", "spectral_entropy", "spectral_irregularity"
    ],
    "statistical": [
        "mean", "variance", "std", "kurtosis", "skewness", "entropy"
    ],
    "perceptual": [
        "sharpness", "reverberation"
    ],
    "high_level": [
        "group_delay", "wavelet_features", "temporal_spectral_slope"
    ],
    "derived": [
        "transient_to_sustained_ratio", "tonality_index",
        "spectral_roughness", "spectral_texture",
        "directional_spectral_features", "low_band_energy",
        "mid_band_energy", "high_band_energy", "lh1000"
    ]
}

PRIVACY_RISKY_FEATURES = {
    "phonetic_linguistic": ["formants", "filter_bank", "lpcc", "mfcc"],
    "speaker_related": ["pitch", "timbre", "chroma_features"],
    "gray_zone_voice": ["hnr", "jitter", "shimmer"]
}


def get_feature_list(include_gray_zone: bool = False):
    features = []
    for group_features in PRIVACY_SAFE_FEATURES.values():
        features.extend(group_features)

    if include_gray_zone:
        features.extend(PRIVACY_RISKY_FEATURES["gray_zone_voice"])

    return features
