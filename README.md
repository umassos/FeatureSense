# FeatureSense

FeatureSense is a Python library for privacy-aware audio feature extraction for always-on, non-speech audio sensing applications.

Instead of exposing raw audio, FeatureSense extracts low-dimensional audio features designed to preserve downstream sensing utility while reducing speech and speaker-attribute leakage.

## Installation

```bash
git clone https://github.com/umassos/featuresense.git
cd featuresense
pip install -e .
```


## Basic Usage

```from featuresense import extract_privacy_features
df = extract_privacy_features(
    audio="example.wav",
    window_size=0.5,
    hop_size=0.25,
    target_sr=16000,
)
print(df.head())
```

## Select Specific Features

``` features = ["rms", "zcr", "spectral_centroid", "spectral_entropy"]
df = extract_privacy_features(
    audio="example.wav",
    feature_list=features,
    window_size=0.5,
    hop_size=0.25,
)
```


## Optimize Features for Privacy-Utility-Cost Tradeoff

``` from featuresense import optimize_features_with_puc_tradeoff
selected = optimize_features_with_puc_tradeoff(
    latency=5.0,
    alpha=0.5,
)

print(selected)
```


## Get Feature Metrics

```from featuresense import get_metrics
metrics = get_metrics()
print(metrics)
```



## `requirements.txt`

```text
numpy
pandas
scipy
librosa
PyWavelets
scikit-learn
```

