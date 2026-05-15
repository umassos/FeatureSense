# FeatureSense
Privacy-aware audio features library


# FeatureSense

FeatureSense is a Python library for privacy-aware audio feature extraction for always-on, non-speech audio sensing applications.

Instead of exposing raw audio, FeatureSense extracts low-dimensional audio features designed to preserve downstream sensing utility while reducing speech and speaker-attribute leakage.

## Installation

```bash
git clone https://github.com/<your-username>/featuresense.git
cd featuresense
pip install -e .


## Basic Usage

'''from featuresense import extract_privacy_features

df = extract_privacy_features(
    audio="example.wav",
    window_size=0.5,
    hop_size=0.25,
    target_sr=16000,
)

print(df.head())'''



