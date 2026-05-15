import pandas as pd


##Read these values from the csv file
DEFAULT_METRICS = {
    "rms": {"privacy": 0.05, "utility": 0.70, "cost": 0.2},
    "zcr": {"privacy": 0.04, "utility": 0.62, "cost": 0.2},
    "spectral_centroid": {"privacy": 0.08, "utility": 0.78, "cost": 0.4},
    "spectral_flatness": {"privacy": 0.06, "utility": 0.65, "cost": 0.4},
    "spectral_contrast": {"privacy": 0.12, "utility": 0.82, "cost": 0.8},
    "spectral_spread": {"privacy": 0.07, "utility": 0.75, "cost": 0.4},
    "short_term_energy": {"privacy": 0.04, "utility": 0.68, "cost": 0.1},
    "temporal_centroid": {"privacy": 0.04, "utility": 0.61, "cost": 0.2},
    "silence_ratio": {"privacy": 0.03, "utility": 0.55, "cost": 0.1},
    "low_band_energy": {"privacy": 0.06, "utility": 0.73, "cost": 0.3},
    "mid_band_energy": {"privacy": 0.06, "utility": 0.74, "cost": 0.3},
    "high_band_energy": {"privacy": 0.05, "utility": 0.72, "cost": 0.3},
    "lh1000": {"privacy": 0.06, "utility": 0.70, "cost": 0.3},
}


def get_metrics(feature_name=None):
    """
    Return privacy, utility, and cost scores.

    In the paper, these are computed from demographic leakage,
    task utility, and empirical latency measurements.
    Replace DEFAULT_METRICS with your final measured scores.
    """
    df = pd.DataFrame(DEFAULT_METRICS).T.reset_index()
    df = df.rename(columns={"index": "feature"})

    if feature_name is not None:
        return df[df["feature"] == feature_name].to_dict(orient="records")[0]

    return df
