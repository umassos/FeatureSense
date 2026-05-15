import numpy as np
import librosa
from scipy.stats import kurtosis, skew


EPS = 1e-12


def amplitude_envelope(y, sr):
    return float(np.max(np.abs(y)))


def rms(y, sr):
    return float(librosa.feature.rms(y=y).mean())


def zcr(y, sr):
    return float(librosa.feature.zero_crossing_rate(y).mean())


def short_term_energy(y, sr):
    return float(np.mean(y ** 2))


def temporal_centroid(y, sr):
    energy = y ** 2
    total_energy = np.sum(energy)
    if total_energy <= EPS:
        return 0.0
    return float(np.sum(np.arange(len(y)) * energy) / total_energy)


def envelope_modulation_rate(y, sr):
    envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    if len(envelope) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(envelope))))


def silence_ratio(y, sr, threshold=0.01):
    return float(np.mean(np.abs(y) < threshold))


def spectral_centroid(y, sr):
    return float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())


def spectral_flatness(y, sr):
    return float(librosa.feature.spectral_flatness(y=y).mean())


def spectral_contrast(y, sr):
    return float(librosa.feature.spectral_contrast(y=y, sr=sr).mean())


def spectral_spread(y, sr):
    return float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())


def spectral_entropy(y, sr):
    ps = np.abs(np.fft.rfft(y)) ** 2
    ps = ps / (np.sum(ps) + EPS)
    return float(-np.sum(ps * np.log2(ps + EPS)))


def spectral_irregularity(y, sr):
    stft = np.abs(librosa.stft(y))
    return float(np.sum(np.abs(np.diff(stft, axis=0))))


def mean(y, sr):
    return float(np.mean(y))


def variance(y, sr):
    return float(np.var(y))


def std(y, sr):
    return float(np.std(y))


def kurtosis_feature(y, sr):
    return float(kurtosis(y, nan_policy="omit"))


def skewness_feature(y, sr):
    return float(skew(y, nan_policy="omit"))


def entropy(y, sr):
    energy = y ** 2
    return float(-np.sum(energy * np.log2(energy + EPS)))


def sharpness(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    energy = librosa.feature.rms(y=y).mean()
    return float(centroid / (energy + EPS))


def reverberation(y, sr):
    ir = librosa.effects.harmonic(y)
    edc = np.cumsum(ir[::-1] ** 2)[::-1]
    if np.max(edc) <= EPS:
        return 0.0

    edc_db = 10 * np.log10(edc / (np.max(edc) + EPS) + EPS)
    start = np.where(edc_db <= -5)[0]
    end = np.where(edc_db <= -60)[0]

    if len(start) == 0 or len(end) == 0:
        return 0.0

    return float((end[0] - start[0]) / sr)


def group_delay(y, sr):
    phase = np.angle(librosa.stft(y))
    gd = -np.gradient(phase, axis=0)
    return float(np.mean(gd))


def wavelet_features(y, sr):
    import pywt
    coeffs = pywt.wavedec(y, "db4", level=4)
    return float(np.mean([np.std(c) for c in coeffs]))


def temporal_spectral_slope(y, sr):
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    if c.shape[1] < 2:
        return 0.0
    return float(np.mean(np.diff(c)))


def transient_to_sustained_ratio(y, sr, threshold=0.01):
    energy = librosa.feature.rms(y=y)
    transient = np.sum(energy[energy > threshold])
    sustained = np.sum(energy[energy <= threshold])
    return float(transient / (sustained + EPS))


def tonality_index(y, sr):
    stft = np.abs(librosa.stft(y))
    threshold = np.mean(stft)
    harmonic = np.sum(stft[stft > threshold])
    noise = np.sum(stft[stft <= threshold])
    return float(harmonic / (noise + EPS))


def spectral_roughness(y, sr):
    stft = np.abs(librosa.stft(y))
    roughness = np.sum(np.abs(np.diff(stft, axis=0)), axis=0)
    return float(np.mean(roughness))


def spectral_texture(y, sr):
    return wavelet_features(y, sr)


def directional_spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    delta = librosa.feature.delta(centroid)
    return float(np.mean(delta))


def band_energies(y, sr):
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    low = np.sum(stft[freqs < 500, :])
    mid = np.sum(stft[(freqs >= 500) & (freqs < 2000), :])
    high = np.sum(stft[freqs >= 2000, :])

    return {
        "low_band_energy": float(low),
        "mid_band_energy": float(mid),
        "high_band_energy": float(high),
        "lh1000": float(low / (high + EPS)),
    }


FEATURE_FUNCTIONS = {
    "amplitude_envelope": amplitude_envelope,
    "rms": rms,
    "zcr": zcr,
    "short_term_energy": short_term_energy,
    "temporal_centroid": temporal_centroid,
    "envelope_modulation_rate": envelope_modulation_rate,
    "silence_ratio": silence_ratio,
    "spectral_centroid": spectral_centroid,
    "spectral_flatness": spectral_flatness,
    "spectral_contrast": spectral_contrast,
    "spectral_spread": spectral_spread,
    "spectral_entropy": spectral_entropy,
    "spectral_irregularity": spectral_irregularity,
    "mean": mean,
    "variance": variance,
    "std": std,
    "kurtosis": kurtosis_feature,
    "skewness": skewness_feature,
    "entropy": entropy,
    "sharpness": sharpness,
    "reverberation": reverberation,
    "group_delay": group_delay,
    "wavelet_features": wavelet_features,
    "temporal_spectral_slope": temporal_spectral_slope,
    "transient_to_sustained_ratio": transient_to_sustained_ratio,
    "tonality_index": tonality_index,
    "spectral_roughness": spectral_roughness,
    "spectral_texture": spectral_texture,
    "directional_spectral_features": directional_spectral_features,
}
