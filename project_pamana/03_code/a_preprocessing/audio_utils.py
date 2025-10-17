# project_pamana/03_code/a_preprocessing/audio_utils.py

import librosa
import numpy as np
import os
import random
import soundfile as sf

# --- Global Constants for the Project ---
SAMPLE_RATE = 22050
TARGET_DURATION = 3.0

# --- 1. Preprocessing Function ---
def standardize_audio(file_path, target_duration=TARGET_DURATION, sample_rate=SAMPLE_RATE):
    """
    Loads an audio file, resamples it, and standardizes its length to a
    target duration by trimming or padding.
    """
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        target_num_samples = int(target_duration * sr)

        if len(signal) > target_num_samples:
            start_index = (len(signal) - target_num_samples) // 2
            end_index = start_index + target_num_samples
            standardized_signal = signal[start_index:end_index]
        elif len(signal) < target_num_samples:
            padding_needed = target_num_samples - len(signal)
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            standardized_signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        else:
            standardized_signal = signal
        return standardized_signal, sr
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

# --- 2. Augmentation Functions (with subtle settings) ---
def augment_pitch_shift(signal, sr):
    """
    Shifts the pitch of a signal by a VERY SMALL, random number of semitones.
    """
    n_steps = random.uniform(-0.5, 0.5)
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

def augment_time_stretch(signal):
    """
    Stretches the time of a signal by a VERY SMALL, random factor.
    """
    rate = random.uniform(0.95, 1.05)
    return librosa.effects.time_stretch(y=signal, rate=rate)

# --- 3. NEW FEATURE EXTRACTION FUNCTION ---
def extract_traditional_features(signal, sr):
    """
    Extracts a set of traditional audio features from a signal.

    For features calculated per frame, this function computes the mean and
    standard deviation across all frames to get a fixed-size feature vector.

    Args:
        signal (np.ndarray): The audio signal.
        sr (int): The sample rate of the signal.

    Returns:
        dict: A dictionary of extracted features, or None if an error occurs.
    """
    features = {}
    try:
        # a. MFCCs (20 coefficients) -> 40 features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_mean_{i}'] = np.mean(mfccs[i, :])
            features[f'mfcc_std_{i}'] = np.std(mfccs[i, :])

        # b. Spectral Centroid -> 2 features
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)

        # c. Chroma Features (12 chroma) -> 24 features
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        for i in range(chroma.shape[0]):
            features[f'chroma_mean_{i}'] = np.mean( chroma[i, :])
            features[f'chroma_std_{i}'] = np.std(chroma[i, :])

        # d. Zero-Crossing Rate -> 2 features
        zcr = librosa.feature.zero_crossing_rate(y=signal)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        return features

    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        return None

# --- Main Block for Testing ---
if __name__ == '__main__':
    # This block is for testing and will confirm the new function works
    print("--- Testing audio_utils.py ---")
    sr_test = 22050
    t = np.linspace(0., 3.0, int(sr_test * 3.0))
    amplitude = np.iinfo(np.int16).max * 0.5
    dummy_signal = amplitude * np.sin(2. * np.pi * 440. * t)
    print("\n--- Testing 'extract_traditional_features' ---")
    extracted_features = extract_traditional_features(dummy_signal, sr_test)
    if extracted_features:
        expected_feature_count = 68
        actual_feature_count = len(extracted_features)
        print(f"   - Extracted {actual_feature_count} features (Expected: {expected_feature_count}).")
        if actual_feature_count == expected_feature_count: print("   - SUCCESS!")
        else: print("   - FAILED: Incorrect number of features.")
    else:
        print("   - FAILED: Feature extraction returned None.")
    print("\n--- Testing Complete ---")