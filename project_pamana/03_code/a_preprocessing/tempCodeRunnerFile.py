# project_pamana/03_code/a_preprocessing/05_generate_spectrograms.py

import os
import sys
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Import Logic for custom modules ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)

import config
from a_preprocessing.audio_utils import standardize_audio, SAMPLE_RATE

# --- UTILITY FUNCTION (CORRECTED) ---
def generate_mel_spectrogram(signal, sr, n_mels=128, target_width=128):
    """
    Generates a fixed-size Mel Spectrogram from an audio signal.
    """
    try:
        total_samples = len(signal)
        hop_length = int(total_samples / (target_width - 1))
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, hop_length=hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        normalized_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        current_width = normalized_spec.shape[1]
        if current_width > target_width:
            final_spec = normalized_spec[:, :target_width]
        elif current_width < target_width:
            padding_width = target_width - current_width
            final_spec = np.pad(normalized_spec, ((0, 0), (0, padding_width)), mode='constant')
        else:
            # THIS IS THE CORRECTED LINE
            final_spec = normalized_spec
            
        return final_spec

    except Exception as e:
        # This will now only catch unexpected errors, not the bug from before.
        print(f"Error generating spectrogram: {e}")
        return None


# --- MAIN PROCESSING LOGIC (No changes needed here) ---
def generate_all_spectrograms():
    print("--- Starting Spectrogram Generation (NPY and PNG) ---")

    # Paths
    logs_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics')
    input_csv_path = os.path.join(logs_dir, 'pamana_master_augmented_splits.csv')
    output_npy_dir = os.path.join(config.PROJECT_ROOT, '02_features', 'deep_learning_inputs', 'spectrogram_arrays_npy')
    output_png_dir = os.path.join(config.PROJECT_ROOT, '02_features', 'deep_learning_inputs', 'spectrogram_images_png')
    output_metadata_csv = os.path.join(config.PROJECT_ROOT, '02_features', 'deep_learning_inputs', 'spectrogram_metadata.csv')

    df = pd.read_csv(input_csv_path)
    print(f"Found {len(df)} audio records to process.")

    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(output_png_dir, exist_ok=True)
    
    spectrogram_records = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Spectrograms"):
        filepath = row['filepath']
        label = row['instrument_label']
        split = row['split']

        signal, sr = standardize_audio(filepath)
        if signal is None: continue

        spec_array = generate_mel_spectrogram(signal, sr)
        if spec_array is None: continue
            
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Save .npy file
        npy_split_dir = os.path.join(output_npy_dir, split)
        os.makedirs(npy_split_dir, exist_ok=True)
        npy_filepath = os.path.join(npy_split_dir, f"{base_name}.npy")
        np.save(npy_filepath, spec_array.astype(np.float32))

        # Save .png file
        png_split_dir = os.path.join(output_png_dir, split)
        os.makedirs(png_split_dir, exist_ok=True)
        png_filepath = os.path.join(png_split_dir, f"{base_name}.png")
        fig, ax = plt.subplots(figsize=(1, 1), dpi=128)
        ax.imshow(spec_array, cmap='magma', origin='lower', aspect='auto')
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.savefig(png_filepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Record metadata
        spectrogram_records.append({
            'original_filepath': filepath,
            'spectrogram_filepath': npy_filepath.replace('\\', '/'),
            'instrument_label': label,
            'split': split
        })

    # EXPORT
    print("\nSpectrogram generation complete.")
    metadata_df = pd.DataFrame(spectrogram_records)
    metadata_df.to_csv(output_metadata_csv, index=False)
    
    print(f"Successfully generated {len(metadata_df)} spectrogram files (.npy and .png).")
    print(f" -> .npy arrays (for model training) are in: '{output_npy_dir}'")
    print(f" -> .png images (for viewing) are in: '{output_png_dir}'")
    print(f"Metadata (pointing to .npy files) saved to: {output_metadata_csv}")
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    generate_all_spectrograms()