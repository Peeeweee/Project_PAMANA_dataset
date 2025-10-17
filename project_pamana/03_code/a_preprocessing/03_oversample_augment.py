# project_pamana/03_code/a_preprocessing/03_oversample_augment.py

import os
import sys
import pandas as pd
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm

# --- Import Logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)

import config
from a_preprocessing.audio_utils import standardize_audio, augment_pitch_shift, augment_time_stretch, TARGET_DURATION, SAMPLE_RATE
import utils # To set random seeds

def oversample_minority_classes():
    """
    Performs oversampling on the TRAIN split using audio augmentation
    to ensure every class has exactly 80 samples.
    """
    utils.set_seeds()
    print("--- Starting Class Oversampling (Target: 80 samples) ---")

    # 1. Define paths
    logs_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics')
    input_csv_path = os.path.join(logs_dir, 'pamana_group_stratified_split_dataset.csv')
    output_folder = os.path.join(config.PROJECT_ROOT, '02_features', 'augmented_data')
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(logs_dir, 'pamana_master_augmented_splits.csv')
    
    # 2. Load the stratified split DataFrame
    print(f"Loading stratified data from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        print(f"ERROR: Input CSV not found. Please run 02_stratified_split.py first.")
        return
    df = pd.read_csv(input_csv_path)

    # 3. Isolate the TRAIN set
    train_df = df[df['split'] == 'TRAIN'].copy()
    if train_df.empty:
        print("No data in the 'TRAIN' split. Aborting oversampling.")
        return
        
    # EDITED: Hardcode the target size to 80.
    target_size = 80
    print(f"Target size for each class in TRAIN set has been set to: {target_size}")

    # 4. Implement Oversampling Loop
    augmented_samples_log = []
    # EDITED: Get all unique classes from the training set to check them for augmentation.
    all_classes = train_df['instrument_label'].unique()
    print(f"Checking all classes for oversampling needs: {all_classes}")

    for class_label in all_classes:
        class_df = train_df[train_df['instrument_label'] == class_label]
        current_class_size = len(class_df)
        samples_to_generate = target_size - current_class_size

        if samples_to_generate <= 0:
            print(f"Class '{class_label}' has {current_class_size} samples and does not need oversampling. Skipping.")
            continue
            
        print(f"Generating {samples_to_generate} new samples for class '{class_label}'...")
        
        # Create a lookup map from filepath to group for efficient access
        group_lookup = class_df.set_index('filepath')['group'].to_dict()
        original_class_files = class_df['filepath'].tolist()
        
        for i in tqdm(range(samples_to_generate), desc=f"Augmenting {class_label}"):
            original_path = random.choice(original_class_files)
            original_group = group_lookup.get(original_path)
            
            signal, sr = standardize_audio(original_path)
            
            if signal is None: continue

            if random.random() < 0.5:
                augmented_signal = augment_pitch_shift(signal, sr)
                augmentation_type = 'pitch_shift'
            else:
                augmented_signal = augment_time_stretch(signal)
                augmentation_type = 'time_stretch'

            target_num_samples = int(TARGET_DURATION * SAMPLE_RATE)
            if len(augmented_signal) > target_num_samples:
                augmented_signal = augmented_signal[:target_num_samples]
            elif len(augmented_signal) < target_num_samples:
                augmented_signal = np.pad(augmented_signal, (0, target_num_samples - len(augmented_signal)), 'constant')

            original_filename = os.path.basename(original_path)
            new_filename = f"aug_{augmentation_type}_{class_label}_{i}_{original_filename}"
            new_filepath = os.path.join(output_folder, new_filename)
            sf.write(new_filepath, augmented_signal, sr)
            
            augmented_samples_log.append({
                'filepath': new_filepath.replace('\\', '/'),
                'instrument_label': class_label,
                'split': 'TRAIN',
                'augmentation_type': augmentation_type,
                'original_file': original_path,
                'group': original_group
            })

    # 5. Final Export
    augmented_df = pd.DataFrame(augmented_samples_log)
    # Combine original data with the new augmented data
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully saved final master DataFrame to: {output_csv_path}")

    # 6. Print final class distribution for verification
    final_train_df = final_df[final_df['split'] == 'TRAIN']
    print("\n--- Final TRAIN Set Class Distribution After Oversampling ---")
    print(final_train_df['instrument_label'].value_counts().sort_index())
    print("-----------------------------------------------------------\n")
    print("--- Oversampling Complete ---")

if __name__ == '__main__':
    oversample_minority_classes()