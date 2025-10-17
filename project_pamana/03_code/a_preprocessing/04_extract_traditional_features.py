# project_pamana/03_code/a_preprocessing/04_extract_traditional_features.py

import os
import sys
import pandas as pd
from tqdm import tqdm

# --- Import Logic ---
# Add the '03_code' directory to Python's path to import our custom modules
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)

import config
# 1. Imports from audio_utils.py
from a_preprocessing.audio_utils import standardize_audio, extract_traditional_features

def process_all_audio_to_features():
    """
    This is the complete script to extract traditional ML features for the entire dataset.
    It loads the master list of files, processes each one, and saves the final
    feature set to a CSV file.
    """
    print("--- Starting Traditional Feature Extraction for Full Dataset ---")

    # Define input and output paths
    logs_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics')
    input_csv_path = os.path.join(logs_dir, 'pamana_master_augmented_splits.csv')
    
    output_dir = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features')
    os.makedirs(output_dir, exist_ok=True)
    # The output file as specified in the prompt
    output_csv_path = os.path.join(output_dir, 'traditional_features_clean.csv')

    # Load the master DataFrame
    if not os.path.exists(input_csv_path):
        print(f"ERROR: Master CSV not found at {input_csv_path}.")
        print("Please run 02_stratified_split.py and 03_oversample_augment.py first.")
        return
        
    df = pd.read_csv(input_csv_path)
    print(f"Found {len(df)} total audio file records to process.")

    all_features_list = []

    # 2. Processing Loop
    print("Beginning feature extraction loop...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Features"):
        file_path = row['filepath']
        
        # A. Standardize the audio
        signal, sr = standardize_audio(file_path)
        
        if signal is not None:
            # B. Extract the 68 traditional features
            features = extract_traditional_features(signal, sr)
            
            if features is not None:
                # C. Add essential metadata to the dictionary
                features['filepath'] = file_path
                features['instrument_label'] = row['instrument_label']
                features['split'] = row['split']
                features['group'] = row.get('group', 'unknown')
                
                # D. Append the combined dictionary to our master list
                all_features_list.append(features)

    if not all_features_list:
        print("No features were extracted. Please check the audio files and paths.")
        return

    # 3. Final Export
    print("\nLoop complete. Converting list of features to a DataFrame...")
    features_df = pd.DataFrame(all_features_list)
    
    # Reorder columns to have metadata first for readability
    metadata_cols = ['filepath', 'instrument_label', 'split', 'group']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    final_cols = metadata_cols + sorted(feature_cols)
    features_df = features_df[final_cols]

    # Save the final DataFrame to the specified CSV
    features_df.to_csv(output_csv_path, index=False)
    
    # Final confirmation message with shape
    print(f"\nSuccessfully saved the final feature dataset!")
    print(f" -> Location: {output_csv_path}")
    print(f" -> Shape: {features_df.shape} (Rows, Columns)")
    print(f"    - (Note: 68 feature columns + 3 metadata columns = {68+3} total columns)")
    print("\n--- Feature Extraction Complete ---")


if __name__ == '__main__':
    process_all_audio_to_features()