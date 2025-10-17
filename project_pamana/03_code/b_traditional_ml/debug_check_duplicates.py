# project_pamana/03_code/b_traditional_ml/debug_check_duplicates.py

import os
import sys
import pandas as pd
import hashlib
from tqdm import tqdm

# --- CORRECTED IMPORT LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up ONE level to the '03_code' directory
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def find_duplicate_audio_files():
    """
    Scans the original audio files to check for exact duplicates.
    """
    print("--- Data Audit: Checking for Duplicate Audio Files ---")

    initial_labels_path = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics', 'pamana_initial_labels.csv')
    if not os.path.exists(initial_labels_path):
        print(f"ERROR: Initial labels file not found at {initial_labels_path}")
        return

    df = pd.read_csv(initial_labels_path)
    
    file_hashes = {}
    duplicates_found = False

    print(f"Scanning {len(df)} original audio files...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Hashing Files"):
        filepath = row['filepath']
        try:
            with open(filepath, 'rb') as f:
                file_content = f.read()
                file_hash = hashlib.sha256(file_content).hexdigest()

                if file_hash in file_hashes:
                    file_hashes[file_hash].append(filepath)
                else:
                    file_hashes[file_hash] = [filepath]
        except FileNotFoundError:
            print(f"Warning: File not found at {filepath}. Skipping.")
            continue

    print("\n--- Audit Results ---")
    for file_hash, filepaths in file_hashes.items():
        if len(filepaths) > 1:
            print("!!! DUPLICATE FOUND !!!")
            print(f"The following files are identical:")
            for fp in filepaths:
                print(f"  - {fp}")
            duplicates_found = True
            
    if not duplicates_found:
        print("✓ SUCCESS: No duplicate audio files were found in the original dataset.")
        print("This significantly reduces the chance of a hidden data leak.")
    
    print("\n--- Audit Complete ---")

if __name__ == '__main__':
    find_duplicate_audio_files()