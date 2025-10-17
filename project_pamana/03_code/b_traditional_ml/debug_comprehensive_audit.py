# project_pamana/03_code/b_traditional_ml/debug_comprehensive_audit.py

import os
import sys
import pandas as pd
import numpy as np
import hashlib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- Import Logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def run_comprehensive_audit():
    """
    Performs a full audit of the data pipeline to investigate the 100% scores.
    1. Checks for duplicate original audio files.
    2. Checks for "leaky augmentation" (source files for training data appearing in val/test).
    3. Performs cross-validation to ensure the model's performance is stable.
    """
    print("--- Starting Comprehensive Data Pipeline Audit ---")

    # --- Load All Necessary Data Files ---
    initial_labels_path = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics', 'pamana_initial_labels.csv')
    master_augmented_path = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics', 'pamana_master_augmented_splits.csv')
    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')

    df_initial = pd.read_csv(initial_labels_path)
    df_master = pd.read_csv(master_augmented_path)
    df_features = pd.read_csv(features_path)

    # ======================================================================
    # CHECK 1: DUPLICATE FILE AUDIT
    # ======================================================================
    print("\n--- AUDIT 1: Checking for Duplicate Original Audio Files ---")
    file_hashes = {}
    duplicates_found = False
    for filepath in tqdm(df_initial['filepath'], desc="Hashing original files"):
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash in file_hashes:
                    duplicates_found = True
                    print(f"\n!!! DUPLICATE HASH FOUND: {filepath} is identical to {file_hashes[file_hash]}")
                else:
                    file_hashes[file_hash] = filepath
        except FileNotFoundError:
            continue
    if not duplicates_found:
        print("✓ RESULT: No duplicate audio files were found. This is a good sign.")
    else:
        print("✗ FAILED: Duplicate files found. This is a potential source of data leakage.")

    # ======================================================================
    # CHECK 2: LEAKY AUGMENTATION AUDIT
    # ======================================================================
    print("\n--- AUDIT 2: Checking for Leaky Augmentation ---")
    # Get the list of ORIGINAL files used to create augmented training data
    augmented_rows = df_master[df_master['augmentation_type'].notna()]
    source_files_for_training = set(augmented_rows['original_file'].unique())

    # Get the list of original files in the validation and test sets
    val_test_rows = df_master[df_master['split'].isin(['VALIDATION', 'TEST'])]
    files_in_val_test = set(val_test_rows['filepath'].unique())

    # Find the intersection - are there any overlaps?
    leakage_overlap = source_files_for_training.intersection(files_in_val_test)

    if not leakage_overlap:
        print("✓ RESULT: No leaky augmentation found.")
        print("   - The original files used for augmentation are NOT present in the validation or test sets.")
    else:
        print("✗ FAILED: Leaky augmentation detected! The following files are a problem:")
        for file in leakage_overlap:
            print(f"  - {file}")

    # ======================================================================
    # CHECK 3: CROSS-VALIDATION STABILITY AUDIT
    # ======================================================================
    print("\n--- AUDIT 3: Checking Model Stability with 5-Fold Cross-Validation ---")
    train_df = df_features[df_features['split'] == 'TRAIN'].copy()
    feature_cols = [col for col in df_features.columns if col not in ['filepath', 'instrument_label', 'split']]
    
    X = train_df[feature_cols].values
    y_str = train_df['instrument_label'].values
    y = LabelEncoder().fit_transform(y_str)

    model = RandomForestClassifier(n_estimators=200, random_state=config.SEED, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    print("Running cross-validation on the full training set...")
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print(f"\nScores for each fold: {np.round(scores, 4)}")
    print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: {scores.std():.4f}")

    if scores.mean() > 0.98:
        print("✓ RESULT: Cross-validation shows consistently high performance.")
    else:
        print("✗ FAILED: Cross-validation scores are lower than expected, suggesting the initial split was lucky.")

    # ======================================================================
    # FINAL CONCLUSION
    # ======================================================================
    print("\n\n--- FINAL AUDIT CONCLUSION ---")
    if not duplicates_found and not leakage_overlap and scores.mean() > 0.98:
        print("✓ All checks passed. The data pipeline appears to be free of leaks.")
        print("The 100% scores are very likely legitimate and are a result of a highly effective")
        print("feature set and a clean, well-separated dataset.")
    else:
        print("✗ One or more audit checks failed. Review the results above to identify the source of the potential data leak.")
    print("--------------------------------")


if __name__ == '__main__':
    run_comprehensive_audit()