# project_pamana/03_code/b_traditional_ml/debug_leakage_audit.py

import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --- Import Logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def audit_data_leakage():
    """
    Performs an audit to check for data leakage in the cross-validation process.
    It verifies two main things:
    1. Group Integrity: No group is shared between the training and validation sets in any fold.
    2. Feature Duplication: No identical feature vectors exist across different groups.
    """
    print("--- Starting Data Leakage Audit ---")

    # 1. Load Data
    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')
    if not os.path.exists(features_path):
        print(f"ERROR: Feature file not found at {features_path}")
        print("Please run the preprocessing pipeline first (01 to 04).")
        return
        
    df = pd.read_csv(features_path)
    cv_df = df[df['split'].isin(['TRAIN', 'VALIDATION'])].copy()
    print(f"Loaded {len(cv_df)} samples for audit.")

    feature_cols = [col for col in df.columns if col not in ['filepath', 'instrument_label', 'split', 'group']]
    
    X = cv_df[feature_cols].values
    y = cv_df['instrument_label'].values
    groups = cv_df['group'].astype(str).values

    # --- Audit 1: Group Integrity in Cross-Validation ---
    print("\n--- Auditing Group Integrity in CV Folds ---")
    N_SPLITS = 5
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED)
    leakage_found = False

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        
        intersection = train_groups.intersection(val_groups)
        
        if len(intersection) > 0:
            print(f"[FAIL] Fold {fold+1}: Leakage detected! {len(intersection)} groups found in both train and validation sets.")
            print(f"Leaked groups: {intersection}")
            leakage_found = True
        else:
            print(f"[PASS] Fold {fold+1}: No group overlap found between train and validation sets.")

    if not leakage_found:
        print("\nSUCCESS: Group integrity is maintained across all CV folds. No leakage detected.")
    else:
        print("\nFAILURE: Group leakage detected. The data splitting logic needs to be re-evaluated.")

    # --- Audit 2: Feature Duplication ---
    print("\n--- Auditing for Duplicate Feature Vectors Across Different Groups ---")
    
    # Create a DataFrame with features and group for checking duplicates
    features_and_groups_df = cv_df[feature_cols + ['group']]
    
    # Find all rows that are duplicates based on feature columns
    duplicated_features = features_and_groups_df[features_and_groups_df.duplicated(subset=feature_cols, keep=False)]

    if duplicated_features.empty:
        print("[PASS] No duplicate feature vectors found in the dataset.")
    else:
        # Group the duplicated features and check if any have more than one unique group
        leaky_duplicates = 0
        for _, group_df in duplicated_features.groupby(feature_cols):
            if group_df['group'].nunique() > 1:
                print("[FAIL] Found duplicate features assigned to multiple groups:")
                print(group_df)
                leaky_duplicates += 1
        
        if leaky_duplicates == 0:
            print("[PASS] Found duplicate feature vectors, but they all belong to the same respective groups.")
            print("         This is acceptable and can happen with identical short audio clips from the same recording.")
        else:
            print(f"\nFAILURE: Found {leaky_duplicates} instances of duplicate features across different groups.")

    print("\n--- Audit Complete ---")

if __name__ == '__main__':
    audit_data_leakage()
