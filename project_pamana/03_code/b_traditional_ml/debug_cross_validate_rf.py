# project_pamana/03_code/b_traditional_ml/debug_cross_validate_rf.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- CORRECTED IMPORT LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up ONE level to the '03_code' directory
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def cross_validate_model():
    """
    Performs Stratified K-Fold Cross-Validation on the training data.
    """
    print("--- 'Lucky Split' Audit: Performing 5-Fold Cross-Validation ---")

    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')
    df = pd.read_csv(features_path)
    
    train_df = df[df['split'] == 'TRAIN'].copy()
    print(f"Using {len(train_df)} samples from the TRAIN split for cross-validation.")

    feature_cols = [col for col in df.columns if col not in ['filepath', 'instrument_label', 'split']]
    X = train_df[feature_cols].values
    y_str = train_df['instrument_label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    model = RandomForestClassifier(n_estimators=200, random_state=config.SEED, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    print("\nRunning cross-validation...")
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print("\n--- Audit Results ---")
    print(f"Scores for each of the 5 folds: {np.round(scores, 4)}")
    print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: {scores.std():.4f}")

    print("\n--- Interpretation ---")
    if scores.mean() > 0.98 and scores.std() < 0.02:
        print("✓ RESULT: The model's performance is consistently high and stable.")
        print("This strongly suggests that the 100% score was not a fluke, and the feature set is very effective.")
    else:
        print("! RESULT: The model's performance is inconsistent across folds.")
        print("This might indicate that the initial 100% score was a 'lucky split' and the model is not as robust as it seemed.")
    
    print("\n--- Audit Complete ---")

if __name__ == '__main__':
    cross_validate_model()