# project_pamana/03_code/b_traditional_ml/debug_noise_robustness_test.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- CORRECTED IMPORT LOGIC ---
# Get the absolute path of the directory containing this script (b_traditional_ml)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (03_code)
code_dir = os.path.dirname(script_dir)
# Add the '03_code' directory to Python's path
sys.path.insert(0, code_dir)
import config

def add_noise_to_features(features, noise_level=0.1):
    """
    Adds Gaussian noise to a set of numerical features.
    """
    np.random.seed(config.SEED)
    noise = np.random.normal(0, noise_level, features.shape)
    return features + noise

def test_noise_robustness():
    """
    Loads the trained Random Forest model and tests its performance on the
    validation set after adding various levels of noise to the features.
    """
    print("--- Debugging: Noise Robustness Test ---")

    # 1. Load the trained model
    model_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'models')
    model_path = os.path.join(model_dir, 'random_forest_clean.pkl')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please run the training script (06_train_random_forest.py) first.")
        return
        
    rf_model = joblib.load(model_path)
    print("Loaded pre-trained Random Forest model.")

    # 2. Load the feature dataset
    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')
    df = pd.read_csv(features_path)
    
    val_df = df[df['split'] == 'VALIDATION'].copy()
    feature_cols = [col for col in df.columns if col not in ['filepath', 'instrument_label', 'split']]
    
    X_val = val_df[feature_cols].values
    y_val_str = val_df['instrument_label'].values

    le = LabelEncoder()
    le.fit(df['instrument_label'].unique())
    y_val = le.transform(y_val_str)

    # 3. Perform tests with increasing levels of noise
    noise_levels_to_test = [0.0, 0.01, 0.05, 0.1, 0.25]

    print("\n--- Stress Test Results ---")
    for noise_level in noise_levels_to_test:
        print(f"\n--- Testing with Noise Level: {noise_level} ---")
        
        X_val_noisy = add_noise_to_features(X_val.copy(), noise_level)
        y_pred = rf_model.predict(X_val_noisy)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"  -> Accuracy: {accuracy * 100:.2f}%")

    print("\n--- Interpretation ---")
    print("If accuracy drops gracefully (e.g., 100% -> 95% -> 80%), the model is somewhat robust.")
    print("If accuracy plummets immediately (e.g., 100% -> 20%), the model is 'brittle' and heavily reliant on the perfectly clean data.")
    print("\n--- Debugging Complete ---")

if __name__ == '__main__':
    test_noise_robustness()