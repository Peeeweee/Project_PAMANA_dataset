# project_pamana/03_code/a_preprocessing/02_stratified_split.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CORRECTED IMPORT LOGIC ---
# Get the absolute path of the directory containing this script (a_preprocessing)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (03_code)
code_dir = os.path.dirname(script_dir)
# Add the '03_code' directory to Python's path
sys.path.insert(0, code_dir)

# Now that '03_code' is in the path, we can import 'config' directly.
import config

def create_stratified_splits():
    """
    Loads the initial labeled dataset and splits it into training, validation,
    and test sets using stratification to maintain the same class distribution
    in each split. The result is saved to a new CSV file.
    """
    print("--- Starting Stratified Dataset Splitting ---")

    # 1. Define paths and load the initial DataFrame
    # Note: We can now use the PROJECT_ROOT from the config file for more robust paths
    logs_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics')
    input_csv_path = os.path.join(logs_dir, 'pamana_initial_labels.csv')
    output_csv_path = os.path.join(logs_dir, 'pamana_stratified_split_dataset.csv')

    if not os.path.exists(input_csv_path):
        print(f"ERROR: Labels CSV not found at {input_csv_path}. Please run 01_label_data.py first.")
        return
        
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} records from {input_csv_path}")

    y = df['instrument_label']

    # 2. First split: 70% for training, 30% for temporary set
    # Using config.SEED for reproducibility
    train_df, temp_df, y_train, y_temp = train_test_split(
        df, y, test_size=0.30, stratify=y, random_state=config.SEED
    )

    # 3. Second split: Split the temporary set into validation and test sets
    val_df, test_df, y_val, y_test = train_test_split(
        temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=config.SEED
    )

    # 4. Add the 'split' column
    train_df = train_df.copy(); train_df['split'] = 'TRAIN'
    val_df = val_df.copy(); val_df['split'] = 'VALIDATION'
    test_df = test_df.copy(); test_df['split'] = 'TEST'

    # 5. Combine and save
    final_df = pd.concat([train_df, val_df, test_df]).sort_index()
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully created and saved stratified dataset to: {output_csv_path}")

    # 7. Print class distribution for verification
    print("\n--- Class Distribution Check (Percentages) ---")
    print(f"Total samples: {len(df)}")
    
    print(f"\nTRAIN set: {len(train_df)} samples ({(len(train_df)/len(df)):.0%})")
    print(train_df['instrument_label'].value_counts(normalize=True).sort_index())
    
    print(f"\nVALIDATION set: {len(val_df)} samples ({(len(val_df)/len(df)):.0%})")
    print(val_df['instrument_label'].value_counts(normalize=True).sort_index())

    print(f"\nTEST set: {len(test_df)} samples ({(len(test_df)/len(df)):.0%})")
    print(test_df['instrument_label'].value_counts(normalize=True).sort_index())

    print("\n--- Stratified Splitting Complete ---")


if __name__ == '__main__':
    create_stratified_splits()