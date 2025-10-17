# project_pamana/03_code/a_preprocessing/02a_group_stratified_split.py

import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- CORRECTED IMPORT LOGIC ---
# Get the absolute path of the directory containing this script (a_preprocessing)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (03_code)
code_dir = os.path.dirname(script_dir)
# Add the '03_code' directory to Python's path
sys.path.insert(0, code_dir)

# Now that '03_code' is in the path, we can import 'config' directly.
import config

def create_group_stratified_splits():
    """
    Loads the initial labeled dataset, creates a group identifier for each recording session,
    and splits the dataset into training, validation, and test sets using group-stratified
    splitting to prevent data leakage. The result is saved to a new CSV file.
    """
    print("--- Starting Group-Stratified Dataset Splitting ---")

    # 1. Define paths and load the initial DataFrame
    logs_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'logs_and_metrics')
    input_csv_path = os.path.join(logs_dir, 'pamana_initial_labels.csv')
    output_csv_path = os.path.join(logs_dir, 'pamana_group_stratified_split_dataset.csv')

    if not os.path.exists(input_csv_path):
        print(f"ERROR: Labels CSV not found at {input_csv_path}. Please run 01_label_data.py first.")
        return
        
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} records from {input_csv_path}")

    # 2. Create a 'group' identifier from the filepath
    # Extracts the filename and removes the extension and any parenthetical versioning
    # e.g., '.../Agung_Gong_01_den_E1(02).wav' -> 'Agung_Gong_01_den_E1'
    df['group'] = df['filepath'].apply(lambda x: os.path.basename(x).split('(')[0].replace('_', ' ').strip())

    print("Created 'group' column for splitting.")
    
    y = df['instrument_label']
    groups = df['group']

    # 3. First split: 70% for training, 30% for temporary set
    gss_train_test = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=config.SEED)
    train_idx, temp_idx = next(gss_train_test.split(df, y, groups))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # 4. Second split: Split the temporary set into validation and test sets (50/50)
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=config.SEED)
    val_idx, test_idx = next(gss_val_test.split(temp_df, temp_df['instrument_label'], temp_df['group']))

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # 5. Add the 'split' column
    train_df = train_df.copy(); train_df['split'] = 'TRAIN'
    val_df = val_df.copy(); val_df['split'] = 'VALIDATION'
    test_df = test_df.copy(); test_df['split'] = 'TEST'

    # 6. Combine and save
    final_df = pd.concat([train_df, val_df, test_df]).sort_index()
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully created and saved group-stratified dataset to: {output_csv_path}")

    # 7. Print class and group distribution for verification
    print("\n--- Group Integrity Check ---")
    train_groups = set(train_df['group'])
    val_groups = set(val_df['group'])
    test_groups = set(test_df['group'])
    
    print(f"Groups in TRAIN: {len(train_groups)}")
    print(f"Groups in VALIDATION: {len(val_groups)}")
    print(f"Groups in TEST: {len(test_groups)}")
    
    print(f"Intersection between TRAIN and VALIDATION groups: {len(train_groups.intersection(val_groups))}")
    print(f"Intersection between TRAIN and TEST groups: {len(train_groups.intersection(test_groups))}")
    print(f"Intersection between VALIDATION and TEST groups: {len(val_groups.intersection(test_groups))}")

    print("\n--- Class Distribution Check (Percentages) ---")
    print(f"Total samples: {len(df)}")
    
    print(f"\nTRAIN set: {len(train_df)} samples ({(len(train_df)/len(df)):.0%})")
    print(train_df['instrument_label'].value_counts(normalize=True).sort_index())
    
    print(f"\nVALIDATION set: {len(val_df)} samples ({(len(val_df)/len(df)):.0%})")
    print(val_df['instrument_label'].value_counts(normalize=True).sort_index())

    print(f"\nTEST set: {len(test_df)} samples ({(len(test_df)/len(df)):.0%})")
    print(test_df['instrument_label'].value_counts(normalize=True).sort_index())

    print("\n--- Group-Stratified Splitting Complete ---")


if __name__ == '__main__':
    create_group_stratified_splits()
