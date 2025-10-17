# project_pamana/03_code/b_traditional_ml/06_train_random_forest.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict

# --- CORRECTED IMPORT LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def tune_and_evaluate_rf():
    """
    Tunes, evaluates, and saves a Random Forest classifier. It uses GridSearchCV
    with StratifiedGroupKFold to find the best hyperparameters, evaluates performance,
    generates a confusion matrix, and saves the best-performing model.
    """
    print("--- Tuning and Evaluating Random Forest with GridSearchCV ---")

    # 1. Load Data
    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')
    df = pd.read_csv(features_path)

    cv_df = df[df['split'].isin(['TRAIN', 'VALIDATION'])].copy()
    print(f"Loaded {len(cv_df)} samples for tuning and cross-validation.")

    feature_cols = [col for col in df.columns if col not in ['filepath', 'instrument_label', 'split', 'group']]
    
    X = cv_df[feature_cols].values
    y_str = cv_df['instrument_label'].values
    groups = cv_df['group'].astype(str).values
    
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_labels = le.classes_
    print(f"\nClasses: {class_labels}")

    # 2. Hyperparameter Tuning with GridSearchCV
    print("\nStarting Hyperparameter Search for Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],       # Number of trees
        'max_depth': [10, 20, None],           # Max depth of trees
        'min_samples_split': [2, 5],          # Min samples to split a node
        'min_samples_leaf': [1, 2]             # Min samples at a leaf node
    }

    N_SPLITS = 5
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight='balanced', random_state=config.SEED, n_jobs=-1),
        param_grid=param_grid,
        scoring='accuracy',
        cv=sgkf,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y, groups=groups)

    # 3. Display Best Results
    print("\n--- GridSearchCV Results for Random Forest ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")
    print("---------------------------------------------")

    best_model = grid_search.best_estimator_

    # 4. Generate and Save Combined Confusion Matrix
    print("\nGenerating confusion matrix from cross-validated predictions...")
    y_pred_cv = cross_val_predict(best_model, X, y, groups=groups, cv=sgkf)
    
    cm = confusion_matrix(y, y_pred_cv, labels=np.arange(len(class_labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Random Forest - Combined Confusion Matrix (Tuned, Cross-Validated)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plots_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'plots_and_charts')
    os.makedirs(plots_dir, exist_ok=True)
    cm_path = os.path.join(plots_dir, 'rf_tuned_confusion_matrix_cv.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # 5. Model Export
    print("\nExporting the best tuned model...")
    model_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, 'rf_tuned_final_model.pkl')
    joblib.dump(best_model, model_output_path)
    print(f"\nBest tuned model saved to: {model_output_path}")
    
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    tune_and_evaluate_rf()
