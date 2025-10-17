# project_pamana/03_code/b_traditional_ml/07_train_svm.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict

# --- Import Logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def tune_and_evaluate_svm():
    """
    Tunes, evaluates, and saves an SVM classifier. It uses GridSearchCV with
    StratifiedGroupKFold to find the best hyperparameters, evaluates performance,
    generates a confusion matrix, and saves the best-performing model.
    """
    print("--- Tuning and Evaluating SVM with GridSearchCV ---")

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

    # 2. Scale features BEFORE GridSearch
    # We scale the entire dataset here. While scaling inside the pipeline is stricter,
    # for hyperparameter tuning, scaling the full CV dataset is a common and practical approach.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Hyperparameter Tuning with GridSearchCV
    print("\nStarting Hyperparameter Search with GridSearchCV...")
    
    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100],       # Regularization parameter
        'gamma': [1, 0.1, 0.01, 0.001], # Kernel coefficient
        'kernel': ['rbf']
    }

    # Define the cross-validation strategy
    N_SPLITS = 5
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=SVC(class_weight='balanced', probability=True, random_state=config.SEED),
        param_grid=param_grid,
        scoring='accuracy',
        cv=sgkf,
        n_jobs=-1, # Use all available CPU cores
        verbose=2
    )

    # Run the search
    grid_search.fit(X_scaled, y, groups=groups)

    # 4. Display Best Results
    print("\n--- GridSearchCV Results ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")
    print("--------------------------")

    # The best model is automatically retrained on the full dataset
    best_model = grid_search.best_estimator_

    # 5. Generate and Save Combined Confusion Matrix
    print("\nGenerating confusion matrix from cross-validated predictions...")
    # Get predictions from the cross-validation process
    y_pred_cv = cross_val_predict(best_model, X_scaled, y, groups=groups, cv=sgkf)
    
    cm = confusion_matrix(y, y_pred_cv, labels=np.arange(len(class_labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('SVM - Combined Confusion Matrix (Tuned, Cross-Validated)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plots_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'plots_and_charts')
    os.makedirs(plots_dir, exist_ok=True)
    cm_path = os.path.join(plots_dir, 'svm_tuned_confusion_matrix_cv.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # 6. Model and Scaler Export
    print("\nExporting the best tuned model and the scaler...")
    model_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, 'svm_tuned_final_model.pkl')
    scaler_output_path = os.path.join(model_dir, 'svm_final_scaler.pkl')
    
    # Save the best model found by GridSearchCV and the scaler fitted on the full CV data
    joblib.dump(best_model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"\nBest tuned model saved to: {model_output_path}")
    print(f"Scaler saved to: {scaler_output_path}")
    
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    tune_and_evaluate_svm()
