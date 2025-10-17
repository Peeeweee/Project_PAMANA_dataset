# project_pamana/03_code/b_traditional_ml/06_train_random_forest.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- CORRECTED IMPORT LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up ONE level to the '03_code' directory
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def train_and_evaluate_rf():
    """
    Trains a Random Forest classifier, evaluates it on the validation set,
    and saves the trained model to disk.
    """
    print("--- Training and Evaluating Random Forest ---")

    # 1. Load Data
    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')
    df = pd.read_csv(features_path)

    # 2. Data Preparation
    train_df = df[df['split'] == 'TRAIN'].copy()
    val_df = df[df['split'] == 'VALIDATION'].copy()
    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    feature_cols = [col for col in df.columns if col not in ['filepath', 'instrument_label', 'split']]
    
    X_train = train_df[feature_cols].values
    y_train_str = train_df['instrument_label'].values
    
    X_val = val_df[feature_cols].values
    y_val_str = val_df['instrument_label'].values
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_val = le.transform(y_val_str)
    class_labels = le.classes_
    print(f"\nClasses: {class_labels}")

    # 3. Class Weight Calculation
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    print("\nClass weights are all 1.0 because the training set is balanced.")

    # 4. Model Training
    print("\nTraining the RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=config.SEED,
        class_weight=class_weights_dict,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate on VALIDATION Set
    print("\n--- Evaluating model on the unseen VALIDATION set ---")
    y_pred = rf_model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=class_labels)
    
    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # 6. Visualize Confusion Matrix
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix on Validation Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plots_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'plots_and_charts')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'rf_confusion_matrix_val.png'))
    plt.show()

    # 7. Model Export and Feature Analysis
    model_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, 'random_forest_clean.pkl')
    joblib.dump(rf_model, model_output_path)
    print(f"\nModel saved to: {model_output_path}")

    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols, 'importance': importances
    }).sort_values(by='importance', ascending=False)
    print("\n--- Top 10 Most Important Features ---")
    print(feature_importance_df.head(10).to_string(index=False))
    print("--------------------------------------")
    
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    train_and_evaluate_rf()