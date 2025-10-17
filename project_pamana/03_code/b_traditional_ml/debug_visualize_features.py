# project_pamana/03_code/b_traditional_ml/debug_visualize_features.py

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- Import Logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
sys.path.insert(0, code_dir)
import config

def visualize_features_with_tsne():
    """
    Performs a t-SNE dimensionality reduction on the feature set to visualize
    how the different instrument classes are clustered in 2D space.
    """
    print("--- Starting Feature Visualization with t-SNE ---")

    # 1. Load Data
    features_path = os.path.join(config.PROJECT_ROOT, '02_features', 'traditional_ml_features', 'traditional_features_clean.csv')
    if not os.path.exists(features_path):
        print(f"ERROR: Feature file not found at {features_path}")
        print("Please run the preprocessing pipeline first (01 to 04).")
        return
        
    df = pd.read_csv(features_path)
    # We'll visualize all the data we have, including augmented samples
    print(f"Loaded {len(df)} total samples for visualization.")

    feature_cols = [col for col in df.columns if col not in ['filepath', 'instrument_label', 'split', 'group']]
    
    X = df[feature_cols].values
    y = df['instrument_label'].values

    # 2. Scale the features
    print("\nScaling features before applying t-SNE...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply t-SNE
    print("\nRunning t-SNE... This may take a moment.")
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=30,  # A common value, can be tuned
        max_iter=1000,   # Max iterations for the optimization
        random_state=config.SEED
    )
    tsne_results = tsne.fit_transform(X_scaled)
    print("t-SNE transformation complete.")

    # 4. Create and Save the Plot
    print("\nGenerating plot...")
    vis_df = pd.DataFrame()
    vis_df['t-SNE 1'] = tsne_results[:,0]
    vis_df['t-SNE 2'] = tsne_results[:,1]
    vis_df['instrument'] = y

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="t-SNE 1", y="t-SNE 2",
        hue="instrument",
        palette=sns.color_palette("hsv", len(df['instrument_label'].unique())),
        data=vis_df,
        legend="full",
        alpha=0.7
    )
    plt.title('2D t-SNE Visualization of Instrument Features')
    
    plots_dir = os.path.join(config.PROJECT_ROOT, '04_results', 'plots_and_charts')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'tsne_feature_visualization.png')
    plt.savefig(plot_path)
    
    print(f"\nSuccessfully saved visualization to: {plot_path}")
    print("--- Visualization Complete ---")

if __name__ == '__main__':
    visualize_features_with_tsne()
