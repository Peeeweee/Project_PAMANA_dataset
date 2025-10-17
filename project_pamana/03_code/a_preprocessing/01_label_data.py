

import os
import pandas as pd

def create_pamana_labels():
    """
    Scans the raw audio directory, extracts and normalizes instrument labels,
    and saves the file paths and labels to a CSV file.
    """
    print("--- Starting Data Labeling ---")

    # Define paths relative to the project root directory
    project_root = 'project_pamana'
    input_dir = os.path.join(project_root, '01_data', 'raw_audio')
    output_dir = os.path.join(project_root, '04_results', 'logs_and_metrics')
    output_csv_path = os.path.join(output_dir, 'pamana_initial_labels.csv')

    print(f"Scanning for audio files in: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found at {input_dir}. Please ensure audio files are extracted.")
        return

    file_data = []
    supported_extensions = ('.wav', '.mp3')

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                filepath = os.path.join(root, file)
                parent_dir_name = os.path.basename(root)
                
                label = ''
                # Case 1: Label is the parent directory name (e.g., raw_audio/Tongatong/clip1.wav)
                if parent_dir_name != 'raw_audio':
                    label = parent_dir_name
                # Case 2: Label is in the filename (e.g., raw_audio/Tongatong_clip1.wav)
                else:
                    filename_no_ext = os.path.splitext(file)[0]
                    label = filename_no_ext.split('_')[0]

                # Normalize and validate the label
                normalized_label = label.strip().lower()
                
                if not normalized_label:
                    print(f"Warning: Could not determine a label for file: {filepath}. Skipping.")
                    continue

                file_data.append({
                    'filepath': filepath,
                    'instrument_label': normalized_label
                })

    if not file_data:
        print("No audio files found. The 'raw_audio' directory may be empty.")
        return

    # Create DataFrame
    df = pd.DataFrame(file_data)

    print("\n--- Initial Class Distribution ---")
    class_distribution = df['instrument_label'].value_counts()
    print(class_distribution)
    print("------------------------------------\n")

    # Save DataFrame to CSV
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully saved label data to: {output_csv_path}")
    
    print("--- Data Labeling Complete ---")

if __name__ == '__main__':
    create_pamana_labels()
