
import os
import zipfile
import shutil
import sys

# Temporarily add the '03_code' directory to the system path to import config
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(script_dir), 'project_pamana', '03_code')
sys.path.insert(0, code_dir)
from Config import PROJECT_ROOT

def extract_audio_archive():
    """
    Extracts the audio ZIP file and ensures all contents are moved 
    directly into the raw_audio directory, removing any intermediate folder.
    """
    print("--- Starting Audio Extraction ---")
    
    zip_filename = "Philipperc NKI.zip"
    zip_filepath = os.path.join(PROJECT_ROOT, zip_filename)
    extract_path = os.path.join(PROJECT_ROOT, "project_pamana", "01_data", "raw_audio")
    
    print(f"Looking for zip file at: {zip_filepath}")
    print(f"Extraction destination: {extract_path}")
    
    # Ensure the extraction path exists
    if not os.path.exists(extract_path):
        print(f"Extraction path {extract_path} not found. Please ensure the project setup was run correctly.")
        return

    # Check if the zip file exists and extract it
    if not os.path.isfile(zip_filepath):
        print(f"ERROR: Zip file not found at {zip_filepath}")
        return
        
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Successfully extracted '{zip_filename}' to '{extract_path}'")

        # Check for and handle a single root folder within the extraction path
        extracted_items = os.listdir(extract_path)
        if len(extracted_items) == 1:
            potential_folder = os.path.join(extract_path, extracted_items[0])
            if os.path.isdir(potential_folder):
                print(f"Detected single directory: {extracted_items[0]}. Moving contents up.")
                
                # Move all files and subdirectories from the intermediate folder up
                for item_name in os.listdir(potential_folder):
                    source_item = os.path.join(potential_folder, item_name)
                    destination_item = os.path.join(extract_path, item_name)
                    shutil.move(source_item, destination_item)
                
                # Remove the now-empty intermediate folder
                os.rmdir(potential_folder)
                print(f"Successfully moved contents and removed empty directory '{extracted_items[0]}'.")

    except zipfile.BadZipFile:
        print(f"ERROR: The file '{zip_filename}' is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    print("--- Audio Extraction Complete ---")

if __name__ == "__main__":
    extract_audio_archive()
