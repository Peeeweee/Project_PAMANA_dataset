import os
import zipfile
import shutil

source_folder = "."
zip_filename = "dataset_wav.zip"

# Read n Write to input/dataset_wav.zip
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_folder)
                zipf.write(file_path, arcname)

output_folder = "input"
os.makedirs(output_folder, exist_ok=True)

shutil.move(zip_filename, os.path.join(output_folder, zip_filename))

# Delete
deleted_count = 0
for root, _, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            deleted_count += 1

print(deleted_count)