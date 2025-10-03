import os
from scipy.io import wavfile
import numpy as np
import pandas as pd
import zipfile

zip_path = r"input\dataset_wav.zip"
os.makedirs("preprocessing", exist_ok=True)

# Extract all audio into preprocessing
with zipfile.ZipFile(zip_path, "r") as zipf:
    file_list = zipf.namelist()
    print("Files inside zip:", file_list[:10], "...")

    zipf.extractall('preprocessing')


# Importing all the Samples and converting into CSV
df = []
folder_path = "preprocessing"

for fname in os.listdir(folder_path):
    if fname.endswith(".wav"):
        fpath = os.path.join(folder_path, fname)
        sr, samples = wavfile.read(fpath)   # sr = sample rate, samples = numpy array

        duration = len(samples) / sr
        channels = 1 if samples.ndim == 1 else samples.shape[1]

        print(f"{fname}: sample rate={sr}, duration={duration:.2f} seconds")

        df.append({
            "filename" : fname,
            "path" : fpath,
            "duration" : duration,
            "sample_rate" : sr,
            "channels" : channels
        })

df = pd.DataFrame(df)
df.head(10)
df.tail(10)
df.info()
df.nunique()

# MetaData's
df.to_csv("PAMANA_dataset.csv", index=False)

new_df = pd.read_csv("PAMANA_dataset.csv")