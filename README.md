# Project PAMANA — Philippine Traditional Instrument Dataset

This is a curated dataset ranging from 1-6 secs isolated audio clips for five traditional Philippine instruments:
- Tongatong
- Kulintang
- Gandingan
- Dabakan
- Agung

## Dataset Structure
the `input/` folder contains `.zip` archives. For each `.wav` instrument class. 

`PAMANA_dataset.csv` is a Metadata file containing essential descriptors for every `.wav` clip in the dataset Colmns:
- filename : the original file name
- path : relative path to the corresponding file in the repository
- duration : audio length in seconds (float)
- sample_rate : sampling frequency in `48kHz`
- channels : number of channels (1 = mono, 2 = stereo)