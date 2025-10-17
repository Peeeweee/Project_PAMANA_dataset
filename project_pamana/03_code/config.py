import os

# This file contains global constants and configuration settings for the project.

# --- Project Structure ---
# Define the absolute path to the project's root directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The universal random seed for ensuring reproducibility across all scripts.
SEED = 42

# You can add other configuration variables here as the project grows.
# For example:
# SAMPLE_RATE = 22050
# DURATION = 3.0
