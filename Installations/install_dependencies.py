
import subprocess
import sys

def install_libraries():
    """Installs the core Python libraries required for the project."""
    print("--- Starting Library Installation ---")
    libraries = [
        "numpy",
        "pandas",
        "librosa",
        "scikit-learn",
        "tensorflow"
    ]
    
    for lib in libraries:
        print(f"Installing {lib}...")
        try:
            # Using sys.executable to ensure pip is called from the correct Python installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"Successfully installed {lib}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install {lib}. Please install it manually.")
            print(f"Error details: {e}")
        except FileNotFoundError:
            print(f"ERROR: 'pip' command not found. Make sure Python and pip are correctly installed and in your system's PATH.")
    print("--- Library Installation Complete ---")

if __name__ == "__main__":
    install_libraries()
