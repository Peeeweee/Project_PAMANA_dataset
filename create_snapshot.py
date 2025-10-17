import os
from pathlib import Path

# --- SCRIPT CONFIGURATION ---

# 1. SET YOUR PROJECT'S ROOT DIRECTORY HERE
#    - '.' means the script will run on the folder it's placed in.
#    - You can also use an absolute path, e.g., Path("C:/Users/YourUser/Projects/SPORTS-SCOREBOARD-PRO")
PROJECT_ROOT = Path('.')

# 2. SET THE NAME OF THE OUTPUT FILE
OUTPUT_FILENAME = "project_snapshot.txt"

# 3. ADD ANY FILES, FOLDERS, OR EXTENSIONS TO IGNORE
#    - This script already ignores common folders like '.git' and 'node_modules'.
#    - Add anything specific to your project you want to exclude.
#    - Wildcards like '*.log' are supported.
IGNORE_PATTERNS = {
    # Version Control & IDE
    '.git',
    '.vscode',
    '.idea',

    # Dependencies & Build Output
    'node_modules',
    'dist',
    'build',
    'out',
    'coverage',

    # Environment & System Files
    '.env',
    '.env.local',
    '.env.development.local',
    '.env.test.local',
    '.env.production.local',
    '.DS_Store',

    # Python specific
    '__pycache__',
    '*.pyc',
    '.venv',
    'venv',

    # Logs and the output file itself
    '*.log',
    OUTPUT_FILENAME,

    # Common binary file extensions (content will be skipped)
    '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg',
    '*.pdf', '*.zip', '*.gz', '*.tar', '*.rar',
    '*.woff', '*.woff2', '*.ttf', '*.eot', '*.otf',
    '*.mp3', '*.wav', '*.mp4', '*.mov', '*.webm',
}

def is_ignored(path: Path, root: Path, ignore_patterns: set) -> bool:
    """
    Checks if a file or directory should be ignored based on the patterns.
    """
    # Check against patterns relative to the project root
    try:
        relative_path_str = str(path.relative_to(root))
        if any(part in ignore_patterns for part in relative_path_str.split(os.sep)):
            return True
    except ValueError:
        pass  # Path is not within the root, should not happen in this script flow

    # Check against file/folder name and glob patterns
    return (
        path.name in ignore_patterns or
        any(path.match(p) for p in ignore_patterns if '*' in p)
    )

def generate_directory_tree(root_path: Path, ignore_patterns: set, prefix: str = "") -> list:
    """
    Generates a list of strings representing the directory tree view.
    """
    if not root_path.is_dir():
        return []

    items = sorted([item for item in root_path.iterdir() if not is_ignored(item, PROJECT_ROOT, ignore_patterns)])
    tree_lines = []
    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        tree_lines.append(f"{prefix}{connector}{item.name}")
        if item.is_dir():
            extension = "    " if i == len(items) - 1 else "│   "
            tree_lines.extend(generate_directory_tree(item, ignore_patterns, prefix + extension))
    return tree_lines

def get_project_files(root_path: Path, ignore_patterns: set) -> list:
    """
    Walks the directory and returns a sorted list of file paths to include.
    """
    files_to_include = []
    for root, _, files in os.walk(root_path):
        root_path_obj = Path(root)
        if is_ignored(root_path_obj, root_path, ignore_patterns):
            continue
        for file in files:
            file_path = root_path_obj / file
            if not is_ignored(file_path, root_path, ignore_patterns):
                files_to_include.append(file_path)
    return sorted(files_to_include)

def main():
    """
    Main function to generate the project snapshot file.
    """
    print("Starting project snapshot generation...")
    
    output_file_path = PROJECT_ROOT / OUTPUT_FILENAME

    with open(output_file_path, "w", encoding="utf-8") as f:
        # --- 1. Write the Directory Tree ---
        f.write("PROJECT DIRECTORY TREE (Filtered View)\n")
        f.write("======================================================================\n")
        
        # Display the root folder name at the top of the tree
        f.write(f"{PROJECT_ROOT.resolve().name}\n")
        
        tree_lines = generate_directory_tree(PROJECT_ROOT, IGNORE_PATTERNS)
        f.write("\n".join(tree_lines))
        f.write("\n\n\n") # Add spacing before file contents

        # --- 2. Write File Contents ---
        project_files = get_project_files(PROJECT_ROOT, IGNORE_PATTERNS)
        
        print(f"Found {len(project_files)} files to include.")

        for file_path in project_files:
            try:
                relative_path = file_path.relative_to(PROJECT_ROOT)
                print(f"  -> Processing: {relative_path}")
                
                f.write(f"--- START FILE: {relative_path} ---\n")
                
                # Read file content with robust error handling
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    f.write(content)
                    # Ensure a newline at the end of the file content
                    if content and not content.endswith('\n'):
                        f.write('\n')
                except Exception as e:
                    f.write(f"*** Error reading file: {e} ***\n")
                    
                f.write(f"--- END FILE: {relative_path} ---\n\n")
            except Exception as e:
                print(f"  -> ERROR processing file {file_path}: {e}")


    print("----------------------------------------------------------------------")
    print(f"Success! Project snapshot created at: {output_file_path.resolve()}")
    print("You can now copy the contents of this file.")
    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    main()