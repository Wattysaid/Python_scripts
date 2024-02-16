import subprocess
import sys

def install(package):
    """Install the specified Python package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_packages(required_packages):
    missing_packages = []
    installed_packages = []  # Keep track of newly installed packages

    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        missing_packages_str = ", ".join(missing_packages)
        consent = input(f"Would you like to install the missing packages ({missing_packages_str})? [yes/no]: ")
        if consent.lower() == 'yes':
            for package in missing_packages:
                print(f"Installing {package}...")
                install(package)
                installed_packages.append(package)  # Add to the list of installed packages
        else:
            print("Missing packages are required to run the application. Exiting.")
            sys.exit(1)
    else:
        print("All required packages are already installed.")

    # Summarising Installed Packages
    if installed_packages:
        print(f"Installed packages: {', '.join(installed_packages)}")
    else:
        print("No new packages were installed.")

    # Wait for User Input
    input("Press Enter to continue with the script...")

# List of required packages
required_packages = ["os", "shutil","sys","collections","tqdm", "psutil","concurrent.futures"]  # Note: corrected package name to lowercase for consistency

# Check and install required packages
check_and_install_packages(required_packages)

# Import Required Libraries
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import sys
from collections import defaultdict
from tqdm import tqdm
import psutil  # For checking available RAM

# Function to prompt user to select the source folder
def select_source_folder():
    source_folder = input("Please enter the path of the source folder: ")
    if not os.path.isdir(source_folder):
        print("The specified path does not exist or is not a directory.")
        sys.exit(1)
    return source_folder

# Function to dynamically suggest the number of simultaneous copies based on available RAM
def suggest_copy_count():
    total_ram_gb = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB
    # Assuming each thread could ideally use up to 0.25GB, this is a heuristic and may need adjustment
    suggested_copies = int(total_ram_gb / 0.25)
    suggested_copies = min(100, max(1, suggested_copies))  # Ensure between 1 and 100
    print(f"Suggested number of simultaneous copies based on available RAM: {suggested_copies}")
    return suggested_copies

# Function to prompt user for the number of simultaneous copies, with a suggestion
def get_copy_count(suggested_copies):
    while True:
        try:
            copy_count = int(input(f"Enter the number of files to copy simultaneously (1-100), suggested: {suggested_copies}: "))
            if 1 <= copy_count <= 100:
                return copy_count
            else:
                print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Function to prompt user to select the destination folder
def select_destination_folder():
    destination_folder = input("Please enter the path of the destination folder: ")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    return destination_folder

# Function to scan all folders including subfolders and summarize files by type
def scan_folders(source_folder):
    file_paths = []
    file_types = defaultdict(int)
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_paths.append(os.path.join(root, file))
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] += 1
    # Print summary of files by type
    print("Summary of files to copy by type:")
    for ext, count in file_types.items():
        print(f"{ext if ext else 'No Extension'}: {count} files")
    return file_paths

# Function to copy files to new folder with a progress bar
def copy_files(file_paths, destination_folder, copy_count, source_folder):
    def copy_file(file_path):
        destination_path = os.path.join(destination_folder, os.path.relpath(file_path, start=source_folder))
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(file_path, destination_path)

    with ThreadPoolExecutor(max_workers=copy_count) as executor:
        list(tqdm(executor.map(copy_file, file_paths), total=len(file_paths), desc="Copying files"))

# Function to ask user if they want to proceed after showing file summary
def confirm_copy():
    while True:
        proceed = input("Would you like to proceed to copy the files? (yes/no): ").lower()
        if proceed in ['yes', 'no']:
            return proceed == 'yes'
        else:
            print("Please answer with 'yes' or 'no'.")

# Main script execution flow
def main():
    source_folder = select_source_folder()
    suggested_copies = suggest_copy_count()
    copy_count = get_copy_count(suggested_copies)
    destination_folder = select_destination_folder()
    file_paths = scan_folders(source_folder)
    print(f"Found {len(file_paths)} files to copy.")

    if confirm_copy():
        copy_files(file_paths, destination_folder, copy_count, source_folder)
        print("Copying complete.")
    else:
        print("Copy operation cancelled.")

    input("Press Enter to exit...")  # This line keeps the window open until the user presses Enter.

if __name__ == "__main__":
    main()
