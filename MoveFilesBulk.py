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
required_packages = ["shutil", "tqdm", "time", "os"]  # Note: corrected package name to lowercase for consistency

# Check and install required packages
check_and_install_packages(required_packages)

import os
import shutil
from tqdm import tqdm
import time

def scan_folder(folder):
    file_types = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            extension = file.split('.')[-1]
            if extension not in file_types:
                file_types[extension] = []
            file_types[extension].append(os.path.join(root, file))
    return file_types

def summarize_files(file_types):
    summary = []
    for extension, files in file_types.items():
        summary.append(f"{len(files)} {extension.upper()} files")
    return summary

def move_files(selected_types, file_types, destination):
    total_files = sum(len(file_types[ext]) for ext in selected_types)
    log_file_path = os.path.join(destination, "moved_files_log.txt")
    with open(log_file_path, "w") as log_file, tqdm(total=total_files) as pbar:
        start_time = time.time()
        for ext in selected_types:
            for file_path in file_types[ext]:
                shutil.move(file_path, destination)
                log_file.write(file_path + "\n")
                pbar.update(1)
        end_time = time.time()
    return total_files, end_time - start_time

def main():
    folder = input("Enter the folder path to scan: ")
    file_types = scan_folder(folder)
    summary = summarize_files(file_types)
    print("Files found:")
    for i, item in enumerate(summary, 1):
        print(f"{i}. {item}")

    selections = input("Select the numbers to move (comma-separated): ")
    selected_indices = [int(index) for index in selections.split(',')]
    selected_types = [list(file_types.keys())[i-1] for i in selected_indices]

    destination = input("Enter the destination folder path: ")
    print(f"Upon approval, I will proceed to move {', '.join([summary[i-1] for i in selected_indices])} to {destination}.")
    confirm = input("Do you wish to proceed? (yes/no): ")

    if confirm.lower() == 'yes':
        total_files, duration = move_files(selected_types, file_types, destination)
        print(f"Moved {total_files} files in {duration:.2f} seconds. Transfer speed: {total_files/duration:.2f} files/sec")
        print(f"A log of moved files has been saved to {destination}")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
