# Check if the required packages are installed
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
required_packages = ["os", "shutil", "tqdm", "time"]  # Note: corrected package name to lowercase for consistency

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
    print("Bulk File Copy/Move Utility")
    print("-" * 30)
    
    # Get source directory
    while True:
        source = input("\nEnter the source directory path: ").strip()
        if os.path.isdir(source):
            break
        print("Error: Directory not found. Please try again.")
    
    # Scan files
    print(f"\nScanning files in: {source}")
    file_types = scan_files(source)
    
    if not file_types:
        print("No files found in the specified directory.")
        return
    
    # Display summary
    print("\nFiles found:")
    summary = summarize_files(file_types)
    for item in summary:
        print(f"  - {item}")
    
    # Select file types to move
    print(f"\nAvailable file types: {', '.join(file_types.keys())}")
    selected = input("Enter file types to move (comma-separated, or 'all'): ").strip()
    
    if selected.lower() == 'all':
        selected_types = list(file_types.keys())
    else:
        selected_types = [ext.strip() for ext in selected.split(',') if ext.strip() in file_types]
    
    if not selected_types:
        print("No valid file types selected.")
        return
    
    # Get destination
    while True:
        destination = input("\nEnter destination directory: ").strip()
        if os.path.isdir(destination):
            break
        print("Error: Destination directory not found.")
    
    # Confirm and move
    total_to_move = sum(len(file_types[ext]) for ext in selected_types)
    if input(f"Move {total_to_move} files? (y/n): ").lower() == 'y':
        total_moved, duration = move_files(selected_types, file_types, destination)
        print(f"\n✓ Moved {total_moved} files in {duration:.2f} seconds")
        print(f"Log file created: {os.path.join(destination, 'moved_files_log.txt')}")

if __name__ == "__main__":
    main()