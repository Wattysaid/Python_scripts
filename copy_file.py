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
# Note: corrected package name to lowercase for consistency
    required_packages = ["os", "shutil", "pathlib", "tqdm"] 

# Check and install required packages
check_and_install_packages(required_packages)

# select the destination file, summarise the file, select destination folder, copy
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def get_file_details(file_path):
    file_info = {
        "name": Path(file_path).name,
        "type": Path(file_path).suffix,
        "size_bytes": Path(file_path).stat().st_size,
        "size_mb": Path(file_path).stat().st_size / (1024 ** 2)  # Convert size to MB
    }
    return file_info

def confirm(prompt):
    while True:
        response = input(prompt + " (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False

def copy_with_progress(src, dst):
    # Get the size of the file
    file_size = os.path.getsize(src)
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Copying") as pbar:
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst, 1024*1024, callback=lambda x: pbar.update(len(x)))

def main():
    file_path = input("Please enter the path of the file you wish to copy: ")
    if not Path(file_path).exists():
        print("File does not exist. Please check the path and try again.")
        return

    file_details = get_file_details(file_path)
    print(f"File Name: {file_details['name']}, File Type: {file_details['type']}, File Size: {file_details['size_mb']:.2f} MB")

    destination_folder = input("Please enter the destination folder path: ")
    if not Path(destination_folder).is_dir():
        print("Destination is not a directory. Please check the path and try again.")
        return

    if confirm("Do you want to proceed with the copy?"):
        destination_path = Path(destination_folder) / file_details['name']
        copy_with_progress(file_path, destination_path)
        print("File copied successfully!")
        print(f"Transfer Summary:\n- Source: {file_path}\n- Destination: {destination_path}\n- File Size: {file_details['size_mb']:.2f} MB")
    else:
        print("File copy cancelled.")

    input("Press any key to exit...")

if __name__ == "__main__":
    main()
