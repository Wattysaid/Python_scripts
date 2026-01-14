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

def choose_chunk_size(file_size):
    print("\nChoose a chunk size for copying:")
    print("1: 512 KB\n2: 1 MB\n3: 2 MB\n4: 4 MB\n5: 8 MB")
    if file_size > 500 * (1024 ** 2):  # Suggest larger chunk size for files larger than 500MB
        print("Suggestion: Choose a larger chunk size (4 or 5) for faster copying of large files.")
    else:
        print("Suggestion: Choose a smaller chunk size (1 or 2) for smaller files or if unsure about hardware capabilities.")

    choice = input("Enter your choice (1-5): ")
    chunk_sizes = {
        '1': 512 * 1024,
        '2': 1024 * 1024,
        '3': 2 * 1024 * 1024,
        '4': 4 * 1024 * 1024,
        '5': 8 * 1024 * 1024
    }
    return chunk_sizes.get(choice, 1024 * 1024)  # Default to 1MB if invalid choice

def copy_with_progress(src, dst, chunk_size):
    file_size = os.path.getsize(src)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst, tqdm(total=file_size, unit='B', unit_scale=True, desc="Copying") as pbar:
        while True:
            buffer = fsrc.read(chunk_size)
            if not buffer:
                break
            fdst.write(buffer)
            pbar.update(len(buffer))

def main():
    print("File Copy Utility")
    print("-" * 20)
    
    # Get source file
    while True:
        src_path = input("\nEnter the path to the source file: ").strip()
        if os.path.isfile(src_path):
            break
        print("Error: Source file not found. Please try again.")
    
    # Display file details
    file_details = get_file_details(src_path)
    print(f"\nFile Details:")
    print(f"Name: {file_details['name']}")
    print(f"Type: {file_details['type']}")
    print(f"Size: {file_details['size_mb']:.2f} MB ({file_details['size_bytes']:,} bytes)")
    
    # Get destination
    while True:
        dst_input = input("\nEnter the destination path (file or directory): ").strip()
        dst_path = Path(dst_input)
        
        if dst_path.is_dir():
            dst_path = dst_path / file_details['name']
        
        if confirm(f"Copy '{src_path}' to '{dst_path}'?"):
            break
    
    # Choose chunk size
    chunk_size = choose_chunk_size(file_details['size_bytes'])
    
    # Perform copy
    try:
        print(f"\nStarting copy operation...")
        copy_with_progress(src_path, str(dst_path), chunk_size)
        print(f"\nFile successfully copied to: {dst_path}")
    except Exception as e:
        print(f"Error during copy: {e}")

if __name__ == "__main__":
    main()