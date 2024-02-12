# first part of the script checks that all the required libraries are installed,
# and installes any missing then continuies to the second part of the script.

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["tqdm"]

for package in required_packages:
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} not found, installing...")
        install(package)

# second part of the scrip runs the dupliates checker

import os
import hashlib
import csv
from tqdm import tqdm

def get_file_hash(filename):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def count_files(start_dirs):
    count = 0
    for start_dir in start_dirs:
        for _, _, filenames in os.walk(start_dir):
            count += len(filenames)
    return count

def find_duplicates(start_dirs):
    files_seen = {}
    duplicates = {}
    unique_id = 0
    total_files = count_files(start_dirs)

    with tqdm(total=total_files, desc="Checking Files", unit="file") as pbar:
        for start_dir in start_dirs:
            for dirpath, _, filenames in os.walk(start_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        file_size = os.path.getsize(filepath)
                        file_hash = get_file_hash(filepath)
                        file_key = (file_size, file_hash)

                        if file_key in files_seen:
                            if file_key not in duplicates:
                                duplicates[file_key] = [files_seen[file_key]]
                                unique_id += 1
                            duplicates[file_key].append(filepath)
                        else:
                            files_seen[file_key] = filepath

                    except OSError as e:
                        print(f"Error accessing file {filepath}: {e}")
                    finally:
                        pbar.update(1)

    return duplicates, unique_id

def write_to_csv(duplicates, unique_id, csv_filename):
    with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Type', 'Size', 'Location', 'Unique ID'])

        for (size, _), paths in duplicates.items():
            for path in paths:
                file_title = os.path.basename(path)
                file_type = os.path.splitext(file_title)[1]
                writer.writerow([file_title, file_type, size, path, unique_id])

# print summary of the results to the screen
def print_summary(duplicates):
    total_groups = len(duplicates)
    total_duplicates = sum(len(paths) for paths in duplicates.values()) - total_groups
    print("\nDuplicate File Summary:")
    print(f"Total Duplicate Groups Found: {total_groups}")
    print(f"Total Duplicate Files: {total_duplicates}")

# Prompt user for directory paths to scan
user_input = input("Enter directory paths to scan, separated by commas: ")
directories = [dir.strip() for dir in user_input.split(',')]

duplicates, unique_id = find_duplicates(directories)

# Prompt user for location to save the CSV file
csv_save_location = input("Enter the destination path to save the CSV report (Press Enter for default Downloads folder): ")
if not csv_save_location:
    csv_save_location = os.path.join(os.path.expanduser("~"), "Downloads")
csv_file_name = os.path.join(csv_save_location, "duplicates_report.csv")

write_to_csv(duplicates, unique_id, csv_file_name)

print(f"\nDuplicate files report generated: {csv_file_name}")
print_summary(duplicates)

# Wait for user to press Enter before closing
input("\nPress Enter to close the application...")
