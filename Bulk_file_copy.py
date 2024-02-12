import os
import shutil
import csv
from datetime import datetime

def find_and_copy_files(src_directory, target_directory, file_extensions, csv_file_name):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    total_files_copied = 0

    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Title', 'File Type', 'Size (bytes)', 'Created On'])

        # Walk through the directory and subdirectories
        for root, dirs, files in os.walk(src_directory):
            for file in files:
                if file.endswith(file_extensions):
                    src_file_path = os.path.join(root, file)  # Full path of the source file
                    dest_file_path = os.path.join(target_directory, os.path.relpath(src_file_path, src_directory))  # Constructing destination path

                    try:
                        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)  # Create subdirectories in target if they don't exist
                        shutil.copy(src_file_path, dest_file_path)
                        total_files_copied += 1

                        # Getting file details
                        file_size = os.path.getsize(src_file_path)
                        file_type = os.path.splitext(file)[1]
                        creation_time = datetime.fromtimestamp(os.path.getctime(src_file_path)).strftime('%Y-%m-%d %H:%M:%S')

                        # Writing details to CSV
                        csv_writer.writerow([file, file_type, file_size, creation_time])
                    except Exception as e:
                        print(f"Error copying file {src_file_path}: {e}")

    return total_files_copied

# File extensions to look for
extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')

# Prompt user for source and target directories
source_directory = input("Enter the source directory path: ")
target_directory = input("Enter the target directory path for downloading the files: ")

csv_save_location = input("Enter the directory path to save the CSV file (Press Enter to use the Downloads folder): ")
if not csv_save_location:
    csv_save_location = os.path.join(os.path.expanduser("~"), "Downloads")
csv_file_name = os.path.join(csv_save_location, "files_copied.csv")

# Core script
total_files_copied = find_and_copy_files(source_directory, target_directory, extensions, csv_file_name)

# Summary and closing
print(f"\nTotal files copied: {total_files_copied}")
print(f"CSV report generated at: {csv_file_name}")

# Wait for user to press Enter before closing
input("\nPress Enter to close the application...")
