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
required_packages = ["pandas", "openpyxl", "pyarrow", "os"]  # Note: corrected package name to lowercase for consistency

# Check and install required packages
check_and_install_packages(required_packages)

import pandas as pd
import os

def split_multiline_cells(excel_file_path, output_folder_path, has_header=True):
    if has_header:
        df = pd.read_excel(excel_file_path)
    else:
        df = pd.read_excel(excel_file_path, header=None)
    
    all_rows = []  # List to store all rows before adding them to the DataFrame
    
    for index, row in df.iterrows():
        max_splits = 1
        row_data = {}
        
        for col in df.columns:
            cell_value = str(row[col])
            if '\n' in cell_value:
                split_content = cell_value.split('\n')
                max_splits = max(max_splits, len(split_content))
                row_data[col] = split_content
            else:
                row_data[col] = [cell_value]
                
        for split_index in range(max_splits):
            new_row = {}
            for col in df.columns:
                try:
                    new_row[col] = row_data[col][split_index]
                except IndexError:
                    new_row[col] = None  # Fill missing splits with None
            all_rows.append(new_row)
    
    # Create a new DataFrame from the list of all rows
    modified_df = pd.DataFrame(all_rows)
    
    if not has_header:
        # If there was no header, the columns will be numerical. Convert them to string to avoid issues.
        modified_df.columns = [str(col) for col in modified_df.columns]
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    output_file_name = "output_excel.xlsx"
    output_file_path = os.path.join(output_folder_path, output_file_name)
    
    modified_df.to_excel(output_file_path, index=False, header=has_header)
    print(f"Output file saved to: {output_file_path}")

def get_user_input():
    excel_file_path = input("Enter the full path to your Excel file (including the file name): ")
    output_folder_path = input("Enter the full path to the destination folder where the output file will be saved: ")
    header_response = input("Does the first row in your spreadsheet contain headers? (yes/no): ").strip().lower()
    has_header = header_response == 'yes'
    
    split_multiline_cells(excel_file_path, output_folder_path, has_header)

def main():
    get_user_input()

if __name__ == "__main__":
    main()