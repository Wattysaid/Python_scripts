import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os

def select_excel_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Select Excel File", "Please select the Excel file to convert.")
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    return file_path

def select_output_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Select Output Folder", "Please select the folder to save the HTML file.")
    directory = filedialog.askdirectory()
    return directory

def convert_excel_to_html(excel_file, output_directory):
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Convert the DataFrame to HTML
        html_string = df.to_html(index=False)  # Exclude DataFrame index from HTML
        
        # Prompt for output file name
        file_name = simpledialog.askstring("Input", "Enter the name for the HTML file:",
                                           parent=tk.Tk().withdraw())

        if not file_name.endswith('.html'):
            file_name += '.html'

        # Save the HTML to a file
        output_path = os.path.join(output_directory, file_name)
        with open(output_path, 'w') as f:
            f.write(html_string)
            
        messagebox.showinfo("Success", f"HTML file has been saved successfully to {output_path}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    excel_file = select_excel_file()
    if excel_file:  # Proceed if the user selected a file
        output_directory = select_output_directory()
        if output_directory:  # Proceed if the user selected an output directory
            convert_excel_to_html(excel_file, output_directory)
        else:
            messagebox.showwarning("Warning", "No output directory selected. Operation cancelled.")
    else:
        messagebox.showwarning("Warning", "No Excel file selected. Operation cancelled.")
