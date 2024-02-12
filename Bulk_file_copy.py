Purpose:
This script is designed to automate the process of copying specific file types from a source directory to a target directory. It generates a detailed CSV report containing information about each file copied, including its name, type, size, and creation date.

Key Features:

Directory Support: Handles both source and target directories, including the creation of non-existent target directories.
File Filtering: Copies files based on specified extensions, allowing for targeted file transfers.
Report Generation: Produces a CSV file report listing details of copied files for record-keeping.
Error Handling: Implements basic error handling during the file copy process to manage exceptions.
Workflow:

Directory Setup:

Checks and creates the target directory if it does not exist.
File Processing:

Iterates over the source directory, including subdirectories.
Filters files by the specified extensions.
Copies eligible files to the corresponding location within the target directory, preserving the subdirectory structure.
Reporting:

For each file copied, gathers file metadata (type, size, creation date).
Writes this information into a CSV file, creating a comprehensive report.
User Interaction:

Prompts the user to input source and target directory paths.
Allows specification of the CSV file save location, defaulting to the user's Downloads folder if no input is provided.
Execution Summary:

Provides a summary of the operation, including the total number of files copied and the location of the CSV report.
Closure:

Waits for user acknowledgment before closing, ensuring the user has time to note the summary details.
Technologies & Libraries:

Python Standard Library: Utilizes os for directory and file manipulation, shutil for file copying, csv for report generation, and datetime for timestamp formatting.
