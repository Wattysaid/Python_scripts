"""
Script to clean up Python files by removing main() functions and CLI code
"""

import os
import re

def clean_python_file(file_path):
    """Remove main() functions and if __name__ == "__main__" blocks from Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into lines
        lines = content.split('\n')
        cleaned_lines = []
        skip_rest = False
        
        for line in lines:
            # Check for main function definition or if __name__ == "__main__"
            if (line.strip().startswith('def main(') or 
                line.strip().startswith('if __name__ == "__main__"') or
                line.strip().startswith("if __name__ == '__main__'")):
                skip_rest = True
                break
            
            if not skip_rest:
                cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and cleaned_lines[-1].strip() == '':
            cleaned_lines.pop()
        
        # Write cleaned content back
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Only write if content changed
        if cleaned_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Cleaned: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def find_and_clean_python_files(root_dir):
    """Find all Python files and clean them."""
    cleaned_count = 0
    total_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_count += 1
                
                if clean_python_file(file_path):
                    cleaned_count += 1
    
    print(f"\nSummary: Cleaned {cleaned_count} out of {total_count} Python files")