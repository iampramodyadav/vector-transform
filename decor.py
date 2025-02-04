import os
import sys
import shutil
import time
from datetime import datetime
# from pathlib import Path
import subprocess
import winreg
import platform
def open_in_notepadpp(file_path):
    """
    Open a file in Notepad++
    
    Args:
        file_path (str): Path to the file to open
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert to absolute path
    file_path = os.path.abspath(file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return False
        
    # List of common Notepad++ installation paths
    possible_paths = [
        r"C:\Program Files\Notepad++\notepad++.exe",
        # r"C:\Program Files (x86)\Notepad++\notepad++.exe",
    ]
    
    # Try to get Notepad++ path from registry
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\notepad++.exe",
                           0, winreg.KEY_READ) as key:
            notepadpp_path = winreg.QueryValue(key, None)
            possible_paths.insert(0, notepadpp_path)  # Add to start of list
    except WindowsError:
        pass
    
    # Try each possible path
    for notepadpp_path in possible_paths:
        if os.path.exists(notepadpp_path):
            try:
                subprocess.Popen([notepadpp_path, file_path])
                return True
            except subprocess.SubprocessError as e:
                print(f"Error launching Notepad++: {e}")
                continue
    
    print("Error: Notepad++ not found. Please ensure it is installed.")
    return False
# --------------------------------------------------------------------------------
def get_next_available_filename(base_filename):
    """Generate next available filename with pattern filename{i}"""
    # Split filename into name and extension
    name, ext = os.path.splitext(base_filename)
    
    # If file doesn't exist, return original filename
    if not os.path.exists(base_filename):
        return base_filename
    
    i = 0
    while True:
        new_filename = f"{name}{i}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        i += 1
# --------------------------------------------------------------------------------
def redirect_prints(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Save the original stdout
            original_stdout = sys.stdout
            
            # Get next available filename
            actual_filename = get_next_available_filename(filename)
            # Open the file and redirect stdout
            with open(actual_filename, 'w') as f:
                sys.stdout = f
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Restore stdout even if there's an error
                    sys.stdout = original_stdout
            
            # Print which file was created (to the original stdout)
            print(f"Output saved to: {actual_filename}")
            
            #----------- open file in notepad++:start ------------
            try:
                 if open_in_notepadpp(actual_filename):
                    print(f"Successfully opened {actual_filename} in Notepad++")
                 else:
                    print("Failed to open file in Notepad++")
            except:
                print("Failed to open file in Notepad++")
            #----------- open file in notepad++:end ------------
            
            return result
        return wrapper
    return decorator
# --------------------------------------------------------------------------------
