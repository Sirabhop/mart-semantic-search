import os
import json

def read_json(file_path, relative_path="/Workspace/Shared/ds-projects/mart-search/poc/data/"):
    """
    Reads and parses JSON data from a file.

    Args:
        file_path (str): The name of the JSON file.
        relative_path (str, optional): The base directory for the file path. Defaults to your specified project folder.

    Returns:
        dict or list: The parsed JSON data (usually a dictionary or a list).
    """
    
    # Construct the full file path using os.path.join for better cross-platform compatibility
    full_path = os.path.join(relative_path, file_path)
    try:
        with open(full_path, 'r') as infile:
            data = json.load(infile)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}") 