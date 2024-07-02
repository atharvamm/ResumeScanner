import os

def load_env_file(file_path: str = ".env") -> None:
    """
    Load environment variables from a specified file.
    
    This function reads key-value pairs from a file and sets them as environment variables.
    
    :param file_path: Path to the environment file, default is ".env".
    """
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Ignore lines starting with '#' (comments) and empty lines
                if not line.startswith("#") and line.strip():
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print(f"Error: Environment file '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading environment variables: {e}")

def get_root_path() -> str:
    """
    Get the root path of the project.
    
    This function returns the directory name of the grandparent directory of the current file.
    
    :return: The root path as a string.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))