import os

# Load .env file
def load_env_file(file_path=".env"):
    """
    Load environment variables from a file.

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

def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))