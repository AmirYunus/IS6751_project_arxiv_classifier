# Data manipulation and analysis
import pandas as pd

# Display utilities
from IPython.display import display

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file and display the first few rows.

    Args:
        file_path (str): The path to the file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        ValueError: If an unsupported file type is provided.
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension == 'bin':
        df = pd.read_pickle(file_path)
    else:
        raise ValueError("Unsupported file type. Use '.csv' or '.bin' files.")
    
    display(df.head())
    return df

def save_data(df: pd.DataFrame, file_path: str, file_type: str = 'pickle') -> None:
    """
    Save a pandas DataFrame to a file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The path where the file will be saved.
        file_type (str): The type of file to save. Either 'csv' or 'pickle'. Default is 'pickle'.

    Returns:
        None

    Raises:
        ValueError: If an unsupported file type is provided.
    """
    if file_type.lower() == 'csv':
        df.to_csv(file_path, index=False)
    elif file_type.lower() == 'pickle':
        df.to_pickle(file_path)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'pickle'.")
    
    print(f"Data saved successfully to {file_path}")