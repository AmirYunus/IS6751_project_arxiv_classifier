# Data manipulation and analysis
import pandas as pd
import numpy as np
import pickle
# PyTorch
import torch

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

def split_data(df: pd.DataFrame, balance_data: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a DataFrame into train, validation and test sets based on values in 'split' column.
    Separates features (X) and target (y) for each split.

    Args:
        df (pd.DataFrame): The DataFrame to split. Must contain a 'split' column with values
                          'train', 'val' or 'test', and a 'target' column for the labels
        balance_data (bool): Whether to balance the number of samples per category. Default is False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, X_val, X_test, y_train, y_val, y_test arrays

    Raises:
        ValueError: If 'split' column is missing or contains invalid values
    """
    if 'split' not in df.columns:
        raise ValueError("DataFrame must contain a 'split' column")
        
    # Verify split column contains valid values
    valid_splits = {'train', 'val', 'test'}
    invalid_splits = set(df['split'].unique()) - valid_splits
    if invalid_splits:
        raise ValueError(f"Invalid split values found: {invalid_splits}. Must be one of {valid_splits}")

    # Split the data
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    if balance_data:
        train_df = __balance_dataset(train_df)
        val_df = __balance_dataset(val_df)
        test_df = __balance_dataset(test_df)

    # Separate features and target
    X_train = train_df.drop(['split', 'category'], axis=1).values
    X_val = val_df.drop(['split', 'category'], axis=1).values  
    X_test = test_df.drop(['split', 'category'], axis=1).values
    
    y_train = train_df['category'].values
    y_val = val_df['category'].values
    y_test = test_df['category'].values

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")

    if balance_data:
        print("\nSamples per category after balancing:")
        for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            print(f"\n{split_name} set:")
            print(split_df['category'].value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test

def __balance_dataset(split_df):
    # Find minimum number of samples per category
    min_samples = split_df['category'].value_counts().min()
    
    # For each category, keep only first min_samples rows
    balanced_dfs = []
    for category in split_df['category'].unique():
        category_df = split_df[split_df['category'] == category]
        balanced_dfs.append(category_df.head(min_samples))
    
    return pd.concat(balanced_dfs)

def save_model(model: torch.nn.Module, file_path: str) -> None:
    """
    Save a PyTorch model to disk using pickle.
    
    Args:
        model (torch.nn.Module): The PyTorch model to save
        file_path (str): Path where the model should be saved
        
    Returns:
        None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {file_path}")

def load_model(file_path: str) -> torch.nn.Module:
    """
    Load a PyTorch model from disk using pickle.
    
    Args:
        file_path (str): Path to the saved model file
        
    Returns:
        torch.nn.Module: The loaded PyTorch model
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {file_path}")
    return model
