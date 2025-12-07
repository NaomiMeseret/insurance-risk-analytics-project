"""
Data loading utilities for insurance risk analytics
"""

import pandas as pd
import os
from pathlib import Path


def load_insurance_data(file_path: str = None) -> pd.DataFrame:
    """
    Load insurance claim data from CSV file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file. If None, looks for data in data/ directory.
    
    Returns:
    --------
    pd.DataFrame
        Loaded insurance data
    """
    if file_path is None:
        # Default path
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / "data" / "insurance_data.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found at {file_path}. "
            "Please ensure the data file is in the data/ directory."
        )
    
    # Load data with appropriate data types
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert TransactionMonth to datetime if it exists
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return info

