"""
Tests for data loading functionality
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.data.load_data import load_insurance_data, get_data_info


def test_load_data_file_not_found():
    """Test that load_data raises FileNotFoundError for non-existent file"""
    with pytest.raises(FileNotFoundError):
        load_insurance_data("nonexistent_file.csv")


def test_get_data_info():
    """Test get_data_info returns correct structure"""
    # Create a dummy dataframe
    df = pd.DataFrame({
        'col1': [1, 2, 3, None, 5],
        'col2': ['a', 'b', 'c', 'd', 'e']
    })
    
    info = get_data_info(df)
    
    assert 'shape' in info
    assert 'columns' in info
    assert 'dtypes' in info
    assert 'missing_values' in info
    assert 'missing_percentage' in info
    assert 'memory_usage_mb' in info
    assert info['shape'] == (5, 2)
    assert len(info['columns']) == 2

