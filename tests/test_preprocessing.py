"""Unit tests for preprocessing functions."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import extract_age, extract_gender, categorize_age


def test_extract_age():
    """Test age extraction functionality."""
    # Test year format
    assert extract_age("A 25-year-old patient") == 25
    assert extract_age("65-year-old man") == 65
    
    # Test month format
    assert extract_age("A 6-month-old infant") == 0.5
    assert extract_age("18-month-old toddler") == 1.5
    
    # Test no age found
    assert extract_age("A patient with symptoms") is None
    assert extract_age("") is None
    assert extract_age(None) is None


def test_extract_gender():
    """Test gender extraction functionality."""
    # Test female
    assert extract_gender("A 30-year-old woman presents") == "Female"
    assert extract_gender("Female patient reports") == "Female"
    
    # Test male
    assert extract_gender("A 45-year-old man presents") == "Male"
    assert extract_gender("Male patient reports") == "Male"
    
    # Test no gender found
    assert extract_gender("A patient presents with") is None
    assert extract_gender("") is None
    assert extract_gender(None) is None


def test_categorize_age():
    """Test age categorization functionality."""
    assert categorize_age(1) == "Toddler"
    assert categorize_age(5) == "Child"
    assert categorize_age(25) == "Adult"
    assert categorize_age(65) == "Senior"
    
    # Test edge cases
    assert categorize_age(3) == "Child"  # Boundary
    assert categorize_age(18) == "Adult"  # Boundary
    assert categorize_age(60) == "Senior"  # Boundary
    assert categorize_age(None) is None
    assert categorize_age(np.nan) is None 