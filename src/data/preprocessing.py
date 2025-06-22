"""Data preprocessing utilities for feature extraction and data consolidation."""

import re
import logging
import pandas as pd
from typing import Optional, List, Dict, Union

logger = logging.getLogger(__name__)


def extract_age(text: str) -> Optional[float]:
    """
    Extract age from text using regex patterns.
    
    Supports the following formats:
    - X-week-old (converted to years)
    - X-month-old (converted to years) 
    - X-year-old
    
    Args:
        text: Text to extract age from
        
    Returns:
        Age in years as float, or None if no age found
        
    Examples:
        >>> extract_age("A 25-year-old patient")
        25
        >>> extract_age("A 6-month-old infant")
        0.5
        >>> extract_age("A 2-week-old newborn")
        0.04
    """
    if not isinstance(text, str):
        return None
        
    # Check for week-old pattern
    match_week = re.search(r'\b(\d+)-week-old\b', text, re.IGNORECASE)
    if match_week:
        weeks = int(match_week.group(1))
        return round(weeks / 52.0, 2)
    
    # Check for month-old pattern
    match_month = re.search(r'\b(\d+)-month-old\b', text, re.IGNORECASE)
    if match_month:
        months = int(match_month.group(1))
        return round(months / 12.0, 2)
    
    # Check for year-old pattern
    match_year = re.search(r'\b(\d+)-year-old\b', text, re.IGNORECASE)
    if match_year:
        return int(match_year.group(1))
    
    return None


def extract_gender(text: str) -> Optional[str]:
    """
    Extract gender from text using keyword matching.
    
    Args:
        text: Text to extract gender from
        
    Returns:
        'Male', 'Female', or None if no gender indicators found
        
    Examples:
        >>> extract_gender("A 30-year-old woman presents with...")
        'Female'
        >>> extract_gender("A male patient reports...")
        'Male'
    """
    if not isinstance(text, str):
        return None
        
    text_lower = text.lower()
    
    # Check for female indicators
    female_terms = ['woman', 'female', 'girl', 'lady', 'she', 'her']
    if any(term in text_lower for term in female_terms):
        return 'Female'
    
    # Check for male indicators
    male_terms = ['man', 'male', 'boy', 'gentleman', 'he', 'his', 'him']
    if any(term in text_lower for term in male_terms):
        return 'Male'
    
    return None


def categorize_age(age: Union[int, float]) -> Optional[str]:
    """
    Categorize age into predefined age groups.
    
    Age categories:
    - Toddler: < 3 years
    - Child: 3-17 years
    - Adult: 18-59 years  
    - Senior: >= 60 years
    
    Args:
        age: Age in years
        
    Returns:
        Age category as string, or None if age is None or invalid
        
    Examples:
        >>> categorize_age(2)
        'Toddler'
        >>> categorize_age(25)
        'Adult'
        >>> categorize_age(65)
        'Senior'
    """
    if age is None or pd.isna(age):
        return None
        
    try:
        age = float(age)
        if age < 0:
            return None
            
        if age < 3:
            return "Toddler"
        elif 3 <= age < 18:
            return "Child"
        elif 18 <= age < 60:
            return "Adult"
        else:
            return "Senior"
            
    except (ValueError, TypeError):
        return None


def assign_prompt_ids(dataframes: Dict[str, pd.DataFrame], 
                     prompt_mapping: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Assign prompt IDs to multiple dataframes.
    
    Args:
        dataframes: Dictionary with dataset names as keys and DataFrames as values
        prompt_mapping: Optional mapping from dataset names to prompt IDs.
                       If None, uses default naming (Prompt 1, Prompt 2, etc.)
    
    Returns:
        Dictionary of DataFrames with 'Prompt ID' column added
        
    Examples:
        >>> dfs = {'data1': df1, 'data2': df2}
        >>> result = assign_prompt_ids(dfs)
        >>> # Each DataFrame will have a 'Prompt ID' column
    """
    if prompt_mapping is None:
        # Create default mapping
        prompt_mapping = {name: f"Prompt {i+1}" 
                         for i, name in enumerate(dataframes.keys())}
    
    processed_dfs = {}
    
    for name, df in dataframes.items():
        df_copy = df.copy()
        prompt_id = prompt_mapping.get(name, f"Prompt {name}")
        df_copy['Prompt ID'] = prompt_id
        processed_dfs[name] = df_copy
        logger.info(f"Assigned '{prompt_id}' to dataset '{name}' with {len(df_copy)} rows")
    
    return processed_dfs


def consolidate_datasets(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Consolidate multiple DataFrames into a single DataFrame.
    
    Args:
        dataframes: List of DataFrames to consolidate
        
    Returns:
        Consolidated DataFrame with reset index
        
    Raises:
        ValueError: If no dataframes provided or if dataframes have incompatible schemas
    """
    if not dataframes:
        raise ValueError("No dataframes provided for consolidation")
    
    # Check that all dataframes have compatible columns
    base_columns = set(dataframes[0].columns)
    for i, df in enumerate(dataframes[1:], 1):
        if set(df.columns) != base_columns:
            logger.warning(f"DataFrame {i} has different columns than DataFrame 0")
    
    logger.info(f"Consolidating {len(dataframes)} datasets")
    
    # Concatenate all dataframes
    consolidated_df = pd.concat(dataframes, ignore_index=True)
    
    logger.info(f"Consolidated dataset shape: {consolidated_df.shape}")
    return consolidated_df


def apply_feature_extraction(df: pd.DataFrame, 
                           text_column: str = 'question',
                           extract_age_flag: bool = True,
                           extract_gender_flag: bool = True,
                           categorize_age_flag: bool = True) -> pd.DataFrame:
    """
    Apply feature extraction to a DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Name of the column containing text to extract features from
        extract_age_flag: Whether to extract age
        extract_gender_flag: Whether to extract gender
        categorize_age_flag: Whether to categorize age
        
    Returns:
        DataFrame with extracted features
        
    Raises:
        ValueError: If text_column doesn't exist in DataFrame
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    df_processed = df.copy()
    
    if extract_age_flag:
        logger.info("Extracting age features...")
        df_processed['Age'] = df_processed[text_column].apply(extract_age)
        age_extracted_count = df_processed['Age'].notna().sum()
        logger.info(f"Age extracted for {age_extracted_count}/{len(df_processed)} records")
    
    if extract_gender_flag:
        logger.info("Extracting gender features...")
        df_processed['Gender'] = df_processed[text_column].apply(extract_gender)
        gender_extracted_count = df_processed['Gender'].notna().sum()
        logger.info(f"Gender extracted for {gender_extracted_count}/{len(df_processed)} records")
    
    if categorize_age_flag and 'Age' in df_processed.columns:
        logger.info("Categorizing age...")
        df_processed['Age_Category'] = df_processed['Age'].apply(categorize_age)
        age_categorized_count = df_processed['Age_Category'].notna().sum()
        logger.info(f"Age categorized for {age_categorized_count}/{len(df_processed)} records")
    
    return df_processed


def validate_extracted_features(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Validate extracted features and provide summary statistics.
    
    Args:
        df: DataFrame with extracted features
        
    Returns:
        Dictionary containing validation results and statistics
    """
    validation_results = {}
    
    # Age validation
    if 'Age' in df.columns:
        age_stats = {
            'total_records': len(df),
            'age_extracted': df['Age'].notna().sum(),
            'age_missing': df['Age'].isna().sum(),
            'age_min': df['Age'].min() if df['Age'].notna().any() else None,
            'age_max': df['Age'].max() if df['Age'].notna().any() else None,
            'age_mean': df['Age'].mean() if df['Age'].notna().any() else None,
            'invalid_ages': (df['Age'] < 0).sum() if df['Age'].notna().any() else 0
        }
        validation_results['age'] = age_stats
    
    # Gender validation
    if 'Gender' in df.columns:
        gender_stats = {
            'total_records': len(df),
            'gender_extracted': df['Gender'].notna().sum(),
            'gender_missing': df['Gender'].isna().sum(),
            'gender_distribution': df['Gender'].value_counts().to_dict()
        }
        validation_results['gender'] = gender_stats
    
    # Age category validation
    if 'Age_Category' in df.columns:
        age_category_stats = {
            'total_records': len(df),
            'category_assigned': df['Age_Category'].notna().sum(),
            'category_missing': df['Age_Category'].isna().sum(),
            'category_distribution': df['Age_Category'].value_counts().to_dict()
        }
        validation_results['age_category'] = age_category_stats
    
    return validation_results 