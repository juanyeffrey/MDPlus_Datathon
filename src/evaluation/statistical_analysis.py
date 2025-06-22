"""Statistical analysis utilities for bias detection and hypothesis testing."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats

logger = logging.getLogger(__name__)


def test_normality(data: np.ndarray) -> Dict[str, Any]:
    """Test normality of data distribution."""
    clean_data = data[~np.isnan(data)] if len(data) > 0 else np.array([])
    
    if len(clean_data) < 3:
        return {'is_normal': False, 'sample_size': len(clean_data)}
    
    try:
        statistic, p_value = stats.shapiro(clean_data)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'sample_size': len(clean_data)
        }
    except Exception as e:
        logger.error(f"Normality test failed: {str(e)}")
        return {'is_normal': False, 'sample_size': len(clean_data)}


def compare_two_groups(group1: np.ndarray, group2: np.ndarray,
                      group1_name: str = "Group 1", group2_name: str = "Group 2") -> Dict[str, Any]:
    """Compare two groups using appropriate statistical test."""
    clean_group1 = group1[~np.isnan(group1)]
    clean_group2 = group2[~np.isnan(group2)]
    
    if len(clean_group1) < 2 or len(clean_group2) < 2:
        return {
            'test_type': 'Insufficient data',
            'p_value': np.nan,
            'significant': False
        }
    
    # Test normality
    norm1 = test_normality(clean_group1)
    norm2 = test_normality(clean_group2)
    
    # Choose appropriate test
    if norm1['is_normal'] and norm2['is_normal']:
        statistic, p_value = stats.ttest_ind(clean_group1, clean_group2)
        test_type = "t-test"
    else:
        statistic, p_value = stats.mannwhitneyu(clean_group1, clean_group2, alternative='two-sided')
        test_type = "Mann-Whitney U"
    
    return {
        'test_type': test_type,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'group1_mean': np.mean(clean_group1),
        'group2_mean': np.mean(clean_group2)
    }


def compare_multiple_groups(groups: List[np.ndarray], group_names: List[str] = None) -> Dict[str, Any]:
    """Compare multiple groups using ANOVA or Kruskal-Wallis test."""
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(len(groups))]
    
    # Clean data
    clean_groups = [group[~np.isnan(group)] for group in groups]
    clean_groups = [group for group in clean_groups if len(group) > 1]
    
    if len(clean_groups) < 2:
        return {
            'test_type': 'Insufficient data',
            'p_value': np.nan,
            'significant': False
        }
    
    # Test normality for each group
    all_normal = all(test_normality(group)['is_normal'] for group in clean_groups)
    
    # Choose appropriate test
    if all_normal:
        statistic, p_value = stats.f_oneway(*clean_groups)
        test_type = "ANOVA"
    else:
        statistic, p_value = stats.kruskal(*clean_groups)
        test_type = "Kruskal-Wallis"
    
    return {
        'test_type': test_type,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'num_groups': len(clean_groups)
    }


def run_gender_analysis(df: pd.DataFrame, 
                       correct_col: str = 'correct',
                       gender_col: str = 'Gender') -> Dict[str, Any]:
    """Run gender comparison analysis."""
    gender_data = df.dropna(subset=[gender_col, correct_col])
    
    if len(gender_data) == 0:
        return {'error': 'No valid data for gender analysis'}
    
    male_data = gender_data[gender_data[gender_col] == 'Male'][correct_col].values
    female_data = gender_data[gender_data[gender_col] == 'Female'][correct_col].values
    
    result = compare_two_groups(male_data, female_data, "Male", "Female")
    logger.info("Completed gender comparison analysis")
    return result


def run_age_analysis(df: pd.DataFrame,
                    correct_col: str = 'correct',
                    age_col: str = 'Age_Category') -> Dict[str, Any]:
    """Run age group analysis."""
    age_data = df.dropna(subset=[age_col, correct_col])
    
    if len(age_data) == 0:
        return {'error': 'No valid data for age analysis'}
    
    # Group data by age category
    age_groups = []
    group_names = []
    for age_group in age_data[age_col].unique():
        group_data = age_data[age_data[age_col] == age_group][correct_col].values
        if len(group_data) > 1:
            age_groups.append(group_data)
            group_names.append(age_group)
    
    result = compare_multiple_groups(age_groups, group_names)
    logger.info("Completed age group analysis")
    return result 