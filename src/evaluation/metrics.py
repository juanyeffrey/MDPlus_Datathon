"""Evaluation metrics for LLM performance assessment."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def calculate_accuracy_metrics(df: pd.DataFrame, 
                             llm_answer_col: str = 'LLM_Answer',
                             correct_answer_col: str = 'answer_idx') -> Dict[str, float]:
    """
    Calculate basic accuracy metrics for LLM responses.
    
    Args:
        df: DataFrame containing LLM answers and correct answers
        llm_answer_col: Column name for LLM answers
        correct_answer_col: Column name for correct answers
        
    Returns:
        Dictionary containing accuracy metrics
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = [llm_answer_col, correct_answer_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out error responses
    valid_df = df[~df[llm_answer_col].str.startswith('Error:', na=False)].copy()
    
    if len(valid_df) == 0:
        logger.warning("No valid responses found for accuracy calculation")
        return {
            'overall_accuracy': 0.0,
            'total_questions': len(df),
            'valid_responses': 0,
            'error_responses': len(df)
        }
    
    # Calculate correctness
    correct_predictions = (valid_df[llm_answer_col] == valid_df[correct_answer_col]).sum()
    total_valid = len(valid_df)
    
    accuracy = correct_predictions / total_valid if total_valid > 0 else 0.0
    
    metrics = {
        'overall_accuracy': accuracy,
        'total_questions': len(df),
        'valid_responses': total_valid,
        'error_responses': len(df) - total_valid,
        'correct_predictions': correct_predictions,
        'accuracy_percentage': accuracy * 100
    }
    
    logger.info(f"Overall accuracy: {accuracy:.4f} ({correct_predictions}/{total_valid})")
    return metrics


def calculate_correctness_indicators(df: pd.DataFrame,
                                   llm_answer_col: str = 'LLM_Answer',
                                   correct_answer_col: str = 'answer_idx') -> pd.DataFrame:
    """
    Add correctness indicator column to DataFrame.
    
    Args:
        df: DataFrame containing LLM answers and correct answers
        llm_answer_col: Column name for LLM answers
        correct_answer_col: Column name for correct answers
        
    Returns:
        DataFrame with added 'correct' column (1 for correct, 0 for incorrect)
    """
    df_with_correct = df.copy()
    
    # Create correctness indicator (1 for correct, 0 for incorrect)
    # Handle error responses as incorrect
    df_with_correct['correct'] = (
        (df_with_correct[llm_answer_col] == df_with_correct[correct_answer_col]) &
        (~df_with_correct[llm_answer_col].str.startswith('Error:', na=False))
    ).astype(int)
    
    correct_count = df_with_correct['correct'].sum()
    logger.info(f"Added correctness indicators: {correct_count}/{len(df_with_correct)} correct")
    
    return df_with_correct


def calculate_demographic_accuracy(df: pd.DataFrame, 
                                 demographic_col: str,
                                 correct_col: str = 'correct') -> pd.DataFrame:
    """
    Calculate accuracy metrics by demographic groups.
    
    Args:
        df: DataFrame with correctness indicators and demographic information
        demographic_col: Column name for demographic grouping (e.g., 'Gender', 'Age_Category')
        correct_col: Column name for correctness indicators
        
    Returns:
        DataFrame with accuracy metrics by demographic group
    """
    if demographic_col not in df.columns:
        raise ValueError(f"Demographic column '{demographic_col}' not found in DataFrame")
    
    if correct_col not in df.columns:
        raise ValueError(f"Correctness column '{correct_col}' not found in DataFrame")
    
    # Filter out rows with missing demographic data
    df_filtered = df.dropna(subset=[demographic_col])
    
    # Calculate accuracy by demographic group
    accuracy_by_demo = df_filtered.groupby(demographic_col).agg({
        correct_col: ['count', 'sum', 'mean', 'std']
    }).round(4)
    
    # Flatten column names
    accuracy_by_demo.columns = ['total_count', 'correct_count', 'accuracy', 'accuracy_std']
    accuracy_by_demo = accuracy_by_demo.reset_index()
    
    # Calculate confidence intervals (assuming normal distribution)
    accuracy_by_demo['se'] = accuracy_by_demo['accuracy_std'] / np.sqrt(accuracy_by_demo['total_count'])
    accuracy_by_demo['ci_lower'] = accuracy_by_demo['accuracy'] - 1.96 * accuracy_by_demo['se']
    accuracy_by_demo['ci_upper'] = accuracy_by_demo['accuracy'] + 1.96 * accuracy_by_demo['se']
    
    # Clean up confidence intervals (bound between 0 and 1)
    accuracy_by_demo['ci_lower'] = accuracy_by_demo['ci_lower'].clip(lower=0)
    accuracy_by_demo['ci_upper'] = accuracy_by_demo['ci_upper'].clip(upper=1)
    
    logger.info(f"Calculated accuracy by {demographic_col}: {len(accuracy_by_demo)} groups")
    return accuracy_by_demo


def calculate_prompt_accuracy(df: pd.DataFrame,
                            prompt_col: str = 'Prompt ID',
                            correct_col: str = 'correct') -> pd.DataFrame:
    """
    Calculate accuracy metrics by prompt.
    
    Args:
        df: DataFrame with correctness indicators and prompt information
        prompt_col: Column name for prompt grouping
        correct_col: Column name for correctness indicators
        
    Returns:
        DataFrame with accuracy metrics by prompt
    """
    return calculate_demographic_accuracy(df, prompt_col, correct_col)


def calculate_interaction_accuracy(df: pd.DataFrame,
                                 factor1_col: str,
                                 factor2_col: str,
                                 correct_col: str = 'correct') -> pd.DataFrame:
    """
    Calculate accuracy metrics by interaction of two factors.
    
    Args:
        df: DataFrame with correctness indicators
        factor1_col: First factor column (e.g., 'Prompt ID')
        factor2_col: Second factor column (e.g., 'Gender')
        correct_col: Column name for correctness indicators
        
    Returns:
        DataFrame with accuracy metrics by factor interaction
    """
    required_cols = [factor1_col, factor2_col, correct_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out rows with missing data
    df_filtered = df.dropna(subset=[factor1_col, factor2_col])
    
    # Calculate accuracy by interaction
    accuracy_by_interaction = df_filtered.groupby([factor1_col, factor2_col]).agg({
        correct_col: ['count', 'sum', 'mean', 'std']
    }).round(4)
    
    # Flatten column names
    accuracy_by_interaction.columns = ['total_count', 'correct_count', 'accuracy', 'accuracy_std']
    accuracy_by_interaction = accuracy_by_interaction.reset_index()
    
    # Calculate standard error and confidence intervals
    accuracy_by_interaction['se'] = (
        accuracy_by_interaction['accuracy_std'] / 
        np.sqrt(accuracy_by_interaction['total_count'])
    )
    accuracy_by_interaction['ci_lower'] = (
        accuracy_by_interaction['accuracy'] - 1.96 * accuracy_by_interaction['se']
    ).clip(lower=0)
    accuracy_by_interaction['ci_upper'] = (
        accuracy_by_interaction['accuracy'] + 1.96 * accuracy_by_interaction['se']
    ).clip(upper=1)
    
    logger.info(f"Calculated accuracy by {factor1_col} x {factor2_col}: {len(accuracy_by_interaction)} combinations")
    return accuracy_by_interaction


def generate_accuracy_summary(df: pd.DataFrame, 
                            demographic_cols: List[str] = None,
                            prompt_col: str = 'Prompt ID',
                            correct_col: str = 'correct') -> Dict[str, Any]:
    """
    Generate comprehensive accuracy summary across all dimensions.
    
    Args:
        df: DataFrame with all necessary columns
        demographic_cols: List of demographic columns to analyze
        prompt_col: Prompt column name
        correct_col: Correctness column name
        
    Returns:
        Dictionary containing comprehensive accuracy summary
    """
    if demographic_cols is None:
        demographic_cols = ['Gender', 'Age_Category']
    
    summary = {}
    
    # Overall accuracy
    summary['overall'] = calculate_accuracy_metrics(df, correct_answer_col=correct_col)
    
    # Accuracy by prompt
    if prompt_col in df.columns:
        summary['by_prompt'] = calculate_prompt_accuracy(df, prompt_col, correct_col)
    
    # Accuracy by demographics
    summary['by_demographics'] = {}
    for demo_col in demographic_cols:
        if demo_col in df.columns:
            summary['by_demographics'][demo_col] = calculate_demographic_accuracy(df, demo_col, correct_col)
    
    # Interaction effects
    summary['interactions'] = {}
    if prompt_col in df.columns:
        for demo_col in demographic_cols:
            if demo_col in df.columns:
                interaction_key = f"{prompt_col}_x_{demo_col}"
                summary['interactions'][interaction_key] = calculate_interaction_accuracy(
                    df, prompt_col, demo_col, correct_col
                )
    
    logger.info("Generated comprehensive accuracy summary")
    return summary


def compare_accuracy_across_groups(df: pd.DataFrame,
                                 group_col: str,
                                 correct_col: str = 'correct') -> Dict[str, Any]:
    """
    Compare accuracy across different groups and identify potential disparities.
    
    Args:
        df: DataFrame with correctness indicators and grouping variable
        group_col: Column name for grouping
        correct_col: Column name for correctness indicators
        
    Returns:
        Dictionary with comparison statistics
    """
    accuracy_by_group = calculate_demographic_accuracy(df, group_col, correct_col)
    
    if len(accuracy_by_group) < 2:
        return {"error": "Need at least 2 groups for comparison"}
    
    # Calculate disparity metrics
    max_accuracy = accuracy_by_group['accuracy'].max()
    min_accuracy = accuracy_by_group['accuracy'].min()
    accuracy_range = max_accuracy - min_accuracy
    
    # Identify highest and lowest performing groups
    best_group = accuracy_by_group.loc[accuracy_by_group['accuracy'].idxmax(), group_col]
    worst_group = accuracy_by_group.loc[accuracy_by_group['accuracy'].idxmin(), group_col]
    
    comparison = {
        'accuracy_by_group': accuracy_by_group,
        'max_accuracy': max_accuracy,
        'min_accuracy': min_accuracy,
        'accuracy_range': accuracy_range,
        'relative_disparity': accuracy_range / max_accuracy if max_accuracy > 0 else 0,
        'best_performing_group': best_group,
        'worst_performing_group': worst_group,
        'num_groups': len(accuracy_by_group)
    }
    
    logger.info(f"Accuracy comparison across {group_col}: range = {accuracy_range:.4f}")
    return comparison 