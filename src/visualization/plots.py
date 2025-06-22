"""Visualization utilities for generating plots and charts."""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")


def plot_accuracy_by_gender(df: pd.DataFrame,
                          prompt_col: str = 'Prompt ID',
                          gender_col: str = 'Gender',
                          correct_col: str = 'correct',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 7)) -> plt.Figure:
    """Generate bar plot showing accuracy by prompt and gender with error bars."""
    # Calculate accuracy and standard error
    grouped = df.groupby([prompt_col, gender_col])[correct_col]
    mean_accuracy = grouped.mean().reset_index(name='mean_correct')
    sem_accuracy = grouped.sem().reset_index(name='sem_correct')
    
    # Merge and pivot
    plot_data = pd.merge(mean_accuracy, sem_accuracy, on=[prompt_col, gender_col])
    mean_pivot = pd.pivot_table(plot_data, values='mean_correct', 
                               index=prompt_col, columns=gender_col)
    sem_pivot = pd.pivot_table(plot_data, values='sem_correct', 
                              index=prompt_col, columns=gender_col)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    mean_pivot.plot(kind='bar', yerr=sem_pivot, ax=ax, capsize=4, alpha=0.8)
    
    ax.set_title('LLM Accuracy by Prompt and Gender', fontsize=16, fontweight='bold')
    ax.set_xlabel('Prompt ID', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved gender accuracy plot to {save_path}")
    
    return fig


def plot_accuracy_by_age(df: pd.DataFrame,
                        prompt_col: str = 'Prompt ID',
                        age_col: str = 'Age_Category',
                        correct_col: str = 'correct',
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 7)) -> plt.Figure:
    """Generate bar plot showing accuracy by prompt and age category."""
    # Ensure proper age ordering
    age_order = ['Toddler', 'Child', 'Adult', 'Senior']
    df_copy = df.copy()
    df_copy[age_col] = pd.Categorical(df_copy[age_col], categories=age_order, ordered=True)
    
    # Calculate accuracy and standard error
    grouped = df_copy.groupby([prompt_col, age_col])[correct_col]
    mean_accuracy = grouped.mean().reset_index(name='mean_correct')
    sem_accuracy = grouped.sem().reset_index(name='sem_correct')
    
    # Merge and pivot
    plot_data = pd.merge(mean_accuracy, sem_accuracy, on=[prompt_col, age_col])
    mean_pivot = pd.pivot_table(plot_data, values='mean_correct', 
                               index=prompt_col, columns=age_col)
    sem_pivot = pd.pivot_table(plot_data, values='sem_correct', 
                              index=prompt_col, columns=age_col)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    mean_pivot.plot(kind='bar', yerr=sem_pivot, ax=ax, capsize=4, alpha=0.8)
    
    ax.set_title('LLM Accuracy by Prompt and Age Category', fontsize=16, fontweight='bold')
    ax.set_xlabel('Prompt ID', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Age Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved age accuracy plot to {save_path}")
    
    return fig


def plot_age_distribution(df: pd.DataFrame,
                         age_col: str = 'Age',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Generate histogram showing age distribution."""
    age_data = df[age_col].dropna()
    
    if len(age_data) == 0:
        logger.warning("No age data available for plotting")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(age_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    ax.set_title('Age Distribution in Dataset', fontsize=16, fontweight='bold')
    ax.set_xlabel('Age (years)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add statistics
    mean_age = age_data.mean()
    stats_text = f'Mean: {mean_age:.1f}\nN: {len(age_data)}'
    ax.text(0.75, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved age distribution plot to {save_path}")
    
    return fig


def save_all_plots(df: pd.DataFrame, 
                  output_dir: str = "data/processed/results/plots",
                  prefix: str = "llm_bias_analysis") -> Dict[str, str]:
    """Generate and save all visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    plots_to_generate = [
        ('gender_accuracy', lambda: plot_accuracy_by_gender(df)),
        ('age_accuracy', lambda: plot_accuracy_by_age(df)),
        ('age_distribution', lambda: plot_age_distribution(df))
    ]
    
    for plot_name, plot_func in plots_to_generate:
        try:
            file_path = output_path / f"{prefix}_{plot_name}.png"
            fig = plot_func()
            
            if fig is not None:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_plots[plot_name] = str(file_path)
                logger.info(f"Saved {plot_name} to {file_path}")
                
        except Exception as e:
            logger.error(f"Error generating {plot_name}: {str(e)}")
    
    return saved_plots 