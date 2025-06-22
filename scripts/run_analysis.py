#!/usr/bin/env python3
"""
Analysis Script for LLM Medical Question-Answering Bias Analysis.

Usage:
    python scripts/run_analysis.py [--config CONFIG_PATH]
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.helpers import setup_logging, load_config, create_output_directories
from src.data.data_loader import DataLoader
from src.evaluation.metrics import calculate_correctness_indicators
from src.evaluation.statistical_analysis import run_gender_analysis, run_age_analysis
from src.visualization.plots import save_all_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM Bias Analysis")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['level'], config['logging']['file'])
    logger.info("Starting LLM Medical Question-Answering Bias Analysis")
    
    # Create output directories
    output_dirs = create_output_directories(config)
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load processed data
    input_file = output_dirs['processed_data'] / config['output']['llm_responses_file']
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    logger.info(f"Loading data from: {input_file}")
    df = data_loader.load_csv_data(str(input_file))
    
    # Add correctness indicators
    logger.info("Calculating correctness indicators...")
    df_with_correct = calculate_correctness_indicators(df)
    
    # Run statistical analysis
    logger.info("Running statistical analysis...")
    results = {}
    
    if 'Gender' in df_with_correct.columns:
        results['gender_analysis'] = run_gender_analysis(df_with_correct)
        logger.info(f"Gender analysis: {results['gender_analysis']['test_type']}, "
                   f"p-value: {results['gender_analysis']['p_value']:.4f}")
    
    if 'Age_Category' in df_with_correct.columns:
        results['age_analysis'] = run_age_analysis(df_with_correct)
        logger.info(f"Age analysis: {results['age_analysis']['test_type']}, "
                   f"p-value: {results['age_analysis']['p_value']:.4f}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    saved_plots = save_all_plots(df_with_correct, str(output_dirs['plots']))
    logger.info(f"Generated {len(saved_plots)} plots")
    
    logger.info("Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 