#!/usr/bin/env python3
"""
Data Collection Script for LLM Medical Question-Answering Bias Analysis.

Usage:
    python scripts/run_data_collection.py [--config CONFIG_PATH]
"""

import argparse
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.helpers import setup_logging, load_config, create_output_directories
from src.data.data_loader import DataLoader
from src.data.preprocessing import apply_feature_extraction
from src.llm_interface.together_api import TogetherAPIClient


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM Data Collection")
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
    logger.info("Starting LLM Medical Question-Answering Data Collection")
    
    # Create output directories
    output_dirs = create_output_directories(config)
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load datathon dataset
    track = config['data']['default_track']
    logger.info(f"Loading datathon dataset for track: {track}")
    train_df, test_df = data_loader.load_datathon_dataset(track)
    logger.info(f"Loaded training data: {len(train_df)} samples")
    
    # Apply feature extraction
    logger.info("Applying feature extraction...")
    train_df_processed = apply_feature_extraction(
        train_df,
        text_column=config['data']['text_column'],
        extract_age_flag=config['features']['extract_age'],
        extract_gender_flag=config['features']['extract_gender'],
        categorize_age_flag=True
    )
    
    # Initialize LLM client
    logger.info("Initializing LLM client...")
    llm_client = TogetherAPIClient(config)
    
    # Process questions with LLM
    logger.info("Processing questions with LLM...")
    train_df_with_llm = llm_client.process_dataframe_questions(
        train_df_processed,
        question_col=config['data']['text_column'],
        options_col='options',
        batch_size=config['llm']['batch_size']
    )
    
    # Save results
    output_file = output_dirs['processed_data'] / config['output']['llm_responses_file']
    train_df_with_llm.to_csv(output_file, index=False)
    logger.info(f"LLM responses saved to: {output_file}")
    
    logger.info("Data collection completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 