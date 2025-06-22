"""Data loading utilities for medical datathon datasets."""

import logging
import pandas as pd
import datasets
from typing import Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of medical datathon datasets from various sources."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        # Default track options for simplified config
        self.track_options = {
            "Medical Education": "meded",
            "Clinical Documentation": "clindoc", 
            "Mental Health": "mentalhealth"
        }
        
    def load_datathon_dataset(self, track: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load medical datathon dataset from Hugging Face.
        
        Args:
            track: Track name (e.g., 'Medical Education', 'Clinical Documentation')
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            ValueError: If track is not supported
            Exception: If dataset loading fails
        """
        if track not in self.track_options:
            raise ValueError(f"Unsupported track: {track}. Available tracks: {list(self.track_options.keys())}")
            
        track_id = self.track_options[track]
        logger.info(f"Loading datathon dataset for track: {track} (ID: {track_id})")
        
        try:
            ds = datasets.load_dataset(
                self.config['data']['datathon_dataset_name'], 
                data_dir=track_id
            )
            
            train_df = ds["train"].to_pandas()
            test_df = ds["test"].to_pandas()
            
            logger.info(f"Successfully loaded {len(train_df)} training samples and {len(test_df)} test samples")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to load datathon dataset: {str(e)}")
            raise
    
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        logger.info(f"Loading CSV data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {str(e)}")
            raise
    
    def load_multiple_csv_files(self, file_paths: list) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files and return as a dictionary.
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            Dictionary with file names as keys and DataFrames as values
        """
        datasets = {}
        
        for file_path in file_paths:
            file_path = Path(file_path)
            filename = file_path.stem
            
            try:
                datasets[filename] = self.load_csv_data(file_path)
                logger.info(f"Loaded {filename}: {len(datasets[filename])} rows")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {str(e)}")
                
        return datasets
    
    def validate_dataset_schema(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that a DataFrame has the required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all required columns are present, False otherwise
        """
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        logger.info("Dataset schema validation passed")
        return True
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get basic information about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Add meta_info value counts if present
        if 'meta_info' in df.columns:
            info['meta_info_counts'] = df['meta_info'].value_counts().to_dict()
            
        return info 