"""General utility functions for the LLM bias analysis project."""

import logging
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration for the project."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Create logger
    logger = logging.getLogger('llm_bias_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = expand_env_vars(config)
    return config


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in configuration values."""
    if isinstance(config, dict):
        return {key: expand_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def create_output_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create all necessary output directories based on configuration."""
    output_config = config.get('output', {})
    
    directories = {
        'raw_data': Path(output_config.get('raw_data_dir', 'data/raw')),
        'processed_data': Path(output_config.get('processed_data_dir', 'data/processed')),
        'results': Path(output_config.get('results_dir', 'data/processed/results')),
        'plots': Path(output_config.get('plots_dir', 'data/processed/results/plots')),
        'logs': Path('logs')
    }
    
    # Create directories
    created_dirs = {}
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        created_dirs[name] = path
    
    return created_dirs 