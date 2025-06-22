"""Together API client for LLM interactions with robust error handling."""

import os
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from together import Together
import backoff

logger = logging.getLogger(__name__)


class TogetherAPIClient:
    """
    Robust client for interacting with Together API for LLM queries.
    
    Features:
    - Automatic retries with exponential backoff
    - Rate limiting
    - Error handling and logging
    - Batch processing capabilities
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Together API client.
        
        Args:
            config: Configuration dictionary containing API settings
            
        Raises:
            ValueError: If API key is not found
        """
        self.config = config
        self.model_id = config['llm']['model_id']
        self.max_retries = config['llm'].get('max_retries', 3)
        self.retry_delay = config['llm'].get('retry_delay', 1.0)
        self.rate_limit_delay = config['llm'].get('rate_limit_delay', 0.5)
        
        # Initialize API client
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable not found. "
                "Please set your Together API key."
            )
        
        self.client = Together(api_key=api_key)
        logger.info(f"Initialized Together API client with model: {self.model_id}")
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=300
    )
    def _make_api_call(self, messages: List[Dict]) -> str:
        """
        Make a single API call with retry logic.
        
        Args:
            messages: List of message dictionaries for the API call
            
        Returns:
            Response content from the API
            
        Raises:
            Exception: If API call fails after all retries
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.config['llm'].get('temperature', 0.1),
                max_tokens=self.config['llm'].get('max_tokens', 10)
            )
            
            response = completion.choices[0].message.content
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return response
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def format_medical_question_prompt(self, question: str, options: Dict[str, str]) -> str:
        """
        Format a medical question into a prompt for the LLM.
        
        Args:
            question: The medical question text
            options: Dictionary of answer options (e.g., {'A': 'option1', 'B': 'option2'})
            
        Returns:
            Formatted prompt string
        """
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        
        prompt = (
            f"{question}\n\n"
            f"{options_text}\n\n"
            "Select the right answer and only give the answer index, i.e., A, B, C, D, E"
        )
        
        return prompt
    
    def get_llm_answer(self, question: str, options: Dict[str, str]) -> str:
        """
        Get LLM answer for a single medical question.
        
        Args:
            question: The medical question text
            options: Dictionary of answer options
            
        Returns:
            LLM response (typically the answer index)
        """
        try:
            prompt = self.format_medical_question_prompt(question, options)
            messages = [{"role": "user", "content": prompt}]
            
            response = self._make_api_call(messages)
            logger.debug(f"LLM response: {response}")
            return response
            
        except Exception as e:
            error_msg = f"Error getting LLM answer: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def process_dataframe_questions(self, df: pd.DataFrame, 
                                  question_col: str = 'question',
                                  options_col: str = 'options',
                                  batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process all questions in a DataFrame to get LLM answers.
        
        Args:
            df: DataFrame containing questions and options
            question_col: Name of the question column
            options_col: Name of the options column
            batch_size: Optional batch size for processing (for progress tracking)
            
        Returns:
            DataFrame with added 'LLM_Answer' column
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [question_col, options_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_processed = df.copy()
        total_questions = len(df_processed)
        
        logger.info(f"Processing {total_questions} questions with LLM")
        
        # Process questions
        llm_answers = []
        for idx, row in df_processed.iterrows():
            if idx % 10 == 0:  # Log progress every 10 questions
                logger.info(f"Processing question {idx + 1}/{total_questions}")
            
            try:
                answer = self.get_llm_answer(row[question_col], row[options_col])
                llm_answers.append(answer)
            except Exception as e:
                logger.error(f"Failed to process question {idx}: {str(e)}")
                llm_answers.append(f"Error: {str(e)}")
        
        df_processed['LLM_Answer'] = llm_answers
        
        logger.info(f"Completed processing {total_questions} questions")
        return df_processed
    
    def batch_process_with_progress(self, df: pd.DataFrame,
                                  question_col: str = 'question',
                                  options_col: str = 'options',
                                  batch_size: int = 50,
                                  save_progress: bool = False,
                                  progress_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process questions in batches with progress saving capability.
        
        Args:
            df: DataFrame containing questions and options
            question_col: Name of the question column
            options_col: Name of the options column
            batch_size: Number of questions to process in each batch
            save_progress: Whether to save progress after each batch
            progress_file: File path to save progress (required if save_progress=True)
            
        Returns:
            DataFrame with LLM answers
        """
        if save_progress and not progress_file:
            raise ValueError("progress_file must be specified when save_progress=True")
        
        df_processed = df.copy()
        df_processed['LLM_Answer'] = None
        
        total_questions = len(df_processed)
        num_batches = (total_questions + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_questions} questions in {num_batches} batches of size {batch_size}")
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_questions)
            
            logger.info(f"Processing batch {batch_num + 1}/{num_batches} (questions {start_idx + 1}-{end_idx})")
            
            # Process batch
            for idx in range(start_idx, end_idx):
                row = df_processed.iloc[idx]
                try:
                    answer = self.get_llm_answer(row[question_col], row[options_col])
                    df_processed.at[idx, 'LLM_Answer'] = answer
                except Exception as e:
                    logger.error(f"Failed to process question {idx}: {str(e)}")
                    df_processed.at[idx, 'LLM_Answer'] = f"Error: {str(e)}"
            
            # Save progress if requested
            if save_progress:
                df_processed.to_csv(progress_file, index=False)
                logger.info(f"Progress saved to {progress_file}")
        
        logger.info("Batch processing completed")
        return df_processed
    
    def validate_llm_responses(self, df: pd.DataFrame, 
                             valid_options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate LLM responses and provide statistics.
        
        Args:
            df: DataFrame with LLM_Answer column
            valid_options: List of valid answer options (e.g., ['A', 'B', 'C', 'D', 'E'])
            
        Returns:
            Dictionary with validation statistics
        """
        if 'LLM_Answer' not in df.columns:
            raise ValueError("DataFrame must contain 'LLM_Answer' column")
        
        total_responses = len(df)
        error_responses = df['LLM_Answer'].str.startswith('Error:').sum()
        valid_responses = total_responses - error_responses
        
        stats = {
            'total_responses': total_responses,
            'valid_responses': valid_responses,
            'error_responses': error_responses,
            'error_rate': error_responses / total_responses if total_responses > 0 else 0,
            'response_distribution': df['LLM_Answer'].value_counts().to_dict()
        }
        
        if valid_options:
            # Check if responses match expected format
            non_error_responses = df[~df['LLM_Answer'].str.startswith('Error:')]['LLM_Answer']
            format_compliant = non_error_responses.isin(valid_options).sum()
            stats['format_compliant_responses'] = format_compliant
            stats['format_compliance_rate'] = format_compliant / valid_responses if valid_responses > 0 else 0
        
        return stats
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """
        Get basic API usage statistics for the session.
        
        Returns:
            Dictionary with usage statistics
        """
        # This is a simplified version - in production, you might want to track
        # actual API usage, costs, tokens used, etc.
        return {
            'model_id': self.model_id,
            'max_retries': self.max_retries,
            'rate_limit_delay': self.rate_limit_delay,
            'configuration': {
                'temperature': self.config['llm'].get('temperature', 0.1),
                'max_tokens': self.config['llm'].get('max_tokens', 10)
            }
        } 