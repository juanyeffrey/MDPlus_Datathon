# Usage Examples

This document provides practical examples of using the LLM Medical Question-Answering Bias Analysis tool.

## Basic Workflow

### 1. Environment Setup

```bash
# Set your Together AI API key
export TOGETHER_API_KEY="your_api_key_here"  # On Windows: set TOGETHER_API_KEY=your_api_key_here

# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Data Collection

```bash
# Run data collection with default settings
python scripts/run_data_collection.py

# Use custom config file
python scripts/run_data_collection.py --config custom_config.yaml
```

**Expected Output:**
```
2024-01-15 10:30:15 - INFO - Starting LLM Medical Question-Answering Data Collection
2024-01-15 10:30:16 - INFO - Loading datathon dataset for track: Medical Education
2024-01-15 10:30:20 - INFO - Loaded training data: 1500 samples
2024-01-15 10:30:21 - INFO - Applying feature extraction...
2024-01-15 10:30:25 - INFO - Initializing LLM client...
2024-01-15 10:30:26 - INFO - Processing questions with LLM...
2024-01-15 10:35:42 - INFO - LLM responses saved to: data/processed/llm_responses.csv
2024-01-15 10:35:42 - INFO - Data collection completed successfully!
```

### 3. Statistical Analysis

```bash
# Run analysis on collected data
python scripts/run_analysis.py

# Use custom config
python scripts/run_analysis.py --config custom_config.yaml
```

**Expected Output:**
```
2024-01-15 10:40:15 - INFO - Starting LLM Medical Question-Answering Bias Analysis
2024-01-15 10:40:16 - INFO - Loading data from: data/processed/llm_responses.csv
2024-01-15 10:40:17 - INFO - Calculating correctness indicators...
2024-01-15 10:40:18 - INFO - Running statistical analysis...
2024-01-15 10:40:19 - INFO - Gender analysis: t-test, p-value: 0.0234
2024-01-15 10:40:20 - INFO - Age analysis: ANOVA, p-value: 0.1456
2024-01-15 10:40:21 - INFO - Generating visualizations...
2024-01-15 10:40:25 - INFO - Generated 3 plots
2024-01-15 10:40:25 - INFO - Analysis completed successfully!
```

## Configuration Examples

### Custom Model Configuration

```yaml
# config/custom_config.yaml
llm:
  model_id: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # Use larger model
  temperature: 0.0  # More deterministic
  batch_size: 25    # Smaller batches for larger model
```

### Custom Age Categories

```yaml
# config/pediatric_focus.yaml
features:
  age_categories:
    "Newborn": [0, 0.1]
    "Infant": [0.1, 2]
    "Child": [2, 12]
    "Adolescent": [12, 18]
    "Adult": [18, 150]
```

## Programmatic Usage

### Using Core Functions

```python
from src.data.data_loader import DataLoader
from src.data.preprocessing import extract_age, extract_gender, categorize_age
from src.evaluation.statistical_analysis import run_gender_analysis
from src.utils.helpers import load_config

# Load configuration
config = load_config("config/config.yaml")

# Load data
data_loader = DataLoader(config)
train_df, test_df = data_loader.load_datathon_dataset("Medical Education")

# Extract features
ages = train_df['question'].apply(extract_age)
genders = train_df['question'].apply(extract_gender)
age_categories = ages.apply(categorize_age)

# Add to dataframe
train_df['Age'] = ages
train_df['Gender'] = genders
train_df['Age_Category'] = age_categories

# Run statistical analysis (after LLM processing)
# Assuming you have a 'correct' column
gender_results = run_gender_analysis(train_df)
print(f"Gender bias test: {gender_results['test_type']}")
print(f"P-value: {gender_results['p_value']:.4f}")
print(f"Significant: {gender_results['significant']}")
```

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: TOGETHER_API_KEY environment variable not set
   ```
   **Solution:** Set your Together AI API key in the environment

2. **Module Import Error**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   **Solution:** Run scripts from the project root directory

3. **Insufficient Data**
   ```
   Gender analysis: Insufficient data, p-value: nan
   ```
   **Solution:** Ensure your dataset has sufficient samples with gender labels

### Dataset Requirements

Your dataset should contain:
- `question`: Medical question text
- `options`: Multiple choice options (A, B, C, D, E)
- `answer_idx`: Correct answer index

The text should ideally contain demographic information like:
- "A 45-year-old woman presents with..."
- "Male patient, 32 years old..."
- "6-month-old infant..."

## Output Files

After running both scripts, you'll find:

```
data/processed/
├── llm_responses.csv          # Raw LLM responses
└── results/
    └── plots/
        ├── llm_bias_analysis_accuracy_by_gender.png
        ├── llm_bias_analysis_accuracy_by_age.png
        └── llm_bias_analysis_age_distribution.png
``` 