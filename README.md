# LLM Medical Question-Answering Bias Analysis

A streamlined Python project for evaluating and analyzing demographic biases (age, gender) of Large Language Models on medical question-answering tasks.

## Key Findings
[Google Slides Presentation](https://docs.google.com/presentation/d/1wzMOq8klshwKAxkrs8uf6xMs_hSCu_-191eBpTN8EtM/edit?usp=sharing)
[Written Report](https://docs.google.com/document/d/1_PujwlJMvAoFGtNECMIjmZOW1pk117O3hgnLmaHL17U/edit?tab=t.0)

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Hugging Face & CSV data loading
│   │   └── preprocessing.py       # Feature engineering & data cleaning
│   ├── llm_interface/
│   │   ├── __init__.py
│   │   └── together_api.py        # Together API client with retry logic
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Accuracy metrics & correctness indicators
│   │   └── statistical_analysis.py # Statistical tests & bias detection
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py               # Plot generation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Logging, config, utilities
├── scripts/
│   ├── run_data_collection.py     # Data collection & LLM querying
│   └── run_analysis.py            # Statistical analysis & visualization
├── config/
│   └── config.yaml                # Centralized configuration
├── data/
│   ├── raw/                       # Original datasets
│   └── processed/                 # Processed data & results
├── tests/                         # Unit tests
├── .gitignore
├── README.md
├── USAGE_EXAMPLES.md
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.8+
- Together AI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MD-Datathon
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export TOGETHER_API_KEY="your_api_key_here"  # On Windows: set TOGETHER_API_KEY=your_api_key_here
   ```

### Basic Usage

1. **Run Data Collection**
   ```bash
   python scripts/run_data_collection.py
   ```

2. **Run Analysis**
   ```bash
   python scripts/run_analysis.py
   ```

3. **View Results**
   - Plots: `data/processed/results/plots/`
   - Console output shows statistical test results

For more detailed examples, see [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md).

## Detailed Usage

### Data Collection Script

The data collection script handles the complete data pipeline:

```bash
python scripts/run_data_collection.py [--config PATH]

Options:
  --config PATH        Configuration file path (default: config/config.yaml)
```

**What it does:**
1. Loads medical datathon dataset from Hugging Face
2. Extracts age and gender features using regex patterns
3. Queries LLM (Meta-Llama) for medical question answers
4. Saves processed data with LLM responses

### Analysis Script

The analysis script performs bias analysis:

```bash
python scripts/run_analysis.py [--config PATH]

Options:
  --config PATH        Configuration file path (default: config/config.yaml)
```

**What it does:**
1. Loads processed data with LLM responses
2. Calculates accuracy metrics by demographics
3. Runs statistical tests for bias detection
4. Generates visualizations

## Configuration

The project uses a centralized YAML configuration file (`config/config.yaml`):

```yaml
# Data Configuration
data:
  datathon_dataset_name: "mdplus/Datathon2024"
  default_track: "Medical Education"
  required_columns: ["question", "options", "answer_idx"]
  text_column: "question"

# LLM Configuration
llm:
  model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
  max_retries: 3
  temperature: 0.1
  max_tokens: 10
  batch_size: 50
  valid_answer_options: ["A", "B", "C", "D", "E"]

# Feature Engineering
features:
  extract_age: true
  extract_gender: true
  age_categories:
    "Toddler": [0, 3]
    "Child": [3, 18]
    "Adult": [18, 60]
    "Senior": [60, 150]

# Analysis Configuration
analysis:
  significance_level: 0.05
  demographic_columns: ["Gender", "Age_Category"]
  correctness_column: "correct"

# Visualization Configuration
visualization:
  figure_size: [10, 7]
  dpi: 300
  format: "png"

# Output Configuration
output:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  results_dir: "data/processed/results"
  plots_dir: "data/processed/results/plots"
  llm_responses_file: "llm_responses.csv"
  plot_prefix: "llm_bias_analysis"

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file: null  # Set to null for console-only logging

# Environment Variables
environment:
  together_api_key: "${TOGETHER_API_KEY}"
```

## Analysis Features

### Statistical Tests

1. **Normality Testing**: Shapiro-Wilk test for distribution assessment
2. **Two-Group Comparisons**: 
   - t-test (parametric)
   - Mann-Whitney U test (non-parametric)
3. **Multi-Group Analysis**:
   - ANOVA (parametric)
   - Kruskal-Wallis test (non-parametric)

### Bias Detection

The analysis evaluates:

- **Gender Bias**: Accuracy differences between male and female patients
- **Age Bias**: Accuracy differences across age categories (Toddler, Child, Adult, Senior)
- **Statistical Significance**: Hypothesis testing with p-value reporting

### Visualization Outputs

- **Accuracy by Gender**: Bar plots showing LLM performance by gender
- **Accuracy by Age**: Bar plots showing performance across age categories  
- **Age Distribution**: Histogram of patient ages in the dataset

## Dependencies

The project uses these core dependencies:

- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Statistical functions
- `statsmodels>=0.12.0` - Statistical modeling
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `datasets>=2.0.0` - Hugging Face datasets
- `together>=0.1.0` - Together AI API
- `pyyaml>=6.0` - Configuration files
- `backoff>=2.0.0` - API retry logic
- `pytest>=6.2.0` - Testing

## Testing

Run the test suite:

```bash
pytest tests/
```

Tests cover core preprocessing functionality:
- Age extraction from medical text
- Gender extraction from medical text  
- Age categorization logic

## License

This project is licensed under the MIT License.

## Acknowledgments

- **MD+** for the datathon and resource access
- **Hugging Face** for the datasets library and medical question datasets
- **Together AI** for providing access to Meta-Llama models
