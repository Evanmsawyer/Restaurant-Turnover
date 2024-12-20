# Restaurant Employee Turnover Analysis

## Overview
This repository contains a comprehensive study for predicting and understanding employee turnover in the restaurant industry using survey data. The project implements multiple machine learning approaches including traditional ML models and deep learning with transformers.

## Project Structure
The project consists of three main scripts:
1. `traditional_models.py` - Traditional machine learning approach with feature engineering
2. `transformer_model.py` - Deep learning approach using a custom transformer architecture
3. `gpu_boosting_models.py` - GPU-accelerated models for high-performance prediction

## Features

### Data Processing
- Advanced cleaning and preprocessing of restaurant industry-specific data
- Comprehensive feature engineering including:
  - Tenure normalization
  - Certification categorization
  - Position mapping
  - Tip amount standardization

### Models
Three different modeling approaches are implemented:

#### Traditional Models (`traditional_models.py`)
- Linear Regression
- Ridge and Lasso Regression
- Random Forest
- XGBoost, LightGBM, CatBoost
- Stacking and Ensemble methods

#### Transformer Model (`transformer_model.py`)
- Custom transformer architecture for sequential data
- Positional encoding
- Multi-head attention
- Advanced regularization techniques
- Calibrated output predictions

#### GPU-Optimized Models (`gpu_optimized_models.py`)
- GPU-accelerated implementations of:
  - XGBoost
  - LightGBM
  - CatBoost
- Optimized for high-performance computing

### Analysis Features
- Comprehensive metrics calculation
- ROC curve analysis
- Feature importance visualization
- Error distribution analysis
- Cross-validation
- Model performance comparison

## Requirements

### Hardware
- CUDA-compatible GPU recommended for `transformer_model.py` and `gpu_optimized_models.py`
- Minimum 32GB RAM recommended

### Software
```
python>=3.8
torch
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
tpot
matplotlib
seaborn
optuna
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/restaurant-turnover-analysis.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Usage

### Data Preparation
Place your survey data file (Excel format) in the project directory. The expected format should include:
- Employee demographic information
- Work experience details
- Survey responses
- Turnover intention indicators (INTN_1 through INTN_6)

### Running the Analysis

1. Traditional Analysis:
```python
python traditional_models.py
```

2. Transformer Model:
```python
python transformer_model.py
```

3. GPU-Optimized Models:
```python
python gpu_optimized_models.py
```

### Output
Each script generates outputs in the respective directories:
- `output_plots/` - Visualizations from traditional models
- `output_plots_2/` - Transformer model analysis
- `output_plots_3/` - GPU-optimized model results

## Model Performance
The system evaluates models using multiple metrics:
- R² Score
- Mean Squared Error (MSE)
- F1 Score
- Accuracy
- Precision
- Recall
- ROC-AUC

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
[Your Name]

## Acknowledgments
- Restaurant industry survey participants
- Open-source ML community
- GPU computing resources providers