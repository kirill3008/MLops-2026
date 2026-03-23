# ML System for Streaming Data Processing

## Project Overview
This project implements an MLOps pipeline for automated ML model deployment in stream data processing environments. The system handles data collection, analysis, preprocessing, model training, validation, maintenance, and management.

## Project Structure
```
├── data/                    # Raw datasets
├── data_collection/         # Stage 1: Data collection and streaming
│   ├── config.yaml         # Configuration for data collection
│   └── data_collection.py  # Data streaming implementation
├── data_analyzer/          # Stage 2: Data quality and analysis
│   ├── config.yaml         # Data quality configuration
│   ├── dq_pipeline.py      # Data quality pipeline
│   └── make_ref_rules.py   # Association rules generator
├── model_pipeline/         # Stages 3-5: Data prep, training, validation
│   ├── model_config.yaml   # Model training configuration
│   └── model_pipeline.py   # Complete model pipeline
├── model_registry/         # Model storage and versioning
├── run.py                  # Stage 6-7: Main controller
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd MLops-2026
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

The system supports three operational modes:

### 1. Inference Mode
Apply the best trained model to external data:
```bash
python run.py -mode "inference" -file "./path_to/data.csv"
```
**Returns**: Path to CSV file with predictions

### 2. Update Mode
Retrain/update models with new streaming data:
```bash
python run.py -mode "update"
```
**Returns**: Boolean success status

### 3. Summary Mode
Generate monitoring and performance reports:
```bash
python run.py -mode "summary"
```
**Returns**: Path to summary report file

## Configuration

### Data Collection Configuration (`data_collection/config.yaml`)
```yaml
batch_size: 5000
delay: 0.01
output_dir: ../raw_data
sources:
- ../data/motor_data11-14lats.csv
- ../data/motor_data14-2018.csv
```

### Model Training Configuration (`model_pipeline/model_config.yaml`)
```yaml
data_folder: ../raw_data
model_registry_path: ./model_registry
test_size: 0.2
random_state: 42
models:
  DecisionTree:
    max_depth: [10, 15]
    min_samples_split: [5, 10]
  RandomForest:
    n_estimators: [100, 200]
    max_depth: [10]
  NeuralNetwork:
    hidden_layer_sizes: [[100], [100, 50]]
    alpha: [0.001, 0.01]
```

## Data Requirements

- Format: CSV, XLS, or XLSX files
- Minimum: 10,000 rows, 10+ features (2+ categorical)
- Required: Temporal variable for streaming simulation
- Required: Missing values (simulated or actual)
- Target variable: CLAIM_PAID (binary classification)

## Implementation Details

### Stage 1: Data Collection
- ✅ Batch streaming with configurable size and delay
- ✅ File system storage for raw data
- ✅ Metadata parameter calculation
- ✅ Error handling and logging

### Stage 2: Data Analysis  
- ✅ Data quality assessment metrics
- ✅ Association rules (Apriori algorithm)
- ✅ Automated EDA and reporting
- ✅ Data drift monitoring capability

### Stage 3: Data Preparation
- ✅ Missing value imputation
- ✅ Categorical variable encoding
- ✅ Numerical variable standardization
- ✅ Pipeline-based preprocessing

### Stage 4: Model Training/Updating
- ✅ Multiple model types: Decision Tree, Random Forest, Neural Network
- ✅ Hyperparameter optimization (GridSearchCV)
- ✅ Time-series cross-validation
- ✅ Model serialization with joblib

### Stage 5: Model Validation
- ✅ Comprehensive metric evaluation
- ✅ Model interpretation (SHAP analysis)
- ✅ Model versioning system
- ✅ Performance monitoring

### Stage 6: Model Maintenance  
- ✅ Model selection and packaging
- ✅ Performance monitoring (time/memory)
- ✅ Flexible prediction strategies
- ✅ Runtime model switching

### Stage 7: Program Management
- ✅ CLI interface with three modes
- ✅ Configuration management
- ✅ Logging and error handling
- ✅ Report generation

## Features

### Model Types Implemented
1. **Decision Tree Classifier** - Interpretable baseline model
2. **Random Forest Classifier** - Ensemble method for improved performance  
3. **Neural Network Classifier** - Deep learning capability

### Data Quality Rules
- Association rule validation using Apriori algorithm
- Customizable quality thresholds
- Automated consistency checks

### Model Monitoring
- Performance metrics tracking over time
- Data and model drift detection
- Hyperparameter evolution analysis
- Resource utilization monitoring

## Contributing

This project follows specific implementation guidelines:
- No external MLOps platforms (MLFlow, AirFlow, etc.)
- Based on standard Python libraries (scikit-learn, pandas, numpy)
- Modular architecture with clear separation of concerns

## Team Information

**Project**: MLOps Pipeline for Streaming Data  
**Course**: Intelligent Data Analysis  
**Track**: MLOps  
**Deadline**: Task 1 - March 23, 2026  

## License

This project is part of academic coursework at the Department of Intelligent Information Technologies.