# ML System for Streaming Data Processing

## Project Overview
This project implements a comprehensive MLOps pipeline for automated ML model deployment in stream data processing environments. The system handles the complete ML lifecycle including data collection, quality analysis, preprocessing, model training/updating, validation, maintenance, and comprehensive monitoring.

## 🏗️ Architectural Principles

This project follows **SOLID design principles** with a modular architecture that emphasizes maintainability and extensibility:

### Successfully Implemented SOLID Principles:
- **Single Responsibility**: Each module has clear, focused responsibilities
- **Open/Closed**: Configuration-driven extensibility without code changes  
- **Liskov Substitution**: Consistent interfaces across model types and components
- **Interface Segregation**: Focused, minimal interfaces between components
- **Dependency Inversion**: Abstraction-based design with configurable implementations

### Why This Architecture Works for MLOps:
The current structure balances practical implementation with sound architectural principles, prioritizing:
- **Operational clarity** over theoretical purity
- **Configuration flexibility** for research iterations  
- **Maintainable separation** between data, training, and monitoring concerns
- **Real-world applicability** for production MLOps environments

## Project Structure
```
├── data/                    # Raw datasets
├── config.py               # ✅ Unified configuration management
├── unified_config.yaml     # ✅ Consolidated configuration file
├── data_collection/        # Stage 1: Data collection and streaming
│   └── data_collection.py  # Data streaming implementation
├── data_analyzer/          # Stage 2: Data quality and analysis  
│   ├── dq_pipeline.py      # ✅ Enhanced data quality pipeline
│   ├── make_ref_rules.py   # Association rules generator (Apriori)
│   └── drift_detector.py   # ✅ Comprehensive drift detection
├── model_pipeline/         # Stages 3-5: Data prep, training, validation
│   └── model_pipeline.py   # Complete model pipeline
├── model_maintenance/      # Stage 6: Model maintenance
│   └── model_maintenance.py # Model packaging and performance monitoring
├── model_registry/         # Model storage and versioning
├── run.py                  # ✅ Enhanced main controller
├── requirements.txt        # Dependencies
└── README.md              # This file
```

> **Note**: Consolidated configuration system replaces individual config files with unified management

## Key Features & Recent Enhancements

### ✅ **Data Drift Monitoring (NEW)**
- **Comprehensive detection**: Feature drift, concept drift, quality drift
- **Statistical tests**: KS-test, Chi-squared, distribution analysis
- **Real-time monitoring**: Integrated into streaming pipeline
- **Aggregated reporting**: Summary integration with trend analysis
- **Visual dashboard**: Drift status indicators and confidence metrics

### ✅ **Consolidated Configuration System**
- **Single source of truth**: Unified YAML configuration
- **Programmatic access**: Type-safe configuration class
- **Hot-reload capable**: Runtime configuration updates
- **Section organization**: Logical grouping of pipeline parameters

### ✅ **Enhanced Model Maintenance**
- **Performance monitoring**: Inference time, memory usage tracking
- **Adaptive model selection**: Runtime model switching based on data characteristics
- **Quality thresholds**: Configurable performance boundaries
- **Version control**: Comprehensive model registry with metadata

### ✅ **Professional Reporting System**
- **Multi-format outputs**: JSON reports, visual dashboards
- **Comprehensive metrics**: Model performance, data quality, drift analysis
- **Historical tracking**: Trend analysis across batches
- **Automated generation**: Integrated with all operational modes

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

The system supports four operational modes with enhanced functionality:

### 1. Inference Mode
Apply the best trained model to external data with comprehensive preprocessing:
```bash
python run.py -mode "inference" -file "./test_inference_data.csv"
```
**Returns**: Path to CSV file with predictions and confidence scores

### 2. Update Mode
Retrain/update models with new streaming data including drift-aware updates:
```bash
python run.py -mode "update" -batch_limit 5
```
**Returns**: Boolean success status with detailed update metrics

### 3. Summary Mode
Generate comprehensive monitoring and performance reports with drift analysis:
```bash
python run.py -mode "summary"
```
**Returns**: Path to summary report file and visual dashboard

### 4. Pipeline Mode (Full Workflow)
Execute complete pipeline from data collection to model deployment:
```bash
python run.py -mode "pipeline" -initial_batches 3 -max_batches 20 -update_every 5
```
**Returns**: Success status with performance metrics and drift analysis

## Configuration

### Unified Configuration System (`unified_config.yaml`)

The project uses a consolidated configuration approach:

```yaml
# Data Collection Configuration
data_collection:
  batch_size: 5000
  delay: 0.01
  sources:
    - ./data/motor_data11-14lats.csv
    - ./data/motor_data14-2018.csv

# Data Analysis with Drift Detection
data_analysis:
  drift_detection:
    enabled: true
    monitoring_fields:
      - "CLAIM_PAID"
      - "INSURED_VALUE"
      - "PREMIUM"
    thresholds:
      feature_drift: 0.05
      concept_drift: 0.1

# Model Training Configuration
model_training:
  data_folder: ./analyzed_data
  model_registry_path: ./model_registry
  test_size: 0.2
  models:
    DecisionTree:
      max_depth: [10, 15]
    RandomForest:
      n_estimators: [100, 200]
    NeuralNetwork:
      hidden_layer_sizes: [[100], [100, 50]]

# Model Maintenance
model_maintenance:
  performance_thresholds:
    accuracy: 0.7
    recall: 0.6
```

## Data Requirements

- **Format**: CSV, XLS, or XLSX files
- **Minimum**: 10,000+ rows, 10+ features (2+ categorical)
- **Required**: Temporal variable for streaming simulation  
- **Required**: Missing values (simulated or actual)
- **Target**: Binary classification variable (CLAIM_PAID)

## Implementation Details

### Stage 1: Data Collection ✅
- **Batch streaming** with configurable size and delay
- **Multiple data sources** support with load balancing
- **Metadata calculation** and progress tracking
- **Error handling** and comprehensive logging

### Stage 2: Data Analysis ✅  
- **Data Quality assessment** with Apriori association rules
- **Automated EDA** with statistical profiling
- **Data drift detection** with multiple statistical tests
- **Quality threshold** enforcement and reporting

### Stage 3: Data Preparation ✅
- **Missing value imputation** with adaptive strategies
- **Categorical encoding** with label preservation
- **Numerical standardization** with outlier handling
- **Pipeline-based preprocessing** for consistency

### Stage 4: Model Training/Updating ✅
- **Multiple model types**: Decision Tree, Random Forest, Neural Network
- **Hyperparameter optimization** with GridSearchCV
- **Incremental learning** with warm-start capability
- **Cross-validation** with time-series awareness

### Stage 5: Model Validation ✅
- **Comprehensive metrics**: ROC-AUC, F1, Precision, Recall
- **Model interpretation** with SHAP analysis
- **Version control** with performance tracking
- **Drift monitoring** with adaptive thresholds

### Stage 6: Model Maintenance ✅  
- **Automated packaging** with joblib serialization
- **Performance monitoring** with resource tracking
- **Adaptive selection** based on data characteristics
- **Quality assurance** with configurable thresholds

### Stage 7: Program Management ✅
- **CLI interface** with four operational modes
- **Unified configuration** management
- **Comprehensive logging** and error handling
- **Professional reporting** with visualization

## Model Types Implemented

### 1. **Decision Tree Classifier**
- Interpretable baseline model
- GridSearch optimization
- Feature importance analysis

### 2. **Random Forest Classifier**  
- Ensemble method for robust performance
- Warm-start incremental learning
- Comprehensive feature interpretation

### 3. **Neural Network Classifier**
- Deep learning capability
- Adaptive architecture search
- Non-linear pattern detection

## Data Quality & Drift Monitoring

### Association Rule Validation
- **Apriori algorithm** implementation
- Customizable quality thresholds  
- Automated consistency checks
- Rule-based data validation

### Drift Detection System
- **Feature drift**: KS-test for numerical, Chi-squared for categorical
- **Concept drift**: Target distribution analysis with PSI/KL divergence
- **Quality drift**: Missing value and outlier pattern changes
- **Trend analysis**: Historical drift pattern tracking

### Monitoring Dashboard
- **Real-time status**: Drift indicators with confidence scores
- **Historical trends**: Drift rate and pattern analysis
- **Feature impact**: Most frequently affected features
- **Severity classification**: Low/medium/high drift levels

## Performance Characteristics

### Resource Usage
- **Memory efficient**: Batch processing with streaming
- **CPU optimized**: Parallel model training
- **Storage smart**: Incremental model versioning
- **Time aware**: Efficient inference pipelines

### Scalability
- **Horizontal scaling**: Multiple data source support
- **Vertical optimization**: Resource-aware processing
- **Batch adaptation**: Dynamic batch size adjustment
- **Model efficiency**: Lightweight inference models

## Team Information & Responsibilities

**Project**: MLOps Pipeline for Streaming Data  
**Course**: Intelligent Data Analysis  
**Track**: MLOps  
**Deadline**: Task 1 - March 23, 2026  

### Team Member Responsibilities:
- **Data Analysis (Stage 2)**: Enhanced drift detection and quality monitoring
- **Model Maintenance (Stage 6)**: Performance tracking and adaptive selection  
- **Program Management (Stage 7)**: CLI interface and comprehensive reporting

## Testing & Validation

### Operational Mode Testing
All four operational modes have been validated:
1. **Inference**: ✅ Working with proper data validation
2. **Update**: ✅ Functional with model versioning
3. **Summary**: ✅ Comprehensive reporting with visualization
4. **Pipeline**: ✅ Full workflow execution with drift monitoring

### Integration Testing
- **Configuration system**: ✅ Unified config loading and validation
- **Data pipeline**: ✅ End-to-end streaming and processing
- **Model lifecycle**: ✅ Training, validation, maintenance integration
- **Monitoring system**: ✅ Real-time drift detection and reporting

## Contributing Guidelines

This project follows specific implementation requirements:
- **No external MLOps platforms** (MLFlow, AirFlow, etc.)
- **Standard Python libraries** (scikit-learn, pandas, numpy)
- **Modular architecture** with clear separation of concerns
- **Configuration-driven** implementation for flexibility

## License

This project is part of academic coursework at the Department of Intelligent Information Technologies.

## Future Enhancements

### Planned for Task 2:
- **Docker containerization** for deployment
- **CI/CD pipeline** integration
- **Advanced monitoring** with alerting
- **Database integration** for production use
- **API endpoint** for remote inference

### Research Directions:
- **Advanced drift detection** with machine learning
- **Automated hyperparameter** optimization
- **Federated learning** capabilities
- **Explainable AI** integration
- **Multi-modal data** support

---

**Last Updated**: March 23, 2026  
**Version**: 1.0 (Task 1 Completion)  
**Status**: ✅ Production Ready