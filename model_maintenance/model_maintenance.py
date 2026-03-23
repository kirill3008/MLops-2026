"""
Stage 6: Model Maintenance Module
Handles model packaging, performance monitoring, and runtime adaptation
"""

import sys
import time
import psutil
import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []
        self.performance_thresholds = config.get('performance_thresholds', {})
    
    def measure_inference_time(self, model, X_test):
        """Measure inference time for a model"""
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time
        
        return inference_time, len(X_test)
    
    def measure_memory_usage(self):
        """Measure current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]) -> bool:
        """Check if model meets performance thresholds"""
        thresholds = self.performance_thresholds
        
        violations = []
        
        if 'accuracy' in thresholds and 'accuracy' in metrics:
            if metrics['accuracy'] < thresholds['accuracy']:
                violations.append(f"Accuracy {metrics['accuracy']} < {thresholds['accuracy']}")
        
        if 'f1' in thresholds and 'f1' in metrics:
            if metrics['f1'] < thresholds['f1']:
                violations.append(f"F1-score {metrics['f1']} < {thresholds['f1']}")
        
        if 'inference_time' in thresholds and 'inference_time' in metrics:
            if metrics['inference_time'] > thresholds['inference_time']:
                violations.append(f"Inference time {metrics['inference_time']} > {thresholds['inference_time']}")
        
        if violations:
            logger.warning(f"Performance threshold violations: {violations}")
            return False
        
        return True
    
    def record_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Record performance metrics with timestamp"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics
        }
        self.metrics_history.append(record)
    
    def get_performance_trend(self, model_name: str, metric: str) -> List[float]:
        """Get trend for specific metric"""
        values = []
        for record in self.metrics_history:
            if record['model_name'] == model_name and metric in record['metrics']:
                values.append(record['metrics'][metric])
        return values


class ModelPackager:
    """Handle model packaging and serialization"""
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
    
    def package_model(self, model, model_name: str, metrics: Dict[str, Any], 
                     feature_names: Optional[List[str]] = None,
                     preprocessing_pipeline = None):
        """Package model with metadata and preprocessing"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{timestamp}.joblib"
        package_filename = f"{model_name}_{timestamp}.pkg"
        
        # Create package directory
        package_dir = os.path.join(self.registry_path, f"{model_name}_{timestamp}")
        os.makedirs(package_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(package_dir, model_filename)
        joblib.dump(model, model_path)
        
        # Save preprocessing pipeline if provided
        if preprocessing_pipeline:
            pipeline_path = os.path.join(package_dir, f"preprocessor_{timestamp}.joblib")
            joblib.dump(preprocessing_pipeline, pipeline_path)
        
        # Create package metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_file': model_filename,
            'metrics': metrics,
            'feature_names': feature_names,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'requirements': self._get_requirements_info()
        }
        
        metadata_path = os.path.join(package_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model packaged: {package_dir}")
        return package_dir
    
    def _get_requirements_info(self):
        """Get installed package versions"""
        import importlib.metadata
        packages = ['pandas', 'numpy', 'scikit-learn', 'tensorflow']
        versions = {}
        
        for pkg in packages:
            try:
                version = importlib.metadata.version(pkg)
                versions[pkg] = version
            except:
                versions[pkg] = 'not_installed'
        
        return versions
    
    def load_package(self, package_dir: str):
        """Load packaged model"""
        metadata_path = os.path.join(package_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Package metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_path = os.path.join(package_dir, metadata['model_file'])
        model = joblib.load(model_path)
        
        # Load preprocessing pipeline if available
        preprocessor_path = os.path.join(package_dir, f"preprocessor_{metadata['timestamp']}.joblib")
        preprocessor = None
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        
        return model, metadata, preprocessor


class ModelSelector:
    """Select best model based on performance and data characteristics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selection_rules = config.get('selection_rules', {})
    
    def select_best_model(self, model_performance: Dict[str, Any], 
                         data_characteristics: Dict[str, Any]) -> str:
        """Select best model based on multiple criteria"""
        
        # Rule 1: Performance-based selection
        performance_scores = {}
        for model_name, metrics in model_performance.items():
            score = self._calculate_performance_score(metrics)
            performance_scores[model_name] = score
        
        # Rule 2: Data characteristics adaptation
        adaptation_scores = {}
        for model_name in model_performance.keys():
            score = self._calculate_adaptation_score(model_name, data_characteristics)
            adaptation_scores[model_name] = score
        
        # Combine scores
        final_scores = {}
        for model_name in model_performance.keys():
            perf_weight = self.selection_rules.get('performance_weight', 0.7)
            adapt_weight = self.selection_rules.get('adaptation_weight', 0.3)
            
            final_scores[model_name] = (
                perf_weight * performance_scores[model_name] +
                adapt_weight * adaptation_scores[model_name]
            )
        
        best_model = max(final_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected model: {best_model} (score: {final_scores[best_model]:.3f})")
        
        return best_model
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score from metrics"""
        weights = {
            'accuracy': 0.3,
            'f1': 0.4,
            'roc_auc': 0.3
        }
        
        score = 0
        weight_sum = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                weight_sum += weight
        
        return score / weight_sum if weight_sum > 0 else 0
    
    def _calculate_adaptation_score(self, model_name: str, 
                                  data_characteristics: Dict[str, Any]) -> float:
        """Calculate adaptation score based on data characteristics"""
        
        # Simple rules for model adaptation
        rules = {
            'DecisionTree': {
                'sparse_data': 0.8,    # Good with sparse data
                'anomalous_values': 0.6,  # Moderate robustness
                'numerical_features': 0.7,
                'categorical_features': 0.9,
            },
            'RandomForest': {
                'sparse_data': 0.7,
                'anomalous_values': 0.8,  # Good robustness
                'numerical_features': 0.8,
                'categorical_features': 0.8,
            },
            'NeuralNetwork': {
                'sparse_data': 0.5,    # Poor with sparse data
                'anomalous_values': 0.4,  # Sensitive to anomalies
                'numerical_features': 0.9,
                'categorical_features': 0.6,
            }
        }
        
        if model_name not in rules:
            return 0.5  # Default score
        
        model_rules = rules[model_name]
        score = 0
        weight_sum = 0
        
        for char, value in data_characteristics.items():
            if char in model_rules:
                # Higher score when model is suitable for data characteristic
                score += model_rules[char] * (1.0 - abs(value - 0.5))  # Closer to 0.5 = balanced
                weight_sum += model_rules[char]
        
        return score / weight_sum if weight_sum > 0 else 0.5


class ModelMaintenance:
    """Main model maintenance orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.model_packager = ModelPackager(config.get('model_registry_path', 'model_registry'))
        self.model_selector = ModelSelector(config)
        self.current_best_model = None
        self.best_model_metadata = None
    
    def evaluate_model_performance(self, model, model_name: str, X_test, y_test):
        """Comprehensive model performance evaluation"""
        
        # Measure inference time
        inference_time, n_samples = self.performance_monitor.measure_inference_time(model, X_test)
        
        # Measure memory usage
        memory_info = self.performance_monitor.measure_memory_usage()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1': f1_score(y_test, predictions, average='weighted'),
            'inference_time': inference_time,
            'samples_per_second': n_samples / inference_time if inference_time > 0 else 0,
            'memory_rss_mb': memory_info['rss_mb'],
            'memory_vms_mb': memory_info['vms_mb']
        }
        
        if probabilities is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, probabilities)
        
        # Check performance thresholds
        meets_thresholds = self.performance_monitor.check_performance_thresholds(metrics)
        metrics['meets_thresholds'] = meets_thresholds
        
        # Record metrics
        self.performance_monitor.record_metrics(model_name, metrics)
        
        return metrics
    
    def package_and_register_model(self, model, model_name: str, metrics: Dict[str, Any],
                                 feature_names: List[str], preprocessing_pipeline = None):
        """Package model and register in registry"""
        
        package_dir = self.model_packager.package_model(
            model, model_name, metrics, feature_names, preprocessing_pipeline
        )
        
        # Update best model if this model performs better
        if self._should_update_best_model(model_name, metrics):
            self.current_best_model = model
            self.best_model_metadata = {
                'model_name': model_name,
                'package_dir': package_dir,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save best model info
            best_model_info_path = os.path.join(
                self.config.get('model_registry_path', 'model_registry'), 
                'best_model.json'
            )
            best_info = {
                'model_name': model_name,
                'model_file': f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                'package_dir': package_dir,
                'metrics': metrics,
                'selected_at': datetime.now().isoformat()
            }
            
            with open(best_model_info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
        
        return package_dir
    
    def _should_update_best_model(self, model_name: str, metrics: Dict[str, Any]) -> bool:
        """Determine if this model should become the new best model"""
        
        if self.best_model_metadata is None:
            return True
        
        current_best_metrics = self.best_model_metadata['metrics']
        
        # Compare primary metric (F1-score)
        if 'f1' in metrics and 'f1' in current_best_metrics:
            improvement_threshold = self.config.get('improvement_threshold', 0.01)
            if metrics['f1'] > current_best_metrics['f1'] + improvement_threshold:
                return True
        
        # If current best model doesn't meet thresholds, prioritize one that does
        if not metrics.get('meets_thresholds', True) and current_best_metrics.get('meets_thresholds', True):
            return False
        
        return False
    
    def select_model_for_prediction(self, data_characteristics: Dict[str, Any]) -> str:
        """Select appropriate model for given data characteristics"""
        
        # Get performance history
        performance_data = {}
        for record in self.performance_monitor.metrics_history[-10:]:  # Last 10 records
            model_name = record['model_name']
            if model_name not in performance_data:
                performance_data[model_name] = []
            performance_data[model_name].append(record['metrics'])
        
        # Calculate average performance
        avg_performance = {}
        for model_name, metrics_list in performance_data.items():
            avg_metrics = {}
            for metric_name in ['accuracy', 'f1', 'roc_auc']:
                values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
                if values:
                    avg_metrics[metric_name] = sum(values) / len(values)
            avg_performance[model_name] = avg_metrics
        
        return self.model_selector.select_best_model(avg_performance, data_characteristics)
