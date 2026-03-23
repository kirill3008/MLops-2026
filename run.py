#!/usr/bin/env python3
"""
MLOps Pipeline Control Script
Task 1: ML System for Streaming Data Processing

Usage:
    python run.py -mode "inference" -file "./path_to.file"
    python run.py -mode "update"
    python run.py -mode "summary"
"""

import sys
import os
import argparse
import pandas as pd
import time
import logging
import json
from datetime import datetime
from model_pipeline.model_pipeline import DataPreprocessor

# Add project directories to path
sys.path.extend(['data_collection', 'data_analyzer', 'model_pipeline'])

from data_collection.data_collection import DataStream
from data_analyzer.dq_pipeline import evaluate_reference_rules_on_batch
from model_pipeline.model_pipeline import ModelPipeline
from config import get_config


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler("pipeline.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class MLOpsPipeline:
    """Main MLOps pipeline controller"""
    
    def __init__(self):
        self.config = self._load_main_config()
        self.model_pipeline = None
        self.best_model = None
        
    def _load_main_config(self):
        """Load main configuration using unified Config class"""
        return get_config()
    
    def inference_mode(self, file_path: str):
        """Apply the best model to external data"""
        logger.info("=== INFERENCE MODE ===")
        logger.info(f"Input file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Load and validate the best model
        self._load_best_model()
        if self.best_model is None:
            raise ValueError("No trained model found for inference")
        
        # Load and preprocess input data
        try:
            input_data = pd.read_csv(file_path)
            logger.info(f"Loaded data: {len(input_data)} rows")
            
            # Create preprocessor and apply it (without fitting scaler)
            preprocessor = DataPreprocessor()
            preprocessed = preprocessor.preprocess(input_data, fit_scaler=False)
            
            # The preprocessor returns (X_scaled, y, feature_cols, label_encoders)
            if isinstance(preprocessed, tuple) and len(preprocessed) >= 2:
                X_processed = preprocessed[0]  # Scaled features
                logger.info(f"Preprocessed data shape: {X_processed.shape}")
            else:
                raise ValueError("Preprocessor returned unexpected format")
            
        except Exception as e:
            logger.error(f"Error loading or preprocessing input data: {e}")
            return None
        
        # Perform inference on preprocessed data
        start_time = time.time()
        predictions = self.best_model.predict(X_processed)
        inference_time = time.time() - start_time
        
        # Save results
        result_df = input_data.copy()
        result_df['predict'] = predictions
        
        output_path = f"inference_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(output_path, index=False)
        
        logger.info(f"Inference completed in {inference_time:.2f}s")
        logger.info(f"Results saved to: {output_path}")
        
        return output_path
    
    def update_mode(self, batch_limit: int = 5):
        """Update/retrain model with new data
        
        Args:
            batch_limit (int): Maximum number of batches to process (prevents long runs)
        """
        logger.info("=== UPDATE MODE ===")
        
        try:
            # Check if we have trained models first
            if self.best_model is None:
                self._load_best_model()
                if self.best_model is None:
                    logger.error("No trained models found. Run pipeline mode first to train initial models.")
                    return False
            
            # Use data collection config from consolidated config instead of loading from file
            data_stream = DataStream(
                self.config.data_collection['sources'], 
                self.config.data_collection['batch_size'], 
                self.config.data_collection['delay']
            )
            
            batch_count = 0
            processed_count = 0
            
            for batch in data_stream.stream():
                batch_count += 1
                
                # Limit number of batches to prevent long runs
                if batch_count > batch_limit:
                    logger.info(f"Batch limit reached ({batch_limit}). Stopping update process.")
                    break
                
                logger.info(f"Processing batch {batch_count}: {len(batch)} rows")
                
                # Get batch info from attrs if available
                batch_info = batch.attrs.get('batch_info', {'batch_num': batch_count})
                
                # Data quality analysis with batch information
                dq_results = evaluate_reference_rules_on_batch(
                    batch, 
                    self.config.data_analysis,
                    batch_info={'batch_num': batch_count, 'timestamp': datetime.now().isoformat()}
                )
                
                if dq_results.get('enabled', False) and not dq_results.get('error'):
                    logger.info("  Data quality: PASSED")
                else:
                    logger.warning(f"  Data quality: ISSUES - {dq_results.get('error', 'Unknown')}")
                
                # Update model pipeline
                if self.model_pipeline is None:
                    self.model_pipeline = ModelPipeline()
                
                # Use incremental_update method which requires proper data preprocessing
                try:
                    preprocessed = self.model_pipeline.preprocessor.preprocess(batch.copy(), fit_scaler=False)
                    if isinstance(preprocessed, tuple) and len(preprocessed) >= 2:
                        X_batch, y_batch = preprocessed[0], preprocessed[1]
                        
                        # Check if we have ground truth labels for this batch
                        if y_batch is not None and len(y_batch) > 0:
                            updated_model, metrics = self.model_pipeline.incremental_update(X_new=X_batch, y_new=y_batch, model_name='RandomForest')
                            if updated_model is not None:
                                logger.info(f"  Model updated successfully for batch {batch_info['batch_num']}")
                                processed_count += 1
                            else:
                                logger.warning(f"  Model update failed for batch {batch_info['batch_num']}")
                        else:
                            logger.info(f"  Skipping update: No ground truth labels available in batch {batch_info['batch_num']}")
                    else:
                        logger.warning("  Could not preprocess batch for update")
                except Exception as e:
                    logger.error(f"  Error preprocessing batch for update: {e}")
                    continue
                
                # Update best model reference
                self._update_best_model()
            
            logger.info(f"Update completed. Processed {processed_count} batches out of {batch_count}")
            return processed_count > 0
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def summary_mode(self):
        """Generate summary report of system performance"""
        logger.info("=== SUMMARY MODE ===")
        
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "model_metrics": self._collect_model_metrics(),
            "data_quality": self._collect_data_quality(),
            "data_drift": self._collect_drift_metrics(),  # Add drift monitoring
            "performance": self._collect_performance_metrics(),
            "hyperparameters": self._collect_hyperparameters(),
            "model_selection_history": self._collect_model_selection_history(),
            "system_info": self._collect_system_info()
        }
        
        # Convert numpy types to JSON-serializable types
        def convert_to_json_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        summary_data = convert_to_json_serializable(summary_data)
        
        # Save summary report
        report_path = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Generate comprehensive visual dashboard with plots
        self._generate_comprehensive_dashboard(summary_data, report_path)
        
        logger.info(f"Summary report generated: {report_path}")
        return report_path

    def _generate_comprehensive_dashboard(self, summary_data, report_path):
        """Generate detailed visual dashboard with multiple plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib import gridspec
            
            # Create a multi-panel dashboard
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, figure=fig)
            
            # Plot 1: Performance metrics over time
            ax1 = fig.add_subplot(gs[0, 0])
            if 'performance_file' in summary_data.get('performance', {}):
                try:
                    perf_df = pd.read_csv(summary_data['performance']['performance_file'])
                    if not perf_df.empty and 'batch' in perf_df.columns and 'accuracy' in perf_df.columns:
                        sns.lineplot(data=perf_df, x='batch', y='accuracy', ax=ax1, label='Accuracy')
                        if 'f1' in perf_df.columns:
                            sns.lineplot(data=perf_df, x='batch', y='f1', ax=ax1, label='F1 Score')
                        ax1.set_title('Model Performance Over Batches')
                        ax1.set_xlabel('Batch Number')
                        ax1.set_ylabel('Metric Value')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                except Exception:
                    ax1.text(0.5, 0.5, 'No performance data available', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Model Performance (No Data)')
            
            # Plot 2: Data Quality Trends
            ax2 = fig.add_subplot(gs[0, 1])
            # Mock data quality trend for demonstration
            dq_metrics = ['missing_total_ratio', 'duplicate_ratio', 'invalid_ratio']
            available_dq = [m for m in dq_metrics if m in summary_data.get('data_quality', {})]
            
            if available_dq:
                values = [summary_data['data_quality'][m] for m in available_dq]
                ax2.bar(available_dq, values, color=['red', 'orange', 'yellow'])
                ax2.set_title('Current Data Quality Metrics')
                ax2.set_ylabel('Ratio')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No data quality metrics available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Data Quality Metrics (No Data)')
            
            # Plot 3: System Resource Usage
            ax3 = fig.add_subplot(gs[1, 0])
            if 'memory_usage_percent' in summary_data.get('performance', {}):
                memory_labels = ['Used', 'Available']
                memory_values = [
                    summary_data['performance']['memory_usage_percent'],
                    100 - summary_data['performance']['memory_usage_percent']
                ]
                ax3.pie(memory_values, labels=memory_labels, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Memory Usage')
            else:
                ax3.text(0.5, 0.5, 'No memory usage data available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Memory Usage (No Data)')
            
            # Plot 4: Data Drift Monitoring
            ax4 = fig.add_subplot(gs[1, 1])
            if summary_data.get('data_drift') and summary_data['data_drift'].get('status') != 'no_drift_data_available':
                drift_data = summary_data['data_drift']
                
                # Create drift status indicators
                drift_status = [
                    ('Feature Drift', drift_data.get('affected_features_count', 0) > 0),
                    ('Concept Drift', drift_data.get('concept_shift_confidence', 0) > 0.1),
                    ('Quality Drift', drift_data.get('quality_issues', False))
                ]
                
                labels = [status[0] for status in drift_status]
                status_values = [1 if status[1] else 0 for status in drift_status]
                colors = ['red' if status[1] else 'green' for status in drift_status]
                
                bars = ax4.bar(labels, status_values, color=colors, alpha=0.7)
                ax4.set_title('Data Drift Status')
                ax4.set_ylabel('Drift Detected (0/1)')
                ax4.set_ylim(0, 1.2)
                ax4.tick_params(axis='x', rotation=45)
                
                # Add confidence value if available
                if drift_data.get('confidence', 0) > 0:
                    ax4.text(0.5, 1.1, f'Confidence: {drift_data["confidence"]:.3f}', 
                            ha='center', va='center', fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                
                # Add drift rate if available
                if drift_data.get('drift_rate', 0) > 0:
                    ax4.text(1.5, 1.1, f'Drift Rate: {drift_data["drift_rate"]:.2f}', 
                            ha='center', va='center', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            else:
                ax4.text(0.5, 0.5, 'No drift data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Data Drift Status (No Data)')
            
            # Plot 5: Model Selection History (move to new position)
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.text(0.5, 0.5, 'Model Selection Trend Graph\n(Enhanced Reporting)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Model Performance Evolution')
            ax5.axis('off')
            
            # Plot 6: Recent Batch Statistics
            ax6 = fig.add_subplot(gs[2, 1:])
            stats = [
                ('Total Batches', summary_data['performance'].get('total_batches', 0)),
                ('Avg Accuracy', summary_data['performance'].get('avg_accuracy', 0)),
                ('Avg F1 Score', summary_data['performance'].get('avg_f1', 0)),
                ('Data Quality Passed', summary_data['data_quality'].get('data_quality_passed', False)),
                ('Consistency Issues', summary_data['data_quality'].get('consistency_issues', False)),
                ('Drift Detected', summary_data['data_drift'].get('drift_detected', False) if summary_data.get('data_drift') else False)
            ]
            
            labels = [s[0] for s in stats]
            values = [s[1] for s in stats]
            
            ax6.barh(labels, values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
            ax6.set_title('System Performance Summary')
            ax6.set_xlabel('Value')
            for i, v in enumerate(values):
                ax6.text(v + 0.01, i, f'{v}', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_path = report_path.replace('.json', '_dashboard.png')
            plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visual dashboard saved: {dashboard_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visual dashboard")
        except Exception as e:
            logger.warning(f"Could not generate visual dashboard: {e}")

    def _collect_model_selection_history(self):
        """Collect model selection history for meta-learning analysis"""
        selection_history = []
        
        try:
            model_dir = self.config.model_registry.get('path', 'model_registry')
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.json') and f != 'best_model.json']
                
                for model_file in sorted(model_files)[-5:]:  # Last 5 models
                    with open(os.path.join(model_dir, model_file), 'r') as f:
                        model_info = json.load(f)
                    
                    selection_history.append({
                        'model_name': model_info.get('model_name', 'unknown'),
                        'timestamp': model_info.get('timestamp', ''),
                        'accuracy': model_info.get('metrics', {}).get('accuracy', 0),
                        'f1_score': model_info.get('metrics', {}).get('f1', 0)
                    })
        except Exception as e:
            logger.warning(f"Could not collect model selection history: {e}")
        
        return selection_history

    def _collect_system_info(self):
        """Collect system information and configuration"""
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd(),
            "config_file": self.config_file if hasattr(self, 'config_file') else 'unified_config.yaml',
            "available_memory_gb": None,
            "cpu_count": None
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info.update({
                "available_memory_gb": memory.available / (1024**3),
                "cpu_count": psutil.cpu_count()
            })
        except ImportError:
            pass
        
        return system_info
    
    def _load_best_model(self):
        """Load the best performing model from registry"""
        registry_path = self.config.model_registry['path']
        if not os.path.exists(registry_path):
            logger.warning("Model registry not found")
            return
        
        try:
            best_model_info_path = os.path.join(registry_path, "best_model.json")
            if os.path.exists(best_model_info_path):
                with open(best_model_info_path, 'r') as f:
                    best_info = json.load(f)
                
                model_path = os.path.join(best_info.get('package_dir', registry_path), best_info['model_file'])
                if os.path.exists(model_path):
                    import joblib
                    self.best_model = joblib.load(model_path)
                    logger.info(f"Loaded best model: {best_info['model_name']}")
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
    
    def _update_best_model(self):
        """Update the best model based on performance"""
        try:
            if self.model_pipeline and hasattr(self.model_pipeline, 'best_model'):
                self.best_model = self.model_pipeline.best_model
                logger.debug("Best model updated from pipeline")
            else:
                # Try to load the best model from registry
                self._load_best_model()
                if self.best_model:
                    logger.debug("Best model loaded from registry")
        except Exception as e:
            logger.warning(f"Could not update best model: {e}")
    
    def _collect_model_metrics(self):
        """Collect model performance metrics from registry and pipeline"""
        metrics = {}
        
        # Try to get metrics from model pipeline first
        if self.model_pipeline and hasattr(self.model_pipeline, 'model_performance'):
            metrics.update(self.model_pipeline.model_performance)
        
        # Try to load metrics from best_model.json
        try:
            registry_path = self.config['model_registry']['path']
            best_model_info_path = os.path.join(registry_path, "best_model.json")
            if os.path.exists(best_model_info_path):
                with open(best_model_info_path, 'r') as f:
                    best_info = json.load(f)
                if 'metrics' in best_info:
                    metrics.update(best_info['metrics'])
                    metrics['best_model_name'] = best_info.get('model_name', 'Unknown')
        except Exception as e:
            logger.warning(f"Could not load best model info: {e}")
        
        # Get model registry statistics
        try:
            metadata_path = os.path.join(self.config.model_registry['path'], 'models_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    registry_data = json.load(f)
                
                model_counts = {}
                total_versions = 0
                for model_name, versions in registry_data.get('models', {}).items():
                    model_counts[model_name] = len(versions)
                    total_versions += len(versions)
                
                metrics.update({
                    'total_models': len(model_counts),
                    'total_versions': total_versions,
                    'model_counts': model_counts
                })
        except Exception as e:
            logger.warning(f"Could not load model registry statistics: {e}")
        
        return metrics if metrics else {"status": "no_model_metrics_available"}
    
    def _collect_data_quality(self):
        """Collect data quality metrics from artifacts"""
        dq_metrics = {}
        
        try:
            # Try to read the latest data quality reports
            artifacts_dir = self.config.data_analysis["dq"]["io"].get("artifacts_dir", "artifacts")
            dq_dir = os.path.join(artifacts_dir, "dq")
            rules_dir = os.path.join(artifacts_dir, "rules")
            
            if os.path.exists(dq_dir):
                dq_files = sorted(
                    [f for f in os.listdir(dq_dir) if f.endswith('.json')],
                    reverse=True
                )
                
                if dq_files:
                    # Load the most recent data quality report
                    latest_dq_file = os.path.join(artifacts_dir, dq_files[0])
                    with open(latest_dq_file, 'r') as f:
                        dq_data = json.load(f)
                    
                    # Extract meaningful metrics
                    if 'after' in dq_data:
                        dq_metrics.update({
                            'missing_total_ratio': dq_data['after'].get('missing_total_ratio', 0),
                            'duplicate_ratio': dq_data['after'].get('duplicate_ratio', 0),
                            'invalid_ratio': dq_data['after'].get('invalid_ratio', 0),
                            'total_rows': dq_data['after'].get('n_rows', 0),
                            'total_columns': dq_data['after'].get('n_cols', 0),
                            'batch_id': dq_files[0].replace('_dq.json', '')
                        })
                    
                    if 'flags_after' in dq_data:
                        dq_metrics['data_quality_passed'] = not dq_data['flags_after'].get('any_issue', True)
            
            # Collect consistency rule metrics
            if os.path.exists(rules_dir):
                consistency_files = sorted(
                    [f for f in os.listdir(rules_dir) if f.startswith('consistency_') and f.endswith('.json')],
                    reverse=True
                )
                
                if consistency_files:
                    latest_consistency_file = os.path.join(rules_dir, consistency_files[0])
                    with open(latest_consistency_file, 'r') as f:
                        consistency_data = json.load(f)
                    
                    dq_metrics.update({
                        'consistency_rules_checked': consistency_data.get('n_rules_checked', 0),
                        'consistency_issues': consistency_data.get('any_issue', False),
                        'consistency_batch_id': consistency_files[0].replace('consistency_', '').replace('.json', '')
                    })
        
        except Exception as e:
            logger.warning(f"Could not collect data quality metrics: {e}")
        
        return dq_metrics if dq_metrics else {"status": "no_dq_data_available"}
    
    def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        perf_metrics = {}
        
        try:
            # Try to get performance data from pipeline log files
            if os.path.exists('pipeline_performance.csv'):
                perf_df = pd.read_csv('pipeline_performance.csv')
                if not perf_df.empty:
                    perf_metrics.update({
                        'avg_accuracy': float(perf_df['accuracy'].mean()),
                        'avg_f1': float(perf_df['f1'].mean()),
                        'avg_inference_time': float(perf_df['inference_time'].mean()),
                        'total_batches_evaluated': int(len(perf_df)),
                        'performance_trend_available': True
                    })
            
            # Check for recent performance CSV files
            perf_files = sorted(
                [f for f in os.listdir('.') if f.startswith('pipeline_performance_') and f.endswith('.csv')],
                reverse=True
            )
            
            if perf_files:
                latest_perf_file = perf_files[0]
                perf_df = pd.read_csv(latest_perf_file)
                if not perf_df.empty:
                    perf_metrics.update({
                        'latest_accuracy': float(perf_df['accuracy'].iloc[-1]),
                        'latest_f1': float(perf_df['f1'].iloc[-1]),
                        'latest_inference_time': float(perf_df['inference_time'].iloc[-1]),
                        'total_batches': int(perf_df['batch'].iloc[-1]),  # Use iloc[-1] instead of max() for correct value
                        'performance_file': latest_perf_file
                    })
            
            # Check system resource usage
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                perf_metrics.update({
                    'memory_usage_percent': float(memory_info.percent),
                    'available_memory_gb': float(memory_info.available / (1024**3))
                })
            except ImportError:
                perf_metrics['psutil_available'] = False
        
        except Exception as e:
            logger.warning(f"Could not collect performance metrics: {e}")
        
        return perf_metrics if perf_metrics else {"status": "no_performance_data_available"}
    
    def _collect_drift_metrics(self):
        """Collect data drift monitoring metrics from artifacts"""
        drift_metrics = {}
        
        try:
            # Look for drift reports in artifacts directory
            artifacts_dir = self.config.data_analysis["dq"]["io"].get("artifacts_dir", "artifacts")
            rules_dir = os.path.join(artifacts_dir, "rules")
            
            if os.path.exists(artifacts_dir):
                drift_files = sorted(
                    [f for f in os.listdir(artifacts_dir) if f.startswith('drift_report_') and f.endswith('.json')],
                    reverse=True
                )
                
                # Also look for consistency reports which contain drift analysis
                consistency_files = sorted(
                    [f for f in os.listdir(rules_dir) if f.startswith('consistency_') and f.endswith('.json')],
                    reverse=True
                )
                
                # Use drift reports if available, otherwise fall back to consistency reports
                if drift_files:
                    # Load the most recent drift report
                    latest_drift_file = os.path.join(artifacts_dir, drift_files[0])
                    with open(latest_drift_file, 'r') as f:
                        drift_data = json.load(f)
                elif consistency_files:
                    # Load the most recent consistency report and extract drift analysis
                    latest_consistency_file = os.path.join(rules_dir, consistency_files[0])
                    with open(latest_consistency_file, 'r') as f:
                        consistency_data = json.load(f)
                    drift_data = consistency_data.get('drift_analysis', {})
                    drift_data['batch_info'] = {'batch_num': consistency_files[0].replace('consistency_batch_', '').replace('.json', '')}
                else:
                    drift_data = {}
                
                if drift_data:
                    # Extract key drift information
                    drift_metrics.update({
                        'drift_detected': drift_data.get('drift_detected', False),
                        'drift_type': drift_data.get('drift_type', 'none'),
                        'confidence': drift_data.get('confidence', 0.0),
                        'batch_number': drift_data.get('batch_info', {}).get('batch_num', 0),
                        'affected_features_count': len(drift_data.get('affected_features', [])),
                        'timestamp': drift_data.get('timestamp', '')
                    })
                    
                    # Add affected features details
                    if drift_data.get('affected_features'):
                        drift_metrics['affected_features'] = [
                            {
                                'feature': feature.get('feature', ''),
                                'drift_type': feature.get('drift_type', ''),
                                'confidence': feature.get('confidence', 0.0)
                            }
                            for feature in drift_data['affected_features']
                        ]
                    
                    # Add concept drift details if available
                    if drift_data.get('concept_shift_confidence') is not None:
                        drift_metrics['concept_shift_confidence'] = drift_data['concept_shift_confidence']
                        drift_metrics['target_distribution_change'] = drift_data.get('target_distribution_change', 0.0)
                    
                    # Add quality drift details
                    if drift_data.get('quality_issues') is not None:
                        drift_metrics.update({
                            'quality_issues': drift_data['quality_issues'],
                            'quality_anomalies': drift_data.get('anomalies', [])
                        })
                    
                    # Calculate drift statistics from all available reports
                    all_drift_reports = []
                    total_batches_with_drift = 0
                    
                    # Collect from drift reports
                    for drift_file in drift_files[:10]:  # Last 10 reports
                        file_path = os.path.join(artifacts_dir, drift_file)
                        with open(file_path, 'r') as f:
                            report = json.load(f)
                            all_drift_reports.append(report)
                            if report.get('drift_detected', False):
                                total_batches_with_drift += 1
                    
                    # Collect from consistency reports (if drift reports are insufficient)
                    if len(all_drift_reports) < 5:
                        for consistency_file in consistency_files[:10]:
                            file_path = os.path.join(rules_dir, consistency_file)
                            with open(file_path, 'r') as f:
                                consistency_data = json.load(f)
                                drift_report = consistency_data.get('drift_analysis', {})
                                if drift_report:
                                    drift_report['batch_info'] = {'batch_num': consistency_file.replace('consistency_batch_', '').replace('.json', '')}
                                    all_drift_reports.append(drift_report)
                                    if drift_report.get('drift_detected', False):
                                        total_batches_with_drift += 1
                    
                    drift_metrics.update({
                        'total_drift_reports_analyzed': len(all_drift_reports),
                        'batches_with_drift': total_batches_with_drift,
                        'drift_rate': total_batches_with_drift / len(all_drift_reports) if all_drift_reports else 0,
                        'drift_history_summary': self._summarize_drift_history(all_drift_reports)
                    })
                    
        except Exception as e:
            logger.warning(f"Could not collect drift metrics: {e}")
        
        return drift_metrics if drift_metrics else {"status": "no_drift_data_available"}
    
    def _summarize_drift_history(self, drift_reports):
        """Summarize drift history across multiple batches"""
        if not drift_reports:
            return {}
        
        summary = {
            'total_batches': len(drift_reports),
            'drift_types_observed': [],
            'most_frequent_affected_features': {},
            'average_confidence': 0.0,
            'drift_trend': 'stable'
        }
        
        # Count drift types
        drift_type_counts = {}
        feature_counts = {}
        total_confidence = 0.0
        confidence_count = 0
        
        for report in drift_reports:
            # Track drift types
            drift_type = report.get('drift_type')
            if drift_type:
                drift_type_counts[drift_type] = drift_type_counts.get(drift_type, 0) + 1
            
            # Track affected features
            if report.get('affected_features'):
                for feature in report['affected_features']:
                    feature_name = feature.get('feature', '')
                    if feature_name:
                        feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
            
            # Track confidence
            confidence = report.get('confidence', 0)
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
        
        summary['drift_types_observed'] = list(drift_type_counts.keys())
        summary['most_frequent_affected_features'] = dict(
            sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        summary['average_confidence'] = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        # Simple trend analysis (last 5 batches)
        recent_reports = drift_reports[-5:]
        recent_drift_count = sum(1 for r in recent_reports if r.get('drift_detected', False))
        
        if recent_drift_count >= 3:
            summary['drift_trend'] = 'increasing'
        elif recent_drift_count <= 1:
            summary['drift_trend'] = 'decreasing'
        else:
            summary['drift_trend'] = 'stable'
        
        return summary
    
    def _collect_hyperparameters(self):
        """Collect hyperparameter information"""
        if self.model_pipeline and hasattr(self.model_pipeline, 'best_params'):
            return self.model_pipeline.best_params
        return {}
    
    def pipeline_mode(self, initial_batches: int = 3, update_every: int = 5, max_batches: int = 20):
        """Full pipeline mode: process batches incrementally with learning and updates"""
        logger.info("=== PIPELINE MODE (FULL WORKFLOW) ===")
        logger.info(f"Initial batches for training: {initial_batches}")
        logger.info(f"Update model every: {update_every} batches")
        logger.info(f"Maximum batches to process: {max_batches}")
        
        # Initialize pipeline using consolidated config (no parameter needed)
        pipeline = ModelPipeline()
        
        # Use data collection config from consolidated config for DataStream
        data_stream = DataStream(
            self.config.data_collection['sources'],
            self.config.data_collection['batch_size'], 
            self.config.data_collection['delay']
        )
        
        batch_counter = 0
        performance_history = []
        
        logger.info("\n=== PHASE 1: INITIAL TRAINING ===")
        # Process initial batches for training
        initial_data = []
        for i, batch in enumerate(data_stream.stream()):
            if i >= initial_batches:
                break

            logger.info(f"Processing initial batch {i+1}/{initial_batches}: {len(batch)} rows")
            
            # Use data analysis config from consolidated config instead of loading from file
            dq_results = evaluate_reference_rules_on_batch(batch, self.config.data_analysis)
            
            if dq_results.get('enabled', False) and not dq_results.get('error'):
                logger.info("  Data quality: PASSED")
            else:
                logger.warning(f"  Data quality: ISSUES - {dq_results.get('error', 'Unknown')}")
            
            initial_data.append(batch)
            batch_counter += 1
        
        if not initial_data:
            logger.error("No initial data collected for training")
            return False
        
        # Combine initial batches and train - SAVE TO TEMP FILE for training
        temp_data_dir = "temp_training_data"
        os.makedirs(temp_data_dir, exist_ok=True)
        temp_data_file = os.path.join(temp_data_dir, f"initial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.concat(initial_data, ignore_index=True).to_csv(temp_data_file, index=False)
        logger.info(f"Initial training data saved to: {temp_data_file} ({len(pd.concat(initial_data))} rows)")
        
        # Create a temporary DataLoader that uses our collected data
        class TemporaryDataLoader:
            def __init__(self, data_file):
                self.data_file = data_file
                
            def load_all_files(self):
                return pd.read_csv(self.data_file)
        
        # Replace the data loader temporarily
        original_loader = pipeline.data_loader
        pipeline.data_loader = TemporaryDataLoader(temp_data_file)
        
        # Run initial training
        results_df, _ = pipeline.run()
        
        # Restore original data loader
        pipeline.data_loader = original_loader
        
        if results_df is None:
            logger.error("Initial training failed")
            # Clean up temp file
            if os.path.exists(temp_data_file):
                os.unlink(temp_data_file)
            return False
        
        # Clean up temp file and directory
        if os.path.exists(temp_data_file):
            os.unlink(temp_data_file)
        # Remove directory if empty
        try:
            if os.path.exists(temp_data_dir) and len(os.listdir(temp_data_dir)) == 0:
                os.rmdir(temp_data_dir)
        except OSError:
            logger.warning(f"Could not remove directory {temp_data_dir}, may not be empty")
        
        logger.info("\n=== PHASE 2: INCREMENTAL PROCESSING ===")
        # Continue processing batches with monitoring and updates
        current_best_model = pipeline.best_model if hasattr(pipeline, 'best_model') else None
        
        
        # Continue with the rest of the batches - start after initial batches already processed
        remaining_batches = max_batches - batch_counter
        logger.info(f"Processing remaining {remaining_batches} batches...")
        
        for i, batch in enumerate(data_stream.stream()):
            if i >= remaining_batches:
                logger.info(f"Reached maximum batch limit: {max_batches}")
                break
                
            batch_num = batch_counter + i + 1
            logger.info(f"\nProcessing batch {batch_num}: {len(batch)} rows")
            
            # Data quality check using consolidated config with batch information
            dq_results = evaluate_reference_rules_on_batch(
                batch, 
                self.config.data_analysis,
                batch_info={'batch_num': batch_num, 'timestamp': datetime.now().isoformat()}
            )
            
            if dq_results.get('enabled', False) and not dq_results.get('error'):
                logger.info("  Data quality: PASSED")
            else:
                logger.warning(f"  Data quality: ISSUES - {dq_results.get('error', 'Unknown')}")
            
            # Preprocess batch for prediction
            try:
                # Pass the dataframe directly to preprocess method
                preprocessed = pipeline.preprocessor.preprocess(batch.copy(), fit_scaler=False)
                
                if isinstance(preprocessed, tuple) and len(preprocessed) >= 2:
                    X_batch, y_batch, feature_cols = preprocessed[0], preprocessed[1], preprocessed[2]
                    
                    # Make predictions if we have a model
                    if current_best_model is not None and hasattr(current_best_model, 'predict'):
                        start_time = time.time()
                        predictions = current_best_model.predict(X_batch)
                        inference_time = time.time() - start_time
                        
                        # Calculate metrics if we have true labels
                        if y_batch is not None and len(y_batch) > 0:
                            from sklearn.metrics import accuracy_score, f1_score
                            accuracy = accuracy_score(y_batch, predictions)
                            f1 = f1_score(y_batch, predictions, average='weighted')
                            
                            logger.info("  Prediction metrics:")
                            logger.info(f"    Accuracy: {accuracy:.4f}")
                            logger.info(f"    F1-score: {f1:.4f}")
                            logger.info(f"    Inference time: {inference_time:.3f}s")
                            
                            # Record performance
                            performance_history.append({
                                'batch': batch_num,
                                'accuracy': accuracy,
                                'f1': f1,
                                'inference_time': inference_time,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            logger.info(f"  Predictions made in {inference_time:.3f}s (no ground truth for evaluation)")
                            
                    else:
                        logger.warning("  No model available for prediction")
                        
                else:
                    logger.warning("  Could not preprocess batch for prediction")
                    
            except Exception as e:
                logger.error(f"  Error processing batch: {e}", exc_info=True)
                continue
            
            # Update model if it's time AND we have ground truth
            if batch_num > initial_batches and (batch_num - initial_batches) % update_every == 0:
                if 'y_batch' in locals() and y_batch is not None and len(y_batch) > 0:
                    logger.info(f"\n=== MODEL UPDATE (Batch {batch_num}) ===")
                    
                    try:
                        # Use the processed batch data that was already prepared
                        if 'X_batch' in locals() and 'y_batch' in locals():
                            updated_model, metrics = pipeline.incremental_update(
                                X_new=X_batch, y_new=y_batch, model_name='RandomForest'
                            )
                            
                            if updated_model is not None:
                                current_best_model = updated_model
                                logger.info("Model updated successfully")
                                
                                # Update maintenance system
                                if hasattr(pipeline, 'model_maintenance'):
                                    pipeline.model_maintenance.package_and_register_model(
                                        updated_model, 'RandomForest', metrics, 
                                        feature_cols, pipeline.preprocessor
                                    )
                            else:
                                logger.warning("Model update skipped - no valid data available")
                        
                    except Exception as e:
                        logger.error(f"Model update failed: {e}")
                else:
                    logger.info("  Skip model update: No ground truth labels available")
            
            batch_counter = batch_num
        
        # Generate final summary
        logger.info("\n=== FINAL SUMMARY ===")
        logger.info(f"Total batches processed: {batch_counter}")
        logger.info(f"Performance history: {len(performance_history)} records")
        
        # Save performance history
        if performance_history:
            perf_df = pd.DataFrame(performance_history)
            perf_file = f"pipeline_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            perf_df.to_csv(perf_file, index=False)
            logger.info(f"Performance data saved to: {perf_file}")
            
            # Calculate statistics
            avg_accuracy = perf_df['accuracy'].mean()
            avg_f1 = perf_df['f1'].mean()
            avg_inference_time = perf_df['inference_time'].mean()
            
            logger.info(f"Average accuracy: {avg_accuracy:.4f}")
            logger.info(f"Average F1-score: {avg_f1:.4f}")
            logger.info(f"Average inference time: {avg_inference_time:.3f}s")
        else:
            logger.warning("No performance metrics recorded - check if ground truth was available")
        
        # Update the instance's best model reference
        self.best_model = current_best_model
        self.model_pipeline = pipeline
        
        return True
        
    def _generate_visual_summary(self, summary_data):
        """Generate visual dashboard"""
        # Simple text-based visualization for now
        logger.info("\n=== SUMMARY DASHBOARD ===")
        logger.info(f"Timestamp: {summary_data['timestamp']}")
        if 'model_metrics' in summary_data:
            logger.info(f"Model Metrics: {summary_data['model_metrics']}")
        if 'performance' in summary_data:
            logger.info(f"Performance: {summary_data['performance']}")
        if 'data_quality' in summary_data:
            logger.info(f"Data Quality: {summary_data['data_quality']}")
        if 'hyperparameters' in summary_data:
            logger.info(f"Hyperparameters: {summary_data['hyperparameters']}")


def main():
    parser = argparse.ArgumentParser(description='MLOps Pipeline Controller')
    parser.add_argument('-mode', type=str, required=True, 
                       choices=['inference', 'update', 'summary', 'pipeline'],
                       help='Operation mode')
    parser.add_argument('-file', type=str, help='Input file for inference mode')
    parser.add_argument('-initial_batches', type=int, default=3, 
                       help='Number of initial batches for training (pipeline mode)')
    parser.add_argument('-update_every', type=int, default=5, 
                       help='Update model every N batches (pipeline mode)')
    parser.add_argument('-max_batches', type=int, default=20, 
                       help='Maximum batches to process (pipeline mode)')
    parser.add_argument('-batch_limit', type=int, default=5, 
                       help='Maximum number of batches to process in update mode (default: 5)')
    
    args = parser.parse_args()
    
    pipeline = MLOpsPipeline()
    
    try:
        if args.mode == 'inference':
            if not args.file:
                logger.error("File path required for inference mode")
                sys.exit(1)
            result = pipeline.inference_mode(args.file)
            if result:
                logger.info(f"Success! Result saved to: {result}")
            else:
                logger.error("Inference failed")
                sys.exit(1)
                
        elif args.mode == 'summary':
            report_path = pipeline.summary_mode()
            logger.info(f"Summary report generated: {report_path}")
                
        elif args.mode == 'update':
            success = pipeline.update_mode(batch_limit=args.batch_limit)
            if success:
                logger.info("Update completed successfully")
            else:
                logger.error("Update failed")
                sys.exit(1)
                
        elif args.mode == 'pipeline':
            success = pipeline.pipeline_mode(
                initial_batches=args.initial_batches,
                update_every=args.update_every,
                max_batches=args.max_batches
            )
            if success:
                logger.info("Pipeline execution completed successfully")
            else:
                logger.error("Pipeline execution failed")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

