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
import yaml
import json
from datetime import datetime
from model_pipeline.model_pipeline import DataPreprocessor

# Add project directories to path
sys.path.extend(['data_collection', 'data_analyzer', 'model_pipeline'])

from data_collection.data_collection import DataStream, load_config as load_dc_config
from data_analyzer.dq_pipeline import evaluate_reference_rules_on_batch, load_yaml
from model_pipeline.model_pipeline import load_config, ModelPipeline, DataLoader


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
        """Load main configuration"""
        config_path = "pipeline_config.yaml"
        if not os.path.exists(config_path):
            config = {
                "data_collection": {
                    "config_path": "data_collection/config.yaml"
                },
                "data_analysis": {
                    "config_path": "data_analyzer/config.yaml"
                },
                "model_training": {
                    "config_path": "model_pipeline/model_config.yaml"
                },
                "model_registry": {"path": "model_registry"},
                "performance_thresholds": {
                    "accuracy": 0.7,
                    "f1": 0.65,
                    "inference_time": 1.0,
                    "memory_limit_mb": 500
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Created main config: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def inference_mode(self, file_path: str):
        """Apply the best model to external data"""
        logger.info(f"=== INFERENCE MODE ===")
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
    
    def update_mode(self):
        """Update/retrain model with new data"""
        logger.info("=== UPDATE MODE ===")
        
        try:
            # Load and analyze new data
            dc_config = load_dc_config(self.config['data_collection']['config_path'])
            data_stream = DataStream(dc_config['sources'], dc_config['batch_size'], dc_config['delay'])
            
            for batch in data_stream.stream():
                logger.info(f"Processing batch: {len(batch)} rows")
                
                # Data quality analysis
                dq_config = load_yaml(self.config['data_analysis']['config_path'])
                dq_results = evaluate_reference_rules_on_batch(batch, dq_config)
                
                if dq_results.get('enabled', False):
                    logger.info(f"Data quality check passed: {dq_results}")
                
                # Update model pipeline
                if self.model_pipeline is None:
                    self.model_pipeline = ModelPipeline(self.config['model_training']['config_path'])
                
                # Use incremental_update method which requires proper data preprocessing
                # First, we need to preprocess the batch like in the pipeline mode
                try:
                    preprocessed = self.model_pipeline.preprocessor.preprocess(batch.copy(), fit_scaler=False)
                    if isinstance(preprocessed, tuple) and len(preprocessed) >= 2:
                        X_batch, y_batch = preprocessed[0], preprocessed[1]
                        updated_model, metrics = self.model_pipeline.incremental_update(X_new=X_batch, y_new=y_batch, model_name='RandomForest')
                        if updated_model is not None:
                            logger.info(f"Model updated successfully for batch {batch_info['batch_num']}")
                        else:
                            logger.warning(f"Model update failed for batch {batch_info['batch_num']}")
                    else:
                        logger.warning("Could not preprocess batch for update")
                except Exception as e:
                    logger.error(f"Error preprocessing batch for update: {e}")
                
                # Update best model
                self._update_best_model()
            
            logger.info("Update completed successfully")
            return True
            
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
            "performance": self._collect_performance_metrics(),
            "hyperparameters": self._collect_hyperparameters()
        }
        
        # Save summary report
        report_path = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Generate visual summary
        self._generate_visual_summary(summary_data)
        
        logger.info(f"Summary report generated: {report_path}")
        return report_path
    
    def _load_best_model(self):
        """Load the best performing model from registry"""
        registry_path = self.config['model_registry']['path']
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
        if self.model_pipeline and hasattr(self.model_pipeline, 'best_model'):
            self.best_model = self.model_pipeline.best_model
    
    def _collect_model_metrics(self):
        """Collect model performance metrics"""
        if self.model_pipeline and hasattr(self.model_pipeline, 'model_performance'):
            return self.model_pipeline.model_performance
        return {}
    
    def _collect_data_quality(self):
        """Collect data quality metrics"""
        # Implementation would depend on your data quality tracking
        return {"status": "implement_data_quality_tracking"}
    
    def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        return {"status": "implement_performance_monitoring"}
    
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
        
        # Initialize pipeline - pass the config path, not the loaded config
        model_config_path = self.config['model_training']['config_path']
        pipeline = ModelPipeline(model_config_path)
        
        # Load data collection config
        dc_config = load_dc_config(self.config['data_collection']['config_path'])
        data_stream = DataStream(dc_config['sources'], dc_config['batch_size'], dc_config['delay'])
        
        batch_counter = 0
        performance_history = []
        
        logger.info("\n=== PHASE 1: INITIAL TRAINING ===")
        # Process initial batches for training
        initial_data = []
        for i, batch in enumerate(data_stream.stream()):
            if i >= initial_batches:
                break
                
            logger.info(f"Processing initial batch {i+1}/{initial_batches}: {len(batch)} rows")
            
            # Data quality check
            dq_config = load_yaml(self.config['data_analysis']['config_path'])
            dq_results = evaluate_reference_rules_on_batch(batch, dq_config)
            
            if dq_results.get('enabled', False) and not dq_results.get('error'):
                logger.info(f"  Data quality: PASSED")
            else:
                logger.warning(f"  Data quality: ISSUES - {dq_results.get('error', 'Unknown')}")
            
            initial_data.append(batch)
            batch_counter += 1
        
        if not initial_data:
            logger.error("No initial data collected for training")
            return False
        
        # Combine initial batches and train - SAVE TO TEMP FILE for training
        import tempfile
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
            
            # Data quality check
            dq_results = evaluate_reference_rules_on_batch(batch, dq_config)
            
            if dq_results.get('enabled', False) and not dq_results.get('error'):
                logger.info(f"  Data quality: PASSED")
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
                            
                            logger.info(f"  Prediction metrics:")
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
                                logger.info(f"Model updated successfully")
                                
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
                    logger.info(f"  Skip model update: No ground truth labels available")
            
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
                
        elif args.mode == 'update':
            success = pipeline.update_mode()
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

