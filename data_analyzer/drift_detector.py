#!/usr/bin/env python3
"""
Data Drift Detection Module
Handles detection of data and concept drift in streaming data
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import json
import os
from datetime import datetime
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detects data drift using statistical tests and rule-based methods"""
    
    def __init__(self, config: Dict[str, Any], reference_data: pd.DataFrame = None):
        self.config = config
        self.reference_data = reference_data
        # Use a more conservative threshold to reduce false positives
        self.drift_threshold = config.get('drift_threshold', 0.01)  # 1% significance level - more conservative
        self.min_effect_size = config.get('min_effect_size', 0.1)  # Minimum practical significance
        self.drift_history = []
        
        # Fields to monitor for drift (whitelist)
        self.monitored_fields = [
            'CLAIM_PAID',      # Target variable - concept drift
            'INSURED_VALUE',   # Numerical feature
            'PREMIUM',         # Numerical feature  
            'PROD_YEAR',       # Numerical feature
            'SEATS_NUM',       # Numerical feature
            # 'EFFECTIVE_YR',   # Removed - contains garbage data
            'SEX',             # Categorical - distribution drift
            'INSR_TYPE'        # Categorical - distribution drift
        ]
        
        # Initialize reference statistics if data provided
        if reference_data is not None:
            self._initialize_reference_stats(reference_data)
    
    def _initialize_reference_stats(self, reference_data: pd.DataFrame):
        """Initialize statistics from reference data (only monitored fields)"""
        self.reference_stats = {}
        
        # Only monitor whitelisted fields
        monitored_cols = [col for col in self.monitored_fields if col in reference_data.columns]
        
        for col in monitored_cols:
            if reference_data[col].dtype in ['float64', 'int64']:
                col_data = reference_data[col].dropna()
                if len(col_data) > 0:
                    self.reference_stats[col] = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'n_samples': len(col_data)
                    }
            elif reference_data[col].dtype == 'object':
                col_data = reference_data[col].dropna()
                if len(col_data) > 0:
                    value_counts = col_data.value_counts(normalize=True)
                    self.reference_stats[col] = {
                        'value_distribution': value_counts.to_dict(),
                        'n_categories': len(value_counts),
                        'n_samples': len(col_data)
                    }
    
    def detect_drift(self, current_data: pd.DataFrame, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift between reference and current data"""
        if self.reference_data is None:
            self._initialize_reference_stats(current_data)
            return {
                'drift_detected': False,
                'status': 'reference_initialized',
                'batch_info': batch_info
            }
        
        drift_results = {
            'drift_detected': False,
            'drift_type': None,
            'confidence': 0.0,
            'affected_features': [],
            'batch_info': batch_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for feature drift (covariate drift)
        feature_drift = self._detect_feature_drift(current_data)
        if feature_drift['drift_detected']:
            drift_results.update(feature_drift)
            drift_results['drift_detected'] = True
            drift_results['drift_type'] = 'covariate'
        
        # Check for concept drift (if target variable available)
        if 'CLAIM_PAID' in current_data.columns and 'CLAIM_PAID' in self.reference_data.columns:
            concept_drift = self._detect_concept_drift(current_data)
            if concept_drift['drift_detected']:
                drift_results['drift_detected'] = True
                drift_results['drift_type'] = 'concept'
                drift_results.update({
                    'concept_shift_confidence': concept_drift['confidence'],
                    'target_distribution_change': concept_drift['distribution_change']
                })
        
        # Check for data quality drift
        quality_drift = self._detect_quality_drift(current_data)
        if quality_drift['quality_issues']:
            drift_results.update(quality_drift)
        
        # Record drift history
        self.drift_history.append(drift_results.copy())
        
        # Save drift report
        self._save_drift_report(drift_results, batch_info)
        
        return drift_results
    
    def _detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in feature distributions using robust statistics"""
        drift_results = {
            'drift_detected': False,
            'affected_features': [],
            'feature_drift_scores': {},
            'confidence': 0.0
        }
        
        drift_scores = []
        
        for col, ref_stats in self.reference_stats.items():
            if col not in current_data.columns:
                continue
                
            current_col_data = current_data[col].dropna()
            if len(current_col_data) == 0:
                continue
            
            if 'mean' in ref_stats:  # Numerical column
                ref_col_data = self.reference_data[col].dropna()
                if len(ref_col_data) == 0:
                    continue
                
                # Use IQR-based drift detection instead of variance for robustness
                ref_q1, ref_q3 = np.percentile(ref_col_data, [25, 75])
                current_q1, current_q3 = np.percentile(current_col_data, [25, 75])
                
                # Calculate IQR-based similarity instead of KS test
                iqr_similarity = 1 - min(1.0, abs((current_q1 - ref_q1) / (ref_q3 - ref_q1 + 1e-8)) + 
                                           abs((current_q3 - ref_q3) / (ref_q3 - ref_q1 + 1e-8)))
                
                # Only flag as drift if significant change in distribution shape
                if iqr_similarity < 0.8:  # More than 20% change in quartiles
                    p_value_equivalent = 1 - iqr_similarity
                    drift_scores.append(p_value_equivalent)
                    
                    drift_results['affected_features'].append({
                        'feature': col,
                        'drift_type': 'robust_distribution',
                        'confidence': 1 - iqr_similarity,
                        'test': 'IQR_quartile_change',
                        'q1_change_percent': ((current_q1 - ref_q1) / ref_q1 * 100) if ref_q1 != 0 else 0,
                        'q3_change_percent': ((current_q3 - ref_q3) / ref_q3 * 100) if ref_q3 != 0 else 0
                    })
                
            elif 'value_distribution' in ref_stats:  # Categorical column
                # Keep existing categorical test (it's more stable)
                ref_dist = ref_stats['value_distribution']
                current_dist = current_col_data.value_counts(normalize=True).to_dict()
                
                all_categories = set(ref_dist.keys()).union(set(current_dist.keys()))
                ref_counts = [ref_dist.get(cat, 0) for cat in all_categories]
                current_counts = [current_dist.get(cat, 0) for cat in all_categories]
                
                chi2_stat, p_value = stats.chisquare(current_counts, ref_counts)
                drift_scores.append(1 - p_value)
                
                if p_value < self.drift_threshold:
                    drift_results['affected_features'].append({
                        'feature': col,
                        'drift_type': 'categorical_distribution',
                        'confidence': 1 - p_value,
                        'test': 'Chi-squared',
                        'statistic': chi2_stat,
                        'p_value': p_value
                    })
        
        # Calculate overall confidence
        if drift_scores:
            drift_results['confidence'] = np.mean(drift_scores)
            drift_results['drift_detected'] = len(drift_results['affected_features']) > 0
        
        return drift_results
    
    def _detect_concept_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect concept drift in target variable distribution"""
        drift_results = {
            'drift_detected': False,
            'confidence': 0.0,
            'distribution_change': 0.0
        }
        
        ref_target = self.reference_data['CLAIM_PAID'].dropna()
        current_target = current_data['CLAIM_PAID'].dropna()
        
        if len(ref_target) == 0 or len(current_target) == 0:
            return drift_results
        
        # Compare target distributions
        ref_dist = ref_target.value_counts(normalize=True)
        current_dist = current_target.value_counts(normalize=True)
        
        # Calculate distribution change (KL divergence-like metric)
        distribution_change = 0.0
        for category in set(ref_dist.index).union(set(current_dist.index)):
            ref_prob = ref_dist.get(category, 0.001)
            current_prob = current_dist.get(category, 0.001)
            distribution_change += abs(ref_prob - current_prob)
        
        drift_results['distribution_change'] = distribution_change
        
        # Simple threshold-based detection
        if distribution_change > 0.1:  # 10% distribution change
            drift_results['drift_detected'] = True
            drift_results['confidence'] = min(distribution_change, 1.0)
        
        return drift_results
    
    def _detect_quality_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in data quality metrics"""
        quality_results = {
            'quality_issues': False,
            'quality_metrics': {},
            'anomalies': []
        }
        
        # Check for missing value drift
        ref_missing = self.reference_data.isnull().sum().sum() / self.reference_data.size
        current_missing = current_data.isnull().sum().sum() / current_data.size
        missing_drift = abs(ref_missing - current_missing)
        
        quality_results['quality_metrics']['missing_value_drift'] = missing_drift
        
        if missing_drift > 0.05:  # 5% change in missing values
            quality_results['quality_issues'] = True
            quality_results['anomalies'].append({
                'type': 'missing_value_drift',
                'severity': missing_drift,
                'description': f"Missing value ratio changed by {missing_drift:.2%}"
            })
        
        # Check for outlier drift
        ref_outliers = self._detect_outliers(self.reference_data)
        current_outliers = self._detect_outliers(current_data)
        outlier_drift = abs(ref_outliers - current_outliers) / max(ref_outliers, 1)
        
        quality_results['quality_metrics']['outlier_drift'] = outlier_drift
        
        if outlier_drift > 0.5:  # 50% change in outliers
            quality_results['quality_issues'] = True
            quality_results['anomalies'].append({
                'type': 'outlier_drift',
                'severity': outlier_drift,
                'description': f"Outlier count changed by {outlier_drift:.2%}"
            })
        
        return quality_results
    
    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """Detect outliers using IQR method"""
        outlier_count = 0
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_count += len(outliers)
        
        return outlier_count
    
    def _save_drift_report(self, drift_results: Dict[str, Any], batch_info: Dict[str, Any]):
        """Save drift detection report"""
        try:
            report_dir = self.config.get('io', {}).get('artifacts_dir', 'artifacts')
            os.makedirs(report_dir, exist_ok=True)
            
            report_file = os.path.join(report_dir, f"drift_report_batch{batch_info['batch_num']:04d}.json")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(drift_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Drift report saved: {report_file}")
            
        except Exception as e:
            logger.warning(f"Could not save drift report: {e}")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection history"""
        if not self.drift_history:
            return {"status": "no_drift_history"}
        
        recent_drifts = [d for d in self.drift_history if d['drift_detected']]
        
        return {
            "total_batches_analyzed": len(self.drift_history),
            "batches_with_drift": len(recent_drifts),
            "drift_rate": len(recent_drifts) / len(self.drift_history) if self.drift_history else 0,
            "last_drift_batch": recent_drifts[-1]['batch_info']['batch_num'] if recent_drifts else None,
            "drift_types": [d['drift_type'] for d in recent_drifts if d['drift_type']],
            "most_affected_features": self._get_most_affected_features()
        }
    
    def _get_most_affected_features(self) -> List[str]:
        """Get features most frequently affected by drift"""
        feature_counts = {}
        
        for drift_event in self.drift_history:
            if drift_event['drift_detected'] and 'affected_features' in drift_event:
                for feature_info in drift_event['affected_features']:
                    feature_name = feature_info['feature']
                    feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
        
        return sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]


# Convenience function for using drift detector
def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                    config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for one-time drift detection"""
    config = config or {}
    detector = DataDriftDetector(config, reference_data)
    
    # Create batch info for the detection
    batch_info = {
        'batch_num': 1,
        'timestamp': datetime.now().isoformat()
    }
    
    return detector.detect_drift(current_data, batch_info)
