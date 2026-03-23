#!/usr/bin/env python3
"""
Unified Configuration Management
Handles loading and accessing all pipeline configurations from a single file
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional


class Config:
    """
    Main configuration class for the MLOps pipeline
    Provides centralized access to all configuration sections
    """
    
    def __init__(self, config_file: str = "unified_config.yaml"):
        self.config_file = config_file
        self._config_data: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from unified YAML file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self._config_data = yaml.safe_load(f)
        
        logging.info(f"Loaded unified configuration from {self.config_file}")
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
    
    @property
    def data_collection(self) -> Dict[str, Any]:
        """Get data collection configuration"""
        return self._config_data.get('data_collection', {})
    
    @property
    def data_analysis(self) -> Dict[str, Any]:
        """Get data analysis configuration"""
        return self._config_data.get('data_analysis', {})
    
    @property  
    def model_training(self) -> Dict[str, Any]:
        """Get model training configuration"""
        return self._config_data.get('model_training', {})
    
    @property
    def model_maintenance(self) -> Dict[str, Any]:
        """Get model maintenance configuration"""
        return self._config_data.get('model_maintenance', {})
    
    @property
    def pipeline_settings(self) -> Dict[str, Any]:
        """Get pipeline settings"""
        return self._config_data.get('pipeline_settings', {})
    
    @property
    def performance_thresholds(self) -> Dict[str, Any]:
        """Get performance thresholds"""
        return self._config_data.get('performance_thresholds', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._config_data.get('logging', {})
    
    @property
    def model_registry(self) -> Dict[str, Any]:
        """Get model registry configuration"""
        return self._config_data.get('model_registry', {})
    
    def get_section(self, section_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a specific configuration section by name"""
        return self._config_data.get(section_name, default or {})
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        current = self._config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to configuration sections"""
        return self._config_data.get(key, {})
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration section exists"""
        return key in self._config_data


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_file: str = "unified_config.yaml") -> Config:
    """
    Get or create the global configuration instance
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


def reload_config():
    """Reload the global configuration"""
    global _config_instance
    if _config_instance:
        _config_instance.reload()


# Example usage:
if __name__ == "__main__":
    # Initialize configuration
    config = get_config()
    
    # Access different sections
    print("Data collection batch size:", config.data_collection.get('batch_size'))
    print("Model training CV folds:", config.model_training.get('cv_folds'))
    print("Performance thresholds for accuracy:", config.performance_thresholds.get('accuracy'))
    
    # Access nested values
    print("Data analysis date format:", config.get_nested('data_analysis', 'dq', 'parsing', 'date_format'))
    print("Model maintenance performance check interval:", 
          config.get_nested('model_maintenance', 'performance_check_interval'))