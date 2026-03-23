import pandas as pd
import numpy as np
import os
import glob
import logging
import sys
import json
import joblib
import matplotlib.pyplot as plt
import shap
import time
from datetime import datetime
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
# Add the project root to path so we can import model_maintenance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_maintenance.model_maintenance import ModelMaintenance
from config import get_config


def setup_logging(log_level="INFO", log_file="model_pipeline.log"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class DataLoader:
    
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        logger.info(f"Инициализирован загрузчик данных: {data_folder}")
    
    def load_all_files(self):
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        
        if not all_files:
            logger.error(f"Нет CSV файлов в папке {self.data_folder}")
            raise ValueError(f"Нет CSV файлов в папке {self.data_folder}")
        
        logger.info(f"Найдено CSV файлов: {len(all_files)}")
        for f in all_files:
            logger.debug(f"  - {os.path.basename(f)}")
        
        column_names = ['SEX', 'INSR_BEGIN', 'INSR_END', 'INSR_TYPE', 
                        'INSURED_VALUE', 'PREMIUM', 'OBJECT_ID', 'PROD_YEAR', 'SEATS_NUM', 
                        'CARRYING_CAPACITY', 'TYPE_VEHICLE', 'CCM_TON', 'MAKE', 'USAGE', 'CLAIM_PAID']
        
        df_list = []
        for file in all_files:
            try:
                logger.debug(f"Загрузка: {os.path.basename(file)}")
                df_part = pd.read_csv(file, names=column_names, header=0)
                df_list.append(df_part)
                logger.debug(f"  Загружено строк: {len(df_part)}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке {file}: {e}")
                continue
        
        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Всего загружено записей: {len(df)}")
        
        return df


class DataPreprocessor:
    
    def __init__(self):
        logger.info("Инициализирован препроцессор данных")
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True):
        logger.info("=" * 40)
        logger.info("ПРЕДОБРАБОТКА ДАННЫХ")
        logger.info("=" * 40)
        
        df_processed = df.copy()
        
        df_processed = self._handle_missing_values(df_processed)
        
        df_processed, label_encoders = self._handle_categorical_variables(df_processed)
        
        df_processed = self._handle_numeric_variables(df_processed)
        
        df_processed['HAS_CLAIM'] = (df_processed['CLAIM_PAID'] > 0).astype(int)
        
        logger.info(f"Результат: {len(df_processed)} записей, {len(df_processed.columns)} признаков")
        
        class_counts = df_processed['HAS_CLAIM'].value_counts()
        logger.info(f"  Распределение HAS_CLAIM: 0={class_counts[0]} ({class_counts[0]/len(df_processed)*100:.1f}%), "
                   f"1={class_counts[1]} ({class_counts[1]/len(df_processed)*100:.1f}%)")
        
        if class_counts[1] / len(df_processed) < 0.05:
            logger.warning("  ВНИМАНИЕ: Сильный дисбаланс классов (<5% положительных)!")
            logger.warning("  Будет использовано взвешивание классов")
        
        feature_cols = [col for col in df_processed.columns if col not in ['HAS_CLAIM', 'CLAIM_PAID']]
        X = df_processed[feature_cols].values
        y = df_processed['HAS_CLAIM'].values
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                logger.warning("Scaler не обучен, выполняем fit_transform")
                X_scaled = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X_scaled = self.scaler.transform(X)
        
        return X_scaled, y, feature_cols, label_encoders
    
    def _handle_missing_values(self, df: pd.DataFrame):
        logger.debug("Обработка пропусков...")
        
        df_processed = df.copy()
        
        numeric_cols = ['INSURED_VALUE', 'PREMIUM', 'PROD_YEAR', 
                        'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON', 'CLAIM_PAID']
        
        missing_before = df_processed[numeric_cols].isnull().sum().sum()
        logger.debug(f"  Пропусков в числовых колонках до: {missing_before}")
        
        for col in numeric_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                if col in ['INSURED_VALUE', 'CLAIM_PAID']:
                    df_processed[col].fillna(0, inplace=True)
                    logger.debug(f"    {col}: заполнено 0")
                else:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
                    logger.debug(f"    {col}: заполнено медианой {median_val:.2f}")
        
        categorical_cols = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col].fillna(mode_val, inplace=True)
                logger.debug(f"    {col}: заполнено модой '{mode_val}'")
        
        missing_after = df_processed[numeric_cols].isnull().sum().sum()
        logger.debug(f"  Пропусков в числовых колонках после: {missing_after}")
        
        return df_processed
    
    def _handle_categorical_variables(self, df: pd.DataFrame):
        logger.debug("Обработка категориальных переменных...")
        
        df_processed = df.copy()
        label_encoders = {}
        
        categorical_cols = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
                logger.debug(f"  {col}: закодировано, классов: {len(le.classes_)}")
        
        return df_processed, label_encoders
    
    def _handle_numeric_variables(self, df: pd.DataFrame):
        logger.debug("Обработка числовых переменных...")
        
        df_processed = df.copy()
        
        df_processed['INSR_BEGIN'] = pd.to_datetime(df_processed['INSR_BEGIN'], errors='coerce', format='%d-%b-%y')
        df_processed['INSR_END'] = pd.to_datetime(df_processed['INSR_END'], errors='coerce', format='%d-%b-%y')
        df_processed['POLICY_DURATION'] = (df_processed['INSR_END'] - df_processed['INSR_BEGIN']).dt.days
        df_processed['POLICY_DURATION'].fillna(1, inplace=True)
        
        logger.debug("  Созданы признаки: POLICY_DURATION")
        
        drop_cols = ['INSR_BEGIN', 'INSR_END', 'OBJECT_ID', 'MAKE', 'TYPE_VEHICLE', 
                     'USAGE', 'SEX', 'INSR_TYPE', 'EFFECTIVE_YR']
        
        df_processed.drop(columns=[col for col in drop_cols if col in df_processed.columns], 
                         inplace=True)
        
        return df_processed


class ModelTrainer:
    
    def __init__(self, config: Dict):
        self.config = config
        self.models_config = config['models']
        self.use_class_weight = config.get('use_class_weight', True)
        logger.info("Инициализирован тренер моделей")
        if self.use_class_weight:
            logger.info("  Включено взвешивание классов для борьбы с дисбалансом")
    
    def create_models(self):
        logger.info("Создание моделей")
        
        models = {
            'DecisionTree': DecisionTreeClassifier(
                random_state=self.config['random_state'],
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced' if self.use_class_weight else None
            ),
            'RandomForest': RandomForestClassifier(
                random_state=self.config['random_state'],
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                class_weight='balanced' if self.use_class_weight else None,
                warm_start=True
            ),
            'NeuralNetwork': MLPClassifier(
                random_state=self.config['random_state'],
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                hidden_layer_sizes=(100, 50),
                alpha=0.001,
                learning_rate_init=0.001,
                verbose=False,
                warm_start=True
            )
        }
        
        logger.info(f"Создано моделей: {len(models)}")
        logger.info("  - DecisionTreeClassifier (class_weight=balanced)")
        logger.info("  - RandomForestClassifier (class_weight=balanced, warm_start=True)")
        logger.info("  - MLPClassifier (warm_start=True)")
        
        return models
    
    def train_with_cv(self, model, X_train, y_train, model_name: str):
        logger.info(f"Обучение {model_name}...")
        
        param_grid = self.models_config.get(model_name, {}).copy()
        
        if model_name in ['DecisionTree', 'RandomForest'] and self.use_class_weight:
            if 'class_weight' not in param_grid:
                param_grid['class_weight'] = ['balanced']
        
        if param_grid:
            logger.debug(f"  Параметры для GridSearch: {param_grid}")
            
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=self.config['cv_folds'], 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            logger.info(f"  Выполняется GridSearchCV (k={self.config['cv_folds']})...")
            grid_search.fit(X_train, y_train)
            
            logger.info(f"  Лучшие параметры: {grid_search.best_params_}")
            logger.info(f"  Лучший CV score (ROC-AUC): {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            logger.info("  Обучение завершено")
            return model, None, {}
    
    def incremental_train(self, model, X_new, y_new, model_name: str):
        """Perform incremental training on existing model"""
        logger.info(f"Incremental training for {model_name}")
        
        try:
            # For models that support partial_fit or warm_start
            if hasattr(model, 'partial_fit'):
                # Neural networks can use partial_fit
                model.partial_fit(X_new, y_new)
                logger.info("  Updated using partial_fit")
            elif hasattr(model, 'warm_start') and model.warm_start:
                # RandomForest/DecisionTree with warm_start
                model.fit(X_new, y_new)
                logger.info("  Updated using warm_start")
            else:
                # Fallback: retrain with combined data
                logger.warning("  Model doesn't support incremental learning, retraining from scratch")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    random_state=self.config['random_state'],
                    n_estimators=100,
                    max_depth=10,
                    warm_start=True,
                    class_weight='balanced' if self.use_class_weight else None
                )
                model.fit(X_new, y_new)
            
            return model
            
        except Exception as e:
            logger.error(f"Incremental training failed: {e}")
            return None


class ModelRegistry:
    
    def __init__(self, storage_path: str = './model_registry'):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.metadata_file = os.path.join(storage_path, 'models_metadata.json')
        self.metadata = self._load_metadata()
        
        logger.info(f"Хранилище моделей: {storage_path}")
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Загружено метаданных: {len(data.get('models', {}))} записей")
            return data
        return {'models': {}}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.debug("Метаданные сохранены")
    
    def register_model(self, model, model_name: str, metrics: Dict, 
                       best_params: Dict = None, is_incremental: bool = False) -> int:
        models_dict = self.metadata['models']
        
        if model_name not in models_dict:
            models_dict[model_name] = []
        
        version = len(models_dict[model_name]) + 1
        timestamp = datetime.now().isoformat()
        
        model_path = os.path.join(self.storage_path, f'{model_name}_v{version}.joblib')
        joblib.dump(model, model_path)
        
        entry = {
            'version': version,
            'timestamp': timestamp,
            'path': model_path,
            'metrics': metrics,
            'best_params': best_params or {},
            'is_incremental': is_incremental
        }
        
        models_dict[model_name].append(entry)
        self._save_metadata()
        
        logger.info(f"  Модель сохранена: {model_name} v{version}")
        
        return version
    
    def get_best_model(self, model_name: str, metric: str = 'roc_auc'):
        models_dict = self.metadata['models']
        
        if model_name not in models_dict or not models_dict[model_name]:
            logger.warning(f"Модель {model_name} не найдена")
            return None, None
        
        best = max(models_dict[model_name], key=lambda x: x['metrics'].get(metric, -float('inf')))
        logger.info(f"Лучшая {model_name} v{best['version']}: {metric}={best['metrics'][metric]:.4f}")
        
        model = joblib.load(best['path'])
        return model, best
    
    def get_latest_model(self, model_name: str):
        """Получение последней версии модели"""
        models_dict = self.metadata['models']
        
        if model_name not in models_dict or not models_dict[model_name]:
            logger.warning(f"Модель {model_name} не найдена")
            return None, None
        
        latest = models_dict[model_name][-1]
        logger.info(f"Загружена последняя модель {model_name} v{latest['version']}")
        
        model = joblib.load(latest['path'])
        return model, latest


class ModelEvaluator:
    
    def __init__(self):
        logger.info("Инициализирован оценщик моделей")
    
    def evaluate(self, model, X_test, y_test):
        logger.debug("Выполняется оценка...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        logger.debug(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.debug(f"  Precision: {metrics['precision']:.4f}")
        logger.debug(f"  Recall: {metrics['recall']:.4f}")
        logger.debug(f"  F1: {metrics['f1']:.4f}")
        logger.debug(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def detect_drift(self, model, X_new, y_new, reference_metrics: Dict, 
                     threshold: float = 0.1) -> Tuple[Dict, Dict]:
        logger.info("Проверка model drift...")
        
        current_metrics, _, _ = self.evaluate(model, X_new, y_new)
        
        drift_detected = {}
        for metric, ref_value in reference_metrics.items():
            if metric not in current_metrics:
                continue
            current_value = current_metrics[metric]
            relative_change = abs(current_value - ref_value) / (abs(ref_value) + 1e-8)
            drift_detected[metric] = relative_change > threshold
            
            if relative_change > threshold:
                logger.warning(f"  Дрейф по {metric}: {ref_value:.4f} -> {current_value:.4f} "
                             f"(изм. {relative_change:.1%})")
        
        return drift_detected, current_metrics


class ModelInterpreter:
    
    def __init__(self):
        logger.info("Инициализирован интерпретатор моделей")
    
    def explain_decision_tree(self, model, feature_names: List[str], max_depth: int = 3):
        logger.info("Визуализация дерева решений...")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("  Топ-5 важных признаков:")
        for idx, row in importance.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(20, 10))
        plot_tree(model, 
                  feature_names=feature_names,
                  filled=True, 
                  rounded=True,
                  max_depth=max_depth,
                  fontsize=10)
        plt.title(f'Decision Tree Visualization (max_depth={max_depth})', fontsize=16)
        plt.tight_layout()
        plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
        logger.info("  График дерева сохранен: decision_tree_visualization.png")
        
        return importance
    
    def explain_random_forest(self, model, feature_names: List[str]):
        logger.info("Интерпретация случайного леса...")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("  Топ-5 важных признаков:")
        for idx, row in importance.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'][:10], importance['importance'][:10])
        plt.xlabel('Важность')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig('random_forest_importance.png', dpi=100, bbox_inches='tight')
        logger.info("  График важности признаков сохранен: random_forest_importance.png")
        
        return importance
    
    def explain_shap_any(self, model, X_sample, feature_names: List[str]):
        try:
            logger.info("Вычисление SHAP значений...")
            
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample[:100])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
                shap_values = explainer.shap_values(X_sample[:100])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample[:100], feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=100, bbox_inches='tight')
            logger.info("  SHAP график сохранен: shap_summary.png")
            
            return shap_values
        except Exception as e:
            logger.error(f"  Ошибка SHAP: {e}")
            return None


class ModelPipeline:
    
    def __init__(self):
        config = get_config()
        self.config = config.model_training
        
        # Merge model_maintenance config
        self.config.update(config.model_maintenance)
        
        self.data_loader = DataLoader(self.config['data_folder'])
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer(self.config)
        self.registry = ModelRegistry(self.config['model_registry_path'])
        self.evaluator = ModelEvaluator()
        self.interpreter = ModelInterpreter()
        
        # Initialize model maintenance system (moved after config loading)
        self.model_maintenance = ModelMaintenance(self.config)
        
        logger.info("=" * 60)
        logger.info("ПАЙПЛАЙН ПОСТРОЕНИЯ МОДЕЛИ ИНИЦИАЛИЗИРОВАН")
        logger.info("Целевая переменная: HAS_CLAIM (факт наличия страховой выплаты)")
        logger.info("Модели: Decision Tree + Random Forest + Neural Network (классификация)")
        if self.config.get('use_class_weight', True):
            logger.info("Включено взвешивание классов для борьбы с дисбалансом")
        logger.info("=" * 60)
    
    def run(self):
        logger.info("\n" + "=" * 60)
        logger.info("ЗАПУСК ПАЙПЛАЙНА")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        logger.info("\n ЗАГРУЗКА ДАННЫХ")
        logger.info("-" * 40)
        
        try:
            df_raw = self.data_loader.load_all_files()
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            return None, None
        
        logger.info("\n ПРЕДОБРАБОТКА ДАННЫХ")
        logger.info("-" * 40)
        
        try:
            X, y, self.feature_cols, encoders = self.preprocessor.preprocess(df_raw.copy(), fit_scaler=True)
        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}", exc_info=True)
            return None, None
        
        logger.info(f"  Признаков: {len(self.feature_cols)}")
        logger.info("  Целевая переменная: HAS_CLAIM")
        logger.info(f"  Распределение классов: 0={len(y[y==0])}, 1={len(y[y==1])}")
        logger.info(f"  Доля положительных: {len(y[y==1])/len(y)*100:.2f}%")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y
        )
        
        logger.info(f"  Train: {len(X_train)} записей")
        logger.info(f"  Test: {len(X_test)} записей")
        
        logger.info("\nОБУЧЕНИЕ МОДЕЛЕЙ")
        logger.info("-" * 40)
        
        models = self.model_trainer.create_models()
        all_results = []
        best_overall_roc_auc = -float('inf')
        best_overall_info = None
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Модель: {model_name}")
            logger.info(f"{'='*40}")
            
            best_model, cv_score, best_params = self.model_trainer.train_with_cv(
                model, X_train, y_train, model_name
            )
            
            metrics, y_pred, y_pred_proba = self.evaluator.evaluate(best_model, X_test, y_test)
            
            logger.info("\n  Результаты на тестовой выборке:")
            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1: {metrics['f1']:.4f}")
            logger.info(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            
            version = self.registry.register_model(
                best_model, model_name, metrics, best_params
            )
            
            all_results.append({
                'model': model_name,
                'version': version,
                'metrics': metrics,
                'cv_score': cv_score,
                'best_params': best_params
            })
            
            if metrics['roc_auc'] > best_overall_roc_auc:
                best_overall_roc_auc = metrics['roc_auc']
                best_overall_info = {
                    'model': model_name,
                    'version': version,
                    'metrics': metrics,
                    'best_params': best_params,
                    'model_obj': best_model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'feature_names': self.feature_cols
                }
        
        logger.info("\n ИНТЕРПРЕТАЦИЯ ЛУЧШЕЙ МОДЕЛИ")
        logger.info("-" * 40)
        
        if best_overall_info:
            logger.info(f"Лучшая модель: {best_overall_info['model']} v{best_overall_info['version']}")
            logger.info(f"  ROC-AUC = {best_overall_info['metrics']['roc_auc']:.4f}")
            logger.info(f"  F1 = {best_overall_info['metrics']['f1']:.4f}")
            logger.info(f"  Recall = {best_overall_info['metrics']['recall']:.4f}")
            logger.info(f"  Precision = {best_overall_info['metrics']['precision']:.4f}")
            
            if best_overall_info['best_params']:
                logger.info(f"  Лучшие параметры: {best_overall_info['best_params']}")
            
            if best_overall_info['model'] == 'DecisionTree':
                self.interpreter.explain_decision_tree(
                    best_overall_info['model_obj'], 
                    best_overall_info['feature_names'], 
                    max_depth=4
                )
            elif best_overall_info['model'] == 'RandomForest':
                self.interpreter.explain_random_forest(
                    best_overall_info['model_obj'],
                    best_overall_info['feature_names']
                )
            else:
                self.interpreter.explain_shap_any(
                    best_overall_info['model_obj'],
                    best_overall_info['X_test'],
                    best_overall_info['feature_names']
                )
            
            y_pred_best = best_overall_info['model_obj'].predict(best_overall_info['X_test'])
            cm = confusion_matrix(best_overall_info['y_test'], y_pred_best)
            logger.info("\n  Матрица ошибок:")
            logger.info(f"    TN={cm[0,0]}, FP={cm[0,1]}")
            logger.info(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        logger.info("\n МОНИТОРИНГ MODEL DRIFT")
        logger.info("-" * 40)
        
        if best_overall_info:
            X_new = best_overall_info['X_test'][:50]
            y_new = best_overall_info['y_test'][:50]
            
            drift_detected, current_metrics = self.evaluator.detect_drift(
                best_overall_info['model_obj'], 
                X_new, y_new, 
                best_overall_info['metrics'],
                threshold=self.config['drift_threshold']
            )
            
            logger.info("\n  Текущие метрики на новых данных:")
            logger.info(f"    ROC-AUC: {current_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"    Recall: {current_metrics.get('recall', 0):.4f}")
            logger.info(f"  Дрейф обнаружен: {'ДА' if any(drift_detected.values()) else 'НЕТ'}")
        
        
        logger.info("\n МОДЕЛЬНЫЙ МЕНЕДЖМЕНТ И СЕРИАЛИЗАЦИЯ")
        logger.info("-" * 40)
        
        # Package and maintain best performing model
        if best_overall_info:
            # Evaluate maintenance metrics
            maintenance_metrics = self.model_maintenance.evaluate_model_performance(
                best_overall_info['model_obj'], 
                best_overall_info['model'],
                best_overall_info['X_test'], 
                best_overall_info['y_test']
            )
            
            # Package the model with maintenance data
            self.model_maintenance.package_and_register_model(
                best_overall_info['model_obj'],
                best_overall_info['model'],
                best_overall_info['metrics'],
                best_overall_info['feature_names'],
                self.preprocessor  # Include preprocessing pipeline
            )
            
            # Set as current best model
            self.best_model = best_overall_info['model_obj']
            self.best_params = best_overall_info['best_params']
            self.model_performance = best_overall_info['metrics']
            
            logger.info(f"Лучшая модель упакована и зарегистрирована: {best_overall_info['model']}")
            logger.info("  Сохранена в реестре моделей")
            
        self._print_storage_info()
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            logger.info("\nСводка результатов:")
            logger.info("-" * 90)
            logger.info(f"{'Модель':<20} | {'ROC-AUC':<8} | {'Recall':<8} | {'Precision':<8} | {'F1':<8} | {'Версия'}")
            logger.info("-" * 90)
            for _, row in results_df.iterrows():
                logger.info(f"{row['model']:<20} | {row['metrics']['roc_auc']:.4f}   | "
                          f"{row['metrics']['recall']:.4f} | {row['metrics']['precision']:.4f} | "
                          f"{row['metrics']['f1']:.4f} | v{row['version']}")
        
        return pd.DataFrame(all_results) if all_results else None, self.registry
    
    def incremental_update(self, new_data_path: str = None, X_new: np.ndarray = None, 
                           y_new: np.ndarray = None, model_name: str = None):
        logger.info("\n" + "=" * 60)
        logger.info("ДООБУЧЕНИЕ МОДЕЛИ")
        logger.info("=" * 60)
        
        if new_data_path:
            logger.info(f"Загрузка новых данных из {new_data_path}...")
            df_new = pd.read_csv(new_data_path)
            X_new, y_new, _, _ = self.preprocessor.preprocess(df_new, fit_scaler=False)
            
        elif X_new is not None and y_new is not None:
            logger.info(f"Используются переданные данные: {len(X_new)} записей")
        else:
            logger.error("Не указаны новые данные")
            return None
        
        if model_name:
            model, model_info = self.registry.get_latest_model(model_name)
            if model is None:
                logger.error(f"Модель {model_name} не найдена")
                return None
        else:
            best_auc = -float('inf')
            best_model = None
            best_model_name = None
            best_model_info = None
            
            for name in self.config['models'].keys():
                model, info = self.registry.get_latest_model(name)
                if model and info and info.get('metrics', {}).get('roc_auc', -1) > best_auc:
                    best_auc = info['metrics']['roc_auc']
                    best_model = model
                    best_model_name = name
                    best_model_info = info
            
            if best_model is None:
                logger.error("Не найдено обученных моделей")
                return None
            
            model = best_model
            model_name = best_model_name
            logger.info(f"Выбрана лучшая модель: {model_name} (ROC-AUC={best_auc:.4f})")
        
        logger.info("\nОценка модели до дообучения...")
        metrics_before, _, _ = self.evaluator.evaluate(model, X_new, y_new)
        logger.info(f"  ROC-AUC (до): {metrics_before['roc_auc']:.4f}")
        logger.info(f"  Recall (до): {metrics_before['recall']:.4f}")
        
        logger.info("\nВыполнение дообучения...")
        updated_model = self.model_trainer.incremental_train(model, X_new, y_new, model_name)
        
        if updated_model is None:
            logger.error("Incremental training failed - returning original model")
            return model, metrics_before
        
        logger.info("\nОценка модели после дообучения...")
        metrics_after, _, _ = self.evaluator.evaluate(updated_model, X_new, y_new)
        logger.info(f"  ROC-AUC (после): {metrics_after['roc_auc']:.4f}")
        logger.info(f"  Recall (после): {metrics_after['recall']:.4f}")
        
        roc_improvement = metrics_after['roc_auc'] - metrics_before['roc_auc']
        recall_improvement = metrics_after['recall'] - metrics_before['recall']
        
        logger.info("\nИзменения:")
        logger.info(f"  ROC-AUC: {roc_improvement:+.4f}")
        logger.info(f"  Recall: {recall_improvement:+.4f}")
        
        # 7. Сохранение дообученной модели
        version = self.registry.register_model(
            updated_model, 
            model_name, 
            metrics_after,
            {'parent_version': model_info.get('version', 0)},
            is_incremental=True
        )
        
        logger.info(f"\nДообученная модель сохранена как v{version}")
        
        return updated_model, metrics_after
    
    def _print_storage_info(self):
        files = [f for f in os.listdir(self.registry.storage_path) if f.endswith('.joblib')]
        total_size = sum(os.path.getsize(os.path.join(self.registry.storage_path, f)) 
                        for f in files) / (1024 * 1024)
        
        logger.info("\nХранилище моделей:")
        logger.info(f"  Директория: {self.registry.storage_path}")
        logger.info(f"  Моделей сохранено: {len(files)}")
        logger.info(f"  Общий размер: {total_size:.2f} МБ")


def main():
    pipeline = ModelPipeline("model_config.yaml")
    
    results, registry = pipeline.run()

    # pipeline.incremental_update(new_data_path="../new_raw_data.csv")



if __name__ == "__main__":
    main()
