import pandas as pd
import logging
import time
from datetime import datetime
import os
from typing import Dict, Any
import sys
import numpy as np

from config import get_config

def setup_logging(log_level="INFO", log_file="data_collection.log"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class DataStream:
    
    def __init__(self, sources, batch_size=1000, delay=0.01, drift_config=None):
        self.sources = sources
        self.batch_size = batch_size
        self.delay = delay
        self.batch_counter = 0
        self.drift_config = drift_config or {}
        
        logger.info(f"Инициализирован поток с {len(self.sources)} источниками")
        logger.info(f"Параметры: batch_size={batch_size}, delay={delay}s")
        if self.drift_config.get('enabled', False):
            logger.info(f"Drift simulation: {self.drift_config.get('drift_type', 'none')} starting at batch {self.drift_config.get('start_drift_batch', 0)}")
    
    def stream(self):
        
        for file_path in self.sources:

            if not os.path.exists(file_path):
                logger.error(f"Файл не найден: {file_path}")
                continue
            
            logger.info(f"Начинаем чтение источника: {file_path}")
            
            try:
                
                df = pd.read_csv(file_path)
                total_rows = len(df)
                num_batches = (total_rows + self.batch_size - 1) // self.batch_size
                
                for batch_num in range(num_batches):

                    start_idx = batch_num * self.batch_size
                    end_idx = min(start_idx + self.batch_size, total_rows)
                    
                    chunk = df.iloc[start_idx:end_idx].copy()
                    
                    # Apply drift simulation if enabled and it's time
                    chunk = self._apply_drift_simulation(chunk, self.batch_counter + 1)
                    
                    chunk.attrs['batch_info'] = {
                        'source': file_path,
                        'batch_num': self.batch_counter + 1,
                        'batch_index': batch_num + 1,
                        'total_batches': num_batches,
                        'start_row': start_idx,
                        'end_row': end_idx,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.batch_counter += 1
                    
                    logger.debug(f"Батч #{self.batch_counter} из {file_path}: "
                               f"строки {start_idx}-{end_idx} ({len(chunk)} записей)")
                    
                    if self.drift_config.get('enabled', False) and self.batch_counter >= self.drift_config.get('start_drift_batch', 0):
                        logger.info(f"Дрифт применён к батчу #{self.batch_counter}")
                    
                    time.sleep(self.delay)
                    
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Ошибка при чтении {file_path}: {e}")
                continue
        
        logger.info(f"Поток завершен. Всего батчей: {self.batch_counter}")
    
    def _apply_drift_simulation(self, batch_df, batch_number):
        """Apply controlled drift to batch data (only on whitelisted fields)"""
        if not self.drift_config.get('enabled', False):
            return batch_df
        
        start_drift_batch = self.drift_config.get('start_drift_batch', 0)
        if batch_number < start_drift_batch:
            return batch_df
        
        drift_type = self.drift_config.get('drift_type', 'concept')
        drift_strength = self.drift_config.get('drift_strength', 0.3)
        whitelist = self.drift_config.get('drift_whitelist', [])
        
        batch_copy = batch_df.copy()
        
        # Only process columns that are in the whitelist
        available_cols = [col for col in whitelist if col in batch_copy.columns]
        
        if not available_cols:
            return batch_copy
        
        if drift_type == 'concept':
            # Simulate concept drift by changing target variable distribution
            if 'CLAIM_PAID' in available_cols:
                # Introduce concept drift: make claims more frequent
                n_rows = len(batch_copy)
                n_to_change = int(n_rows * drift_strength)
                
                # Randomly select rows to change
                change_indices = np.random.choice(n_rows, n_to_change, replace=False)
                # In insurance context: make more claims happen
                batch_copy.loc[change_indices, 'CLAIM_PAID'] = 1
                
        elif drift_type == 'covariate':
            # Simulate covariate drift by shifting feature distributions
            numeric_cols = [col for col in available_cols 
                          if batch_copy[col].dtype in [np.number] and col != 'CLAIM_PAID']
            
            if numeric_cols:
                # Shift numerical features
                for col in numeric_cols[:3]:  # Shift first 3 numeric columns
                    if batch_copy[col].dtype in [np.number]:
                        shift_amount = drift_strength * batch_copy[col].std()
                        batch_copy[col] += shift_amount
        
        elif drift_type == 'anomaly':
            # Simulate anomaly drift by introducing outliers
            n_rows = len(batch_copy)
            n_anomalies = int(n_rows * drift_strength)
            
            # Introduce extreme values in numerical columns (excluding target)
            numeric_cols = [col for col in available_cols 
                          if batch_copy[col].dtype in [np.number] and col != 'CLAIM_PAID']
            
            if numeric_cols:
                for i in range(n_anomalies):
                    row_idx = np.random.randint(0, n_rows)
                    col_idx = np.random.choice(range(len(numeric_cols)))
                    col = numeric_cols[col_idx]
                    
                    # Set extreme value (5 standard deviations away)
                    mean_val = batch_copy[col].mean()
                    std_val = batch_copy[col].std()
                    anomaly_val = mean_val + 5 * std_val * (1 if np.random.rand() > 0.5 else -1)
                    batch_copy.at[row_idx, col] = anomaly_val
        
        return batch_copy


class FileStorage:

    def __init__(self, output_dir="raw_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Хранилище инициализировано: {output_dir}")

    def save_batch(self, batch: pd.DataFrame, batch_info: Dict[str, Any]):

        source_name = os.path.basename(batch_info['source']).replace('.csv', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{source_name}_batch{batch_info['batch_num']:04d}_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        batch.to_csv(filepath, index=False)

        logger.debug(f"Батч #{batch_info['batch_num']} сохранен: {filepath}")
        return filepath

    def get_all_files(self):
        return [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]

    def load_batch(self, filename):
        filepath = os.path.join(self.output_dir, filename)
        return pd.read_csv(filepath)


class MetadataCalculator:
    
    def __init__(self):
        logger.info("Калькулятор метапараметров инициализирован")
    
    def calculate(self, batch: pd.DataFrame, batch_num: int):

        metadata = {
            'batch_num': batch_num,
            'total_rows': len(batch),
            'columns_count': len(batch.columns),
            'data_quality': batch.count().sum() / (len(batch) * len(batch.columns)),
            'column_stats': []
        }
        
        for col in batch.columns:
            col_data = batch[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "numeric"
                min_val = col_data.min() if not col_data.isnull().all() else None
                max_val = col_data.max() if not col_data.isnull().all() else None
                mean_val = col_data.mean() if not col_data.isnull().all() else None
            else:
                data_type = "categorical"
                min_val = None
                max_val = None
                mean_val = None
            
            col_stats = {
                'column_name': col,
                'data_type': data_type,
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'min_value': min_val,
                'max_value': max_val,
                'mean_value': mean_val
            }
            
            metadata['column_stats'].append(col_stats)
        
        logger.debug(f"Батч #{batch_num}: качество данных = {metadata['data_quality']:.2%}")
        
        return metadata
    
    def print_summary(self, metadata):
        print(f"\nСтатистика батча #{metadata['batch_num']}:")
        print(f"   Записей: {metadata['total_rows']}")
        print(f"   Колонок: {metadata['columns_count']}")
        print(f"   Качество данных: {metadata['data_quality']:.2%}")
        print("\n   Топ-5 колонок по пропускам:")
        
        sorted_stats = sorted(metadata['column_stats'], 
                             key=lambda x: x['null_count'], reverse=True)
        
        for stat in sorted_stats[:5]:
            print(f"     - {stat['column_name']}: {stat['null_count']} пропусков")


class DataCollection:

    def __init__(self):
        config = get_config()
        self.config = config.data_collection

        self.storage = FileStorage(self.config['output_dir'])
        self.stream = DataStream(
            self.config['sources'],
            self.config['batch_size'],
            self.config['delay'],
            self.config.get('drift_simulation', {})
        )
        self.calculator = MetadataCalculator()

        logger.info("Сбор готов к работе")

    def run(self):
        logger.info("=" * 50)
        logger.info("ЗАПУСК ПАЙПЛАЙНА ОБРАБОТКИ")
        logger.info("=" * 50)

        processed_batches = 0
        error_count = 0

        try:
            for batch in self.stream.stream():
                try:
                    batch_info = batch.attrs['batch_info']

                    filepath = self.storage.save_batch(batch, batch_info)
                    metadata = self.calculator.calculate(batch, batch_info['batch_num'])

                    if batch_info['batch_num'] % 10 == 0:
                        self.calculator.print_summary(metadata)
                        self._print_storage_info()

                    processed_batches += 1
                    logger.info(f"Батч #{batch_info['batch_num']} сохранен: {os.path.basename(filepath)}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"Ошибка при обработке батча: {e}")

        except KeyboardInterrupt:
            logger.info("Остановка пользователем")

        finally:
            self._print_final_stats(processed_batches, error_count)

    def _print_storage_info(self):
        files = self.storage.get_all_files()
        total_size = sum(os.path.getsize(os.path.join(self.storage.output_dir, f))
                         for f in files) / (1024 * 1024)

        print(f"\nХранилище: {self.storage.output_dir}")
        print(f"   Файлов: {len(files)}")
        print(f"   Размер: {total_size:.2f} МБ")

    def _print_final_stats(self, processed_batches, error_count):
        print("\n" + "=" * 50)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 50)
        print(f"Обработано батчей: {processed_batches}")
        print(f"Ошибок: {error_count}")

        files = self.storage.get_all_files()
        total_size = sum(os.path.getsize(os.path.join(self.storage.output_dir, f))
                         for f in files) / (1024 * 1024)

        print("\nФайловое хранилище:")
        print(f"   Директория: {self.storage.output_dir}")
        print(f"   Всего файлов: {len(files)}")
        print(f"   Общий размер: {total_size:.2f} МБ")
        print("=" * 50)

def main():
    pipeline = DataCollection()
    pipeline.run()


if __name__ == "__main__":
    main()
