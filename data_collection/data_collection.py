import pandas as pd
import json
import logging
import time
import yaml
from datetime import datetime
import os
from typing import Generator, Dict, Any
import sys

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


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Конфигурация загружена из {config_path}")
    else:
        config = {
            "batch_size": 1000,
            "delay": 0.01,
            "output_dir": "raw_data",
            "sources": [
                "insurance/motor_data11-14lats.csv",
                "insurance/motor_data14-2018.csv"
            ]
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Создан файл конфигурации: {config_path}")

    return config

class DataStream:
    
    def __init__(self, sources, batch_size=1000, delay=0.01):
        self.sources = sources
        self.batch_size = batch_size
        self.delay = delay
        self.batch_counter = 0
        
        logger.info(f"Инициализирован поток с {len(self.sources)} источниками")
        logger.info(f"Параметры: batch_size={batch_size}, delay={delay}s")
    
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
                    
                    time.sleep(self.delay)
                    
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Ошибка при чтении {file_path}: {e}")
                continue
        
        logger.info(f"Поток завершен. Всего батчей: {self.batch_counter}")


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
        print(f"\n   Топ-5 колонок по пропускам:")
        
        sorted_stats = sorted(metadata['column_stats'], 
                             key=lambda x: x['null_count'], reverse=True)
        
        for stat in sorted_stats[:5]:
            print(f"     - {stat['column_name']}: {stat['null_count']} пропусков")


class DataCollection:

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)

        self.storage = FileStorage(self.config['output_dir'])
        self.stream = DataStream(
            self.config['sources'],
            self.config['batch_size'],
            self.config['delay']
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

        print(f"\nФайловое хранилище:")
        print(f"   Директория: {self.storage.output_dir}")
        print(f"   Всего файлов: {len(files)}")
        print(f"   Общий размер: {total_size:.2f} МБ")
        print("=" * 50)

def main():

    pipeline = DataCollection("config.yaml")

    pipeline.run()


if __name__ == "__main__":
    main()