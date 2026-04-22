## Документация к модулю "data_collection.py"

Модуль реализует потоковую загрузку данных из CSV-файлов с поддержкой батчевой обработки и симуляции дрифта. Обеспечивает хранение батчей и расчёт метаданных.

---

### Основные классы

#### 1. **Класс `DataStream`**

Генератор потока данных из CSV-файлов с разбивкой на батчи.

```python
def __init__(self, sources, batch_size=1000, delay=0.01, drift_config=None):
    self.sources = sources
    self.batch_size = batch_size
    self.delay = delay
    self.batch_counter = 0
    self.drift_config = drift_config or {}
```

**Параметры:**
- **`sources`**: Список путей к CSV-файлам для загрузки.
- **`batch_size`**: Размер батча (количество строк). По умолчанию `1000`.
- **`delay`**: Задержка между батчами в секундах. По умолчанию `0.01`.
- **`drift_config`**: Конфигурация для симуляции дрифта (dict).

---

#### **Метод `stream()`**

```python
def stream(self):
    for file_path in self.sources:
        df = pd.read_csv(file_path)
        total_rows = len(df)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_rows)
            
            chunk = df.iloc[start_idx:end_idx].copy()
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
            time.sleep(self.delay)
            
            yield chunk
```

**Описание:**
- Итерирует по всем файлам из `sources`.
- Разбивает каждый файл на батчи размером `batch_size`.
- Применяет симуляцию дрифта к батчам (если включено).
- Добавляет метаданные батча в атрибут `batch_info`.
- Возвращает `DataFrame` через `yield` (генератор).

**Возвращает**: Генератор батчей (`pd.DataFrame`) с метаданными.

---

#### **Метод `_apply_drift_simulation(batch_df, batch_number)`**

```python
def _apply_drift_simulation(self, batch_df, batch_number):
```

Применяет различные типы дрифта к данным начиная с определённого батча.

**Типы дрифта:**

##### **Concept Drift** (`drift_type='concept'`)
- Изменяет целевую переменную `CLAIM_PAID`.
- Случайным образом заменяет `drift_strength * n_rows` значений на `1`.
- Имитирует изменение концепции (больше страховых случаев).

##### **Covariate Drift** (`drift_type='covariate'`)
- Сдвигает числовые признаки на `drift_strength * std`.
- Применяется к первым 3 числовым колонкам из whitelist.
- Имитирует изменение распределения входных данных.

##### **Anomaly Drift** (`drift_type='anomaly'`)
- Внедряет выбросы в числовые признаки.
- Количество аномалий: `drift_strength * n_rows`.
- Значение аномалии: `mean ± 5 * std`.

**Параметры из конфигурации:**
- **`enabled`**: Включить/выключить симуляцию дрифта.
- **`start_drift_batch`**: Номер батча, с которого начинается дрифт.
- **`drift_type`**: Тип дрифта (`concept`, `covariate`, `anomaly`).
- **`drift_strength`**: Сила дрифта (0.0 - 1.0).
- **`drift_whitelist`**: Список колонок, к которым применяется дрифт.

---

### 2. **Класс `FileStorage`**

Управление сохранением и загрузкой батчей на диск.

```python
def __init__(self, output_dir="raw_data"):
    self.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
```

**Параметры:**
- **`output_dir`**: Директория для сохранения батчей. По умолчанию `raw_data`.

---

#### **Метод `save_batch(batch, batch_info)`**

```python
def save_batch(self, batch: pd.DataFrame, batch_info: Dict[str, Any]):
    source_name = os.path.basename(batch_info['source']).replace('.csv', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{source_name}_batch{batch_info['batch_num']:04d}_{timestamp}.csv"
    
    filepath = os.path.join(self.output_dir, filename)
    batch.to_csv(filepath, index=False)
    
    return filepath
```

**Описание:**
- Формирует имя файла на основе источника и номера батча.
- Сохраняет батч в CSV-файл.
- Формат имени: `{источник}_batch{номер}_{timestamp}.csv`.

**Возвращает**: Полный путь к сохранённому файлу.

---

#### **Методы `get_all_files()` и `load_batch(filename)`**

```python
def get_all_files(self):
    return [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]

def load_batch(self, filename):
    filepath = os.path.join(self.output_dir, filename)
    return pd.read_csv(filepath)
```

- **`get_all_files()`**: Возвращает список всех CSV-файлов в директории.
- **`load_batch(filename)`**: Загружает батч из файла.

---

### 3. **Класс `MetadataCalculator`**

Вычисляет статистику и метаданные для каждого батча.

```python
def __init__(self):
    logger.info("Калькулятор метапараметров инициализирован")
```

---

#### **Метод `calculate(batch, batch_num)`**

```python
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
            col_stats = {
                'column_name': col,
                'data_type': "numeric",
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'min_value': col_data.min(),
                'max_value': col_data.max(),
                'mean_value': col_data.mean()
            }
        else:
            col_stats = {
                'column_name': col,
                'data_type': "categorical",
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'min_value': None,
                'max_value': None,
                'mean_value': None
            }
        
        metadata['column_stats'].append(col_stats)
    
    return metadata
```

**Описание:**
- Вычисляет общую статистику батча: количество строк, колонок, качество данных.
- Для каждой колонки собирает:
  - **Числовые**: min, max, mean, количество пропусков, уникальных значений.
  - **Категориальные**: количество пропусков, уникальных значений.

**Возвращает**: Словарь с метаданными батча.

---

#### **Метод `print_summary(metadata)`**

```python
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
```

- Выводит сводку по батчу: количество записей, колонок, качество данных.
- Показывает топ-5 колонок с наибольшим количеством пропусков.

---

### 4. **Класс `DataCollection`**

Главный класс, объединяющий все компоненты сбора данных.

```python
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
```

**Описание:**
- Загружает конфигурацию через `get_config()`.
- Инициализирует хранилище, поток данных и калькулятор метаданных.

---

#### **Метод `run()`**

```python
def run(self):
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
                
            except Exception as e:
                error_count += 1
                logger.error(f"Ошибка при обработке батча: {e}")
    
    except KeyboardInterrupt:
        logger.info("Остановка пользователем")
    
    finally:
        self._print_final_stats(processed_batches, error_count)
```

**Описание:**
- Итерирует по всем батчам из потока.
- Сохраняет каждый батч в хранилище.
- Вычисляет метаданные для каждого батча.
- Каждые 10 батчей выводит сводку и информацию о хранилище.
- Обрабатывает ошибки и прерывания пользователя.

---

### Итоговая структура работы

1. **Инициализация** компонентов из конфигурации.
2. **Потоковая загрузка** данных из CSV-файлов.
3. **Разбивка на батчи** с заданным размером.
4. **Симуляция дрифта** (опционально) для тестирования.
5. **Сохранение батчей** в файловое хранилище.
6. **Расчёт метаданных** для каждого батча.
7. **Логирование** прогресса и ошибок.
