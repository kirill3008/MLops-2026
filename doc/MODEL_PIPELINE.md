## Документация к модулю "model_pipeline.py"

Модуль реализует пайплайн обучения, оценки и управления моделями машинного обучения. Включает загрузку данных, предобработку, обучение с Grid Search, инкрементальное обучение и реестр моделей.

---

### Основные классы

#### 1. **Класс `DataLoader`**

Загружает данные из CSV-файлов.

```python
def __init__(self, data_folder: str):
    self.data_folder = data_folder
```

**Метод `load_all_files()`**:
- Находит все CSV-файлы в директории.
- Загружает файлы с фиксированными именами колонок.
- Объединяет все файлы в единый DataFrame.

**Колонки**: `SEX`, `INSR_BEGIN`, `INSR_END`, `INSR_TYPE`, `INSURED_VALUE`, `PREMIUM`, `OBJECT_ID`, `PROD_YEAR`, `SEATS_NUM`, `CARRYING_CAPACITY`, `TYPE_VEHICLE`, `CCM_TON`, `MAKE`, `USAGE`, `CLAIM_PAID`.

---

#### 2. **Класс `DataPreprocessor`**

Выполняет предобработку данных для обучения моделей.

```python
def __init__(self):
    self.scaler = StandardScaler()
    self.is_fitted = False
```

---

**Метод `preprocess(df, fit_scaler)`**:

```python
def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True):
```

**Этапы:**
1. Обработка пропусков (`_handle_missing_values`).
2. Кодирование категориальных переменных (`_handle_categorical_variables`).
3. Обработка числовых переменных и создание признаков (`_handle_numeric_variables`).
4. Создание целевой переменной `HAS_CLAIM = (CLAIM_PAID > 0)`.
5. Нормализация признаков через `StandardScaler`.

**Возвращает**: `(X_scaled, y, feature_cols, label_encoders)`

---

**Метод `_handle_missing_values(df)`**:

**Импутация числовых колонок:**
- `INSURED_VALUE`, `CLAIM_PAID` → заполняются `0`.
- Остальные числовые → заполняются медианой.

**Импутация категориальных колонок:**
- Заполняются модой (наиболее частым значением).
- Если мода отсутствует → заполняются `'Unknown'`.

---

**Метод `_handle_categorical_variables(df)`**:

- Кодирует категориальные переменные через `LabelEncoder`.
- Создаёт новые колонки с суффиксом `_encoded`.
- Категориальные колонки: `SEX`, `INSR_TYPE`, `TYPE_VEHICLE`, `MAKE`, `USAGE`.

**Возвращает**: `(df_processed, label_encoders)`

---

**Метод `_handle_numeric_variables(df)`**:

**Создание признаков:**
- `POLICY_DURATION = INSR_END - INSR_BEGIN` (в днях).

**Удаление колонок:**
- `INSR_BEGIN`, `INSR_END`, `OBJECT_ID`, `MAKE`, `TYPE_VEHICLE`, `USAGE`, `SEX`, `INSR_TYPE`, `EFFECTIVE_YR`.

---

#### 3. **Класс `ModelTrainer`**

Обучает модели с Grid Search и поддерживает инкрементальное обучение.

```python
def __init__(self, config: Dict):
    self.config = config
    self.models_config = config['models']
    self.use_class_weight = config.get('use_class_weight', True)
```

---

**Метод `create_models()`**:

Создаёт набор моделей для обучения:

1. **DecisionTreeClassifier**:
   - `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`.
   - `class_weight='balanced'` (для борьбы с дисбалансом классов).

2. **RandomForestClassifier**:
   - `n_estimators=100`, `max_depth=10`.
   - `class_weight='balanced'`, `warm_start=True` (для инкрементального обучения).

3. **MLPClassifier** (нейронная сеть):
   - `hidden_layer_sizes=(100, 50)`, `max_iter=500`.
   - `early_stopping=True`, `warm_start=True`.

**Возвращает**: Словарь с моделями.

---

**Метод `train_with_cv(model, X_train, y_train, model_name)`**:

```python
def train_with_cv(self, model, X_train, y_train, model_name: str):
```

**Описание:**
- Обучает модель с использованием Grid Search CV.
- Использует параметры из конфигурации (`models_config`).
- Метрика оптимизации: `roc_auc`.

**Возвращает**: `(best_estimator, best_score, best_params)`

---

**Метод `incremental_train(model, X_new, y_new, model_name)`**:

```python
def incremental_train(self, model, X_new, y_new, model_name: str):
```

**Описание:**
- Выполняет инкрементальное обучение на новых данных.
- Использует `partial_fit()` (если доступно) или `warm_start`.
- Если модель не поддерживает инкрементальное обучение, переобучает с нуля.

**Возвращает**: Обновлённую модель.

---

#### 4. **Класс `ModelRegistry`**

Управляет хранением и версионированием моделей.

```python
def __init__(self, storage_path: str = './model_registry'):
    self.storage_path = storage_path
    self.metadata_file = os.path.join(storage_path, 'models_metadata.json')
    self.metadata = self._load_metadata()
```

---

**Метод `register_model(model, model_name, metrics, best_params, is_incremental)`**:

```python
def register_model(self, model, model_name: str, metrics: Dict, 
                  best_params: Dict = None, is_incremental: bool = False) -> int:
```

**Описание:**
- Сохраняет модель в файл с именем `{model_name}_v{version}.joblib`.
- Регистрирует метаданные: версия, метрики, параметры, временная метка.
- Обновляет файл `models_metadata.json`.

**Возвращает**: Номер версии модели.

---

**Метод `get_best_model(model_name, metric)`**:

```python
def get_best_model(self, model_name: str, metric: str = 'roc_auc'):
```

**Описание:**
- Находит лучшую версию модели по заданной метрике.
- Загружает модель из файла через `joblib.load()`.

**Возвращает**: `(model, metadata)`

---

**Метод `get_latest_model(model_name)`**:

```python
def get_latest_model(self, model_name: str):
```

**Описание:**
- Загружает последнюю по времени версию модели.

**Возвращает**: `(model, metadata)`

---

#### 5. **Класс `ModelEvaluator`**

Оценивает качество моделей на тестовых данных.

```python
def evaluate(self, model, X_test, y_test):
```

**Метрики:**
- **`accuracy`**: Точность классификации.
- **`precision`**: Precision (weighted).
- **`recall`**: Recall (weighted).
- **`f1`**: F1-score (weighted).
- **`roc_auc`**: ROC-AUC score.
- **`confusion_matrix`**: Матрица ошибок.

**Возвращает**: Словарь с метриками.

---

#### 6. **Класс `ModelPipeline`**

Главный класс, объединяющий все этапы пайплайна.

```python
def __init__(self):
    self.config = get_config().model_training
    self.data_loader = DataLoader(self.config['data_folder'])
    self.preprocessor = DataPreprocessor()
    self.trainer = ModelTrainer(self.config)
    self.evaluator = ModelEvaluator()
    self.registry = ModelRegistry(get_config().model_registry['path'])
    self.model_maintenance = ModelMaintenance()
```

---

**Метод `run()`**:

```python
def run(self):
```

**Этапы:**
1. Загрузка данных через `DataLoader`.
2. Предобработка данных через `DataPreprocessor`.
3. Разбивка на train/test (80/20).
4. Создание моделей через `ModelTrainer.create_models()`.
5. Обучение каждой модели с Grid Search CV.
6. Оценка на тестовой выборке.
7. Регистрация моделей в `ModelRegistry`.
8. Выбор лучшей модели по ROC-AUC.
9. Упаковка и регистрация лучшей модели через `ModelMaintenance`.

**Возвращает**: `(results_df, best_model_info)`
- `results_df`: DataFrame с метриками всех моделей.
- `best_model_info`: Информация о лучшей модели.

---

**Метод `incremental_update(X_new, y_new, model_name)`**:

```python
def incremental_update(self, X_new, y_new, model_name: str = 'RandomForest'):
```

**Описание:**
- Загружает последнюю версию модели из реестра.
- Выполняет инкрементальное обучение на новых данных.
- Регистрирует обновлённую модель с новыми метриками.

**Возвращает**: `(updated_model, metrics)`

---

### Итоговая структура пайплайна

1. **Загрузка данных** из CSV-файлов.
2. **Предобработка**: импутация, кодирование, нормализация.
3. **Обучение моделей** с Grid Search CV.
4. **Оценка моделей** на тестовой выборке.
5. **Регистрация моделей** с версионированием.
6. **Выбор лучшей модели** по метрике ROC-AUC.
7. **Инкрементальное обучение** для поддержки онлайн-обучения.
8. **Упаковка и деплой** через `ModelMaintenance`.
