## Документация к модулю "model_maintenance.py"

Модуль реализует обслуживание моделей машинного обучения: мониторинг производительности, упаковку моделей, выбор лучшей модели и управление жизненным циклом моделей.

---

### Основные классы

#### 1. **Класс `PerformanceMonitor`**

Мониторинг производительности моделей.

```python
def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.metrics_history = []
    self.performance_thresholds = config.get('performance_thresholds', {})
```

---

**Метод `measure_inference_time(model, X_test)`**:

```python
def measure_inference_time(self, model, X_test):
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time
    
    return inference_time, len(X_test)
```

**Описание:**
- Измеряет время выполнения предсказаний.
- Возвращает время инференса и количество обработанных образцов.

---

**Метод `measure_memory_usage()`**:

```python
def measure_memory_usage(self):
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }
```

**Описание:**
- Измеряет использование памяти процессом через `psutil`.
- Возвращает RSS и VMS в мегабайтах.

---

**Метод `check_performance_thresholds(metrics)`**:

```python
def check_performance_thresholds(self, metrics: Dict[str, Any]) -> bool:
```

**Описание:**
- Проверяет соответствие метрик пороговым значениям из конфигурации.

**Проверяемые метрики:**
- **`accuracy`**: Должна быть >= порога.
- **`f1`**: Должна быть >= порога.
- **`inference_time`**: Должно быть <= порога.

**Возвращает**: `True` если все пороги соблюдены, `False` если есть нарушения.

---

**Метод `record_metrics(model_name, metrics)`**:

```python
def record_metrics(self, model_name: str, metrics: Dict[str, Any]):
```

**Описание:**
- Сохраняет метрики модели в историю с временной меткой.

---

**Метод `get_performance_trend(model_name, metric)`**:

```python
def get_performance_trend(self, model_name: str, metric: str) -> List[float]:
```

**Описание:**
- Возвращает историю значений указанной метрики для модели.

---

#### 2. **Класс `ModelPackager`**

Упаковка и загрузка моделей с метаданными.

```python
def __init__(self, registry_path: str):
    self.registry_path = registry_path
    os.makedirs(registry_path, exist_ok=True)
```

---

**Метод `package_model(model, model_name, metrics, feature_names, preprocessing_pipeline)`**:

```python
def package_model(self, model, model_name: str, metrics: Dict[str, Any], 
                 feature_names: Optional[List[str]] = None,
                 preprocessing_pipeline = None):
```

**Описание:**
- Создаёт пакет модели с метаданными.

**Структура пакета:**
```
model_registry/
└── {model_name}_{timestamp}/
    ├── {model_name}_{timestamp}.joblib      # Модель
    ├── preprocessor_{timestamp}.joblib      # Препроцессор (опционально)
    └── metadata.json                        # Метаданные
```

**Метаданные:**
- `model_name`, `timestamp`, `model_file`
- `metrics`: метрики производительности
- `feature_names`: список признаков
- `python_version`: версия Python
- `requirements`: версии пакетов (pandas, numpy, scikit-learn, tensorflow)

**Возвращает**: Путь к директории пакета.

---

**Метод `load_package(package_dir)`**:

```python
def load_package(self, package_dir: str):
```

**Описание:**
- Загружает модель и препроцессор из пакета.
- Читает метаданные из `metadata.json`.

**Возвращает**: `(model, metadata, preprocessor)`

---

**Метод `_get_requirements_info()`**:

```python
def _get_requirements_info(self):
```

**Описание:**
- Собирает версии ключевых библиотек через `importlib.metadata`.
- Пакеты: pandas, numpy, scikit-learn, tensorflow.

---

#### 3. **Класс `ModelSelector`**

Выбор лучшей модели на основе производительности и характеристик данных.

```python
def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.selection_rules = config.get('selection_rules', {})
```

---

**Метод `select_best_model(model_performance, data_characteristics)`**:

```python
def select_best_model(self, model_performance: Dict[str, Any], 
                     data_characteristics: Dict[str, Any]) -> str:
```

**Описание:**
- Выбирает лучшую модель на основе комбинированного скора.

**Формула:**
```
final_score = performance_weight * performance_score + adaptation_weight * adaptation_score
```

**Параметры:**
- **`performance_weight`**: Вес метрик производительности (по умолчанию 0.7).
- **`adaptation_weight`**: Вес адаптации к данным (по умолчанию 0.3).

**Возвращает**: Имя лучшей модели.

---

**Метод `_calculate_performance_score(metrics)`**:

```python
def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
```

**Описание:**
- Вычисляет взвешенный скор производительности.

**Веса метрик:**
- `accuracy`: 0.3
- `f1`: 0.4
- `roc_auc`: 0.3

---

**Метод `_calculate_adaptation_score(model_name, data_characteristics)`**:

```python
def _calculate_adaptation_score(self, model_name: str, 
                              data_characteristics: Dict[str, Any]) -> float:
```

**Описание:**
- Вычисляет скор адаптации модели к характеристикам данных.

**Правила адаптации:**

| Модель | Разреженные данные | Аномалии | Числовые признаки | Категориальные |
|--------|-------------------|----------|-------------------|----------------|
| DecisionTree | 0.8 | 0.6 | 0.7 | 0.9 |
| RandomForest | 0.7 | 0.8 | 0.8 | 0.8 |
| NeuralNetwork | 0.5 | 0.4 | 0.9 | 0.6 |

---

#### 4. **Класс `ModelMaintenance`**

Главный класс для управления жизненным циклом моделей.

```python
def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.performance_monitor = PerformanceMonitor(config)
    self.model_packager = ModelPackager(config.get('model_registry_path', 'model_registry'))
    self.model_selector = ModelSelector(config)
    self.current_best_model = None
    self.best_model_metadata = None
```

---

**Метод `evaluate_model_performance(model, model_name, X_test, y_test)`**:

```python
def evaluate_model_performance(self, model, model_name: str, X_test, y_test):
```

**Описание:**
- Комплексная оценка производительности модели.

**Метрики:**
- **`accuracy`**: Точность классификации.
- **`f1`**: F1-score (weighted).
- **`roc_auc`**: ROC-AUC score (если доступны вероятности).
- **`inference_time`**: Время инференса.
- **`samples_per_second`**: Скорость обработки.
- **`memory_rss_mb`**: Использование памяти (RSS).
- **`memory_vms_mb`**: Использование памяти (VMS).
- **`meets_thresholds`**: Соответствие пороговым значениям.

**Возвращает**: Словарь с метриками.

---

**Метод `package_and_register_model(model, model_name, metrics, feature_names, preprocessing_pipeline)`**:

```python
def package_and_register_model(self, model, model_name: str, metrics: Dict[str, Any],
                             feature_names: List[str], preprocessing_pipeline = None):
```

**Описание:**
- Упаковывает модель в пакет.
- Обновляет лучшую модель, если новая модель лучше.
- Сохраняет информацию о лучшей модели в `best_model.json`.

**Возвращает**: Путь к пакету модели.

---

**Метод `_should_update_best_model(model_name, metrics)`**:

```python
def _should_update_best_model(self, model_name: str, metrics: Dict[str, Any]) -> bool:
```

**Описание:**
- Определяет, следует ли обновить лучшую модель.

**Условия обновления:**
1. Если лучшей модели ещё нет → обновить.
2. Если F1-score улучшился более чем на `improvement_threshold` (по умолчанию 0.01) → обновить.
3. Если новая модель не соответствует порогам, а текущая соответствует → не обновлять.

**Возвращает**: `True` если нужно обновить, `False` иначе.

---

**Метод `select_model_for_prediction(data_characteristics)`**:

```python
def select_model_for_prediction(self, data_characteristics: Dict[str, Any]) -> str:
```

**Описание:**
- Выбирает оптимальную модель для предсказания на основе характеристик данных.
- Усредняет метрики по последним 10 записям из истории.
- Использует `ModelSelector` для выбора.

**Возвращает**: Имя выбранной модели.

---

### Итоговая структура

1. **Мониторинг производительности**: время инференса, использование памяти, проверка порогов.
2. **Упаковка моделей**: сериализация модели, препроцессора и метаданных.
3. **Выбор лучшей модели**: на основе метрик производительности и адаптации к данным.
4. **Управление жизненным циклом**: автоматическое обновление лучшей модели при улучшении метрик.
5. **Версионирование**: все модели сохраняются с временными метками.
