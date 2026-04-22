## Документация к модулю "drift_detector.py"

Модуль реализует детекцию дрифта данных с использованием статистических тестов. Обнаруживает ковариатный дрифт (изменение распределения признаков), концептуальный дрифт (изменение целевой переменной) и дрифт качества данных.

---

### Класс `DataDriftDetector`

Основной класс для детекции дрифта.

#### **`__init__(config, reference_data)`**

```python
def __init__(self, config: Dict[str, Any], reference_data: pd.DataFrame = None):
    self.config = config
    self.reference_data = reference_data
    self.drift_threshold = config.get('drift_threshold', 0.01)
    self.min_effect_size = config.get('min_effect_size', 0.1)
    self.drift_history = []
    
    self.monitored_fields = [
        'CLAIM_PAID', 'INSURED_VALUE', 'PREMIUM', 
        'PROD_YEAR', 'SEATS_NUM', 'SEX', 'INSR_TYPE'
    ]
```

**Параметры:**
- **`config`**: Конфигурация детектора (пороги, параметры).
- **`reference_data`**: Референсные данные для сравнения (опционально).

**Атрибуты:**
- **`drift_threshold`**: Уровень значимости для статистических тестов (по умолчанию 0.01).
- **`min_effect_size`**: Минимальный размер эффекта для практической значимости (по умолчанию 0.1).
- **`monitored_fields`**: Список колонок для мониторинга дрифта (whitelist).
- **`drift_history`**: История обнаруженных дрифтов.

---

#### **`_initialize_reference_stats(reference_data)`**

```python
def _initialize_reference_stats(self, reference_data: pd.DataFrame):
```

**Описание:**
- Вычисляет статистики референсных данных для мониторируемых колонок.

**Для числовых колонок:**
- `mean`, `std`, `min`, `max`, `n_samples`

**Для категориальных колонок:**
- `value_distribution` (нормализованное распределение).
- `n_categories`, `n_samples`

---

### Методы детекции дрифта

#### 1. **`detect_drift(current_data, batch_info)`**

```python
def detect_drift(self, current_data: pd.DataFrame, batch_info: Dict[str, Any]) -> Dict[str, Any]:
```

**Описание:**
- Главный метод детекции дрифта между референсными и текущими данными.

**Этапы:**
1. Проверка дрифта признаков (`_detect_feature_drift`).
2. Проверка концептуального дрифта (`_detect_concept_drift`).
3. Проверка дрифта качества данных (`_detect_quality_drift`).
4. Сохранение результатов в историю.
5. Запись отчёта в файл.

**Возвращает**: Словарь с результатами:
```python
{
    'drift_detected': bool,
    'drift_type': str,  # 'covariate', 'concept', None
    'confidence': float,
    'affected_features': list,
    'batch_info': dict,
    'timestamp': str
}
```

---

#### 2. **`_detect_feature_drift(current_data)`**

```python
def _detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
```

**Описание:**
- Обнаруживает дрифт в распределениях признаков (ковариатный дрифт).

**Для числовых признаков:**
- Использует IQR-метод (межквартильный размах).
- Сравнивает квартили Q1 и Q3 между референсными и текущими данными.
- Вычисляет IQR-similarity: `1 - (изменение_Q1 + изменение_Q3)`.
- Дрифт обнаруживается, если `iqr_similarity < 0.8` (более 20% изменения).

**Для категориальных признаков:**
- Использует Chi-squared тест (`scipy.stats.chisquare`).
- Сравнивает распределения категорий.
- Дрифт обнаруживается, если `p_value < drift_threshold`.

**Возвращает**:
```python
{
    'drift_detected': bool,
    'affected_features': [
        {
            'feature': str,
            'drift_type': str,  # 'robust_distribution' или 'categorical_distribution'
            'confidence': float,
            'test': str,
            'q1_change_percent': float,  # для числовых
            'q3_change_percent': float,  # для числовых
            'p_value': float  # для категориальных
        }
    ],
    'confidence': float  # средняя уверенность
}
```

---

#### 3. **`_detect_concept_drift(current_data)`**

```python
def _detect_concept_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
```

**Описание:**
- Обнаруживает концептуальный дрифт (изменение распределения целевой переменной `CLAIM_PAID`).

**Метод:**
- Сравнивает распределения значений целевой переменной.
- Вычисляет `distribution_change` как сумму абсолютных разностей вероятностей.
- Дрифт обнаруживается, если изменение > 10%.

**Возвращает**:
```python
{
    'drift_detected': bool,
    'confidence': float,
    'distribution_change': float  # от 0.0 до 2.0
}
```

---

#### 4. **`_detect_quality_drift(current_data)`**

```python
def _detect_quality_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
```

**Описание:**
- Обнаруживает дрифт в метриках качества данных.

**Проверки:**

##### **Missing Value Drift**
- Сравнивает процент пропущенных значений.
- Дрифт обнаруживается, если изменение > 5%.

##### **Outlier Drift**
- Использует IQR-метод для подсчёта выбросов.
- Дрифт обнаруживается, если изменение > 50%.

**Возвращает**:
```python
{
    'quality_issues': bool,
    'quality_metrics': {
        'missing_value_drift': float,
        'outlier_drift': float
    },
    'anomalies': [
        {
            'type': str,
            'severity': float,
            'description': str
        }
    ]
}
```

---

#### 5. **`_detect_outliers(data)`**

```python
def _detect_outliers(self, data: pd.DataFrame) -> int:
```

**Описание:**
- Подсчитывает выбросы в числовых колонках с помощью IQR-метода.

**Метод:**
- Для каждой числовой колонки:
  - `Q1 = 25-й перцентиль`
  - `Q3 = 75-й перцентиль`
  - `IQR = Q3 - Q1`
  - `lower_bound = Q1 - 1.5 * IQR`
  - `upper_bound = Q3 + 1.5 * IQR`
  - Выбросы: значения вне `[lower_bound, upper_bound]`.

**Возвращает**: Общее количество выбросов во всех числовых колонках.

---

### Вспомогательные методы

#### **`_save_drift_report(drift_results, batch_info)`**

```python
def _save_drift_report(self, drift_results: Dict[str, Any], batch_info: Dict[str, Any]):
```

**Описание:**
- Сохраняет отчёт о детекции дрифта в JSON-файл.
- Путь: `{artifacts_dir}/drift_report_batch{batch_num:04d}.json`.

---

#### **`get_drift_summary()`**

```python
def get_drift_summary(self) -> Dict[str, Any]:
```

**Описание:**
- Возвращает сводку по истории детекции дрифта.

**Возвращает**:
```python
{
    'total_batches_analyzed': int,
    'batches_with_drift': int,
    'drift_rate': float,
    'last_drift_batch': int,
    'drift_types': list,
    'most_affected_features': list  # топ-5 признаков
}
```

---

#### **`_get_most_affected_features()`**

```python
def _get_most_affected_features(self) -> List[str]:
```

**Описание:**
- Возвращает топ-5 признаков, наиболее часто подверженных дрифту.
- Подсчитывает частоту появления каждого признака в `affected_features` из истории.

---

### Функция `detect_data_drift()`

```python
def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                     config: Dict[str, Any] = None) -> Dict[str, Any]:
```

**Описание:**
- Удобная функция для одноразовой детекции дрифта.
- Создаёт экземпляр `DataDriftDetector` и выполняет детекцию.

**Параметры:**
- **`reference_data`**: Референсные данные.
- **`current_data`**: Текущие данные для проверки.
- **`config`**: Конфигурация (опционально).

**Возвращает**: Результаты детекции дрифта.

---

### Итоговая структура

1. **Инициализация** референсных статистик из первого батча.
2. **Детекция ковариатного дрифта** через IQR-метод и Chi-squared тест.
3. **Детекция концептуального дрифта** через сравнение распределений целевой переменной.
4. **Детекция дрифта качества** через мониторинг пропусков и выбросов.
5. **Сохранение отчётов** в JSON-файлы.
6. **Ведение истории** для анализа трендов дрифта.
