## Описание поставленной задачи

### Источник данных

**Датасет**: Ethiopian Motor Vehicle Insurance Data  
**Источники**:
- `motor_data11-14lats.csv` (данные 2011-2014 гг.)
- `motor_data14-2018.csv` (данные 2014-2018 гг.)

**Характеристики данных**:
- **Объём**: ~40,000+ записей
- **Признаков**: 15 колонок
- **Тип задачи**: Бинарная классификация
- **Временной диапазон**: 2011-2018 гг.

**Структура данных**:
```
- SEX                 # Пол страхователя (категориальный)
- INSR_BEGIN         # Дата начала страховки
- INSR_END           # Дата окончания страховки
- INSR_TYPE          # Тип страховки (категориальный)
- INSURED_VALUE      # Страховая сумма (числовой)
- PREMIUM            # Премия (числовой)
- OBJECT_ID          # ID объекта
- PROD_YEAR          # Год производства автомобиля
- SEATS_NUM          # Количество мест
- CARRYING_CAPACITY  # Грузоподъёмность
- TYPE_VEHICLE       # Тип транспортного средства (категориальный)
- CCM_TON            # Объём двигателя/тоннаж
- MAKE               # Марка автомобиля (категориальный)
- USAGE              # Назначение использования (категориальный)
- CLAIM_PAID         # Сумма выплаты по страховому случаю (целевая переменная)
```

---

### Целевая переменная

**Прогнозируемая величина**: Наличие страховой выплаты  
**Формализация**: `HAS_CLAIM = (CLAIM_PAID > 0)`

**Распределение классов**:
- Класс 0 (нет выплаты): ~95% данных
- Класс 1 (есть выплата): ~5% данных

**Тип задачи**: Бинарная классификация с дисбалансом классов.

---

## Data Quality и валидация

### Контроль качества данных

#### 1. **Метрики качества**
```python
Оценка пропусков:
- missing_total_ratio: общий процент пропусков
- missing_per_column: процент пропусков по колонкам

Оценка дубликатов:
- duplicate_ratio: процент дублирующихся строк

Оценка валидности:
- bad_date_ratio: некорректные даты (INSR_BEGIN > INSR_END)
- invalid_ratio: значения вне допустимых диапазонов
```

#### 2. **Правила валидации**

**Числовые признаки**:
- `PROD_YEAR`: диапазон 1980-2020
- `SEATS_NUM`: диапазон 1-50
- Неотрицательные: `INSURED_VALUE`, `PREMIUM`, `CARRYING_CAPACITY`, `CCM_TON`, `CLAIM_PAID`

**Категориальные признаки**:
- `SEX`: допустимые значения из конфигурации
- `INSR_TYPE`: допустимые значения из конфигурации

**Пороговые значения** (из `unified_config.yaml`):
```yaml
thresholds:
  max_missing_total: 0.3      # Макс. 30% пропусков
  max_duplicate_ratio: 0.1    # Макс. 10% дубликатов
  max_bad_date_ratio: 0.05    # Макс. 5% некорректных дат
  max_invalid_ratio: 0.1      # Макс. 10% невалидных значений
```

#### 3. **Правила консистентности (Apriori)**

**Алгоритм**: Apriori для майнинга ассоциативных правил

**Параметры**:
- `min_support`: 0.01 (минимальная поддержка)
- `min_confidence`: 0.5 (минимальная уверенность)
- `min_lift`: 1.2 (минимальный лифт)
- `top_k_rules`: 5 (количество правил)

**Проверки**:
- `flag_conf_drop`: падение уверенности > 30%
- `flag_support_drop`: падение поддержки > 50%
- `flag_conf_abs`: абсолютная уверенность < 0.3

#### 4. **Детекция дрифта**

**Типы дрифта**:
- **Feature drift**: изменение распределения признаков (KS-test, Chi-squared)
- **Concept drift**: изменение распределения целевой переменной
- **Quality drift**: изменение метрик качества (пропуски, выбросы)

**Мониторируемые признаки**:
```python
monitored_fields = [
    'CLAIM_PAID',      # Целевая переменная
    'INSURED_VALUE',   # Страховая сумма
    'PREMIUM',         # Премия
    'PROD_YEAR',       # Год производства
    'SEATS_NUM',       # Количество мест
    'SEX',             # Пол (категориальный)
    'INSR_TYPE'        # Тип страховки (категориальный)
]
```

**Пороги детекции**:
- `drift_threshold`: 0.05 (уровень значимости для тестов)
- `min_effect_size`: 0.1 (минимальный практический размер эффекта)

---

### Сохранение артефактов

#### Структура артефактов

```
artifacts/
├── dq/                              # Метрики качества данных
│   └── batch_XXX_dq.json            # DQ-метрики до/после очистки
├── rules/                           # Правила консистентности
│   ├── reference_rules.csv          # Референсные правила Apriori
│   ├── consistency_batch_XXX.json   # Результаты проверки консистентности
│   └── batch_XXX_apriori_rules.csv  # Правила для конкретного батча
└── drift_report_batchXXXX.json      # Отчёты о дрифте

analyzed_data/                       # Очищенные данные
└── batch_XXX.parquet                # Данные после DQ-обработки

model_registry/                      # Реестр моделей
├── models_metadata.json             # Метаданные всех версий моделей
├── best_model.json                  # Информация о лучшей модели
└── {ModelName}_{timestamp}/         # Пакет модели
    ├── {ModelName}_{timestamp}.joblib
    ├── preprocessor_{timestamp}.joblib
    └── metadata.json

pipeline_performance_YYYYMMDD_HHMMSS.csv   # Метрики производительности по батчам
summary_report_YYYYMMDD_HHMMSS.json        # Агрегированные отчёты
summary_report_YYYYMMDD_HHMMSS_dashboard.png  # Визуальные дашборды
```

#### Формат артефактов

**DQ-метрики** (`batch_XXX_dq.json`):
```json
{
  "before": {
    "n_rows": 5000,
    "missing_total_ratio": 0.15,
    "duplicate_ratio": 0.05,
    "invalid_ratio": 0.08
  },
  "after": {
    "n_rows": 4750,
    "missing_total_ratio": 0.0,
    "duplicate_ratio": 0.0,
    "invalid_ratio": 0.0
  },
  "flags_after": {
    "any_issue": false
  }
}
```

**Метаданные модели** (`metadata.json`):
```json
{
  "model_name": "RandomForest",
  "timestamp": "20260323_143022",
  "metrics": {
    "accuracy": 0.92,
    "f1": 0.78,
    "roc_auc": 0.89
  },
  "feature_names": [...],
  "python_version": "3.10",
  "requirements": {
    "scikit-learn": "1.3.0",
    "pandas": "2.0.0"
  }
}
```

---

## Проектирование модели

### Обработка признакового пространства

#### 1. **Обработка пропусков**

**Числовые признаки**:
- `INSURED_VALUE`, `CLAIM_PAID` → заполнение `0`
- Остальные числовые → заполнение медианой

**Категориальные признаки**:
- Заполнение модой (наиболее частым значением)
- Если мода отсутствует → `'Unknown'`

#### 2. **Кодирование категориальных переменных**

**Метод**: Label Encoding через `sklearn.preprocessing.LabelEncoder`

**Признаки**:
```python
categorical_cols = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
```

**Результат**: Создание новых колонок с суффиксом `_encoded`

#### 3. **Инженерия признаков**

**Создаваемые признаки**:
```python
POLICY_DURATION = (INSR_END - INSR_BEGIN).days  # Срок действия полиса в днях
HAS_CLAIM = (CLAIM_PAID > 0).astype(int)        # Бинарная целевая переменная
```

**Удаляемые признаки**:
- Исходные временные метки: `INSR_BEGIN`, `INSR_END`
- ID-колонки: `OBJECT_ID`
- Исходные категориальные (после кодирования): `MAKE`, `TYPE_VEHICLE`, `USAGE`, `SEX`, `INSR_TYPE`
- Некачественные: `EFFECTIVE_YR` (содержит мусор)

#### 4. **Нормализация**

**Метод**: `StandardScaler` из scikit-learn

**Применение**:
```python
X_scaled = scaler.fit_transform(X)  # При обучении
X_scaled = scaler.transform(X)      # При инференсе
```

**Результат**: Все числовые признаки приведены к стандартному виду (mean=0, std=1)

#### 5. **Обработка дисбаланса классов**

**Метод**: Взвешивание классов (`class_weight='balanced'`)

**Применение**: Для Decision Tree и Random Forest

**Формула веса**:
```
weight[class] = n_samples / (n_classes * n_samples[class])
```

---

### Рассматриваемые модели

#### 1. **Decision Tree Classifier**

**Пространство гиперпараметров** (GridSearchCV):
```python
param_grid = {
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced']
}
```

**Фиксированные параметры**:
- `random_state`: 42 (воспроизводимость)
- `criterion`: 'gini' (критерий разбиения)

**Применение**: Базовая интерпретируемая модель для сравнения.

---

#### 2. **Random Forest Classifier**

**Пространство гиперпараметров** (GridSearchCV):
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced']
}
```

**Фиксированные параметры**:
- `random_state`: 42
- `n_jobs`: -1 (параллелизация)
- `warm_start`: True (для инкрементального обучения)

**Применение**: Основная модель для продакшена (баланс качества и скорости).

---

#### 3. **Neural Network Classifier (MLP)**

**Пространство гиперпараметров** (GridSearchCV):
```python
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (150, 75)],
    'alpha': [0.0001, 0.001, 0.01],              # L2-регуляризация
    'learning_rate_init': [0.001, 0.01]
}
```

**Фиксированные параметры**:
- `random_state`: 42
- `max_iter`: 500
- `early_stopping`: True (остановка при переобучении)
- `validation_fraction`: 0.1
- `warm_start`: True (для инкрементального обучения)
- `activation`: 'relu'
- `solver`: 'adam'

**Применение**: Детекция сложных нелинейных паттернов.

---

### Стратегия обучения

#### 1. **Разбиение данных**

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

**Пропорции**:
- Train: 80%
- Test: 20%

#### 2. **Кросс-валидация**

**Метод**: GridSearchCV с k-fold CV

**Параметры**:
- `cv`: 3 фолда (из конфигурации)
- `scoring`: 'roc_auc' (метрика оптимизации)
- `n_jobs`: -1 (параллелизация)

**Процесс**:
1. GridSearch перебирает комбинации гиперпараметров.
2. Для каждой комбинации выполняется 3-fold CV.
3. Выбирается комбинация с лучшим ROC-AUC.

#### 3. **Инкрементальное обучение**

**Поддержка**:
- Random Forest: через `warm_start=True` (добавление деревьев)
- Neural Network: через `partial_fit()` и `warm_start=True`
- Decision Tree: переобучение с нуля

**Применение**: Режим `update` и `pipeline` при обработке новых батчей.

---

### Метрики оценки

#### Основные метрики

```python
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted'),
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}
```

#### Критерии выбора лучшей модели

**Приоритет метрик**:
1. **ROC-AUC** (30%) — устойчивость к дисбалансу классов
2. **F1-score** (40%) — баланс precision и recall
3. **Accuracy** (30%) — общая точность

**Формула финального скора**:
```
final_score = 0.3 * accuracy + 0.4 * f1 + 0.3 * roc_auc
```

**Пороговые значения** (из конфигурации):
```yaml
performance_thresholds:
  accuracy: 0.7
  f1: 0.6
  recall: 0.6
```

---

### Мониторинг производительности

**Отслеживаемые метрики**:
- Время инференса (секунды)
- Пропускная способность (samples/second)
- Использование памяти (RSS, VMS в МБ)
- Соответствие пороговым значениям

**Адаптивный выбор модели**:
- Учёт характеристик данных (разреженность, аномалии)
- Комбинированный скор: производительность + адаптация
- Автоматическое переключение при деградации метрик
