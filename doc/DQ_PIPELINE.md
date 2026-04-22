## Документация к модулю "dq_pipeline.py"

Модуль реализует пайплайн контроля качества данных (Data Quality) с использованием правил консистентности, алгоритма Apriori и детекции дрифта. Обеспечивает парсинг, валидацию, очистку данных и мониторинг изменений.

---

### Основные функции

#### 1. **`parse_types(df, cfg)`**

```python
def parse_types(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
```

**Описание:**
- Парсит типы данных на основе конфигурации.
- Преобразует колонки дат в формат `datetime`.
- Приводит числовые колонки к типу `numeric`.

**Параметры:**
- **`df`**: Входной DataFrame.
- **`cfg`**: Конфигурация с параметрами парсинга.

**Колонки для парсинга:**
- **Даты**: `INSR_BEGIN`, `INSR_END` (формат из конфигурации).
- **Числовые**: `INSR_TYPE`, `SEX`, `INSURED_VALUE`, `PREMIUM`, `PROD_YEAR`, `SEATS_NUM`, `CLAIM_PAID`.

**Возвращает**: DataFrame с преобразованными типами.

---

#### 2. **`compute_dq_metrics(df, cfg)`**

```python
def compute_dq_metrics(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
```

**Описание:**
- Вычисляет метрики качества данных.

**Метрики:**
- **`missing_total_ratio`**: Общий процент пропусков.
- **`missing_per_column`**: Процент пропусков по каждой колонке.
- **`duplicate_ratio`**: Процент дубликатов.
- **`bad_time_ratio`**: Процент некорректных временных меток.
- **`bad_date_ratio`**: Процент некорректных дат (неправильный порядок `INSR_BEGIN` > `INSR_END`).
- **`invalid_ratio`**: Максимальный процент невалидных значений.
- **`invalid_breakdown`**: Детализация по типам невалидности:
  - `sex_unknown`: неизвестные значения пола.
  - `insr_type_unknown`: неизвестные типы страховки.
  - `prod_year_out_of_range`: год выпуска вне допустимого диапазона.
  - `seats_out_of_range`: количество мест вне допустимого диапазона.
  - `{col}_negative`: отрицательные значения в неотрицательных колонках.

**Возвращает**: Словарь с метриками качества данных.

---

#### 3. **`quality_flags(metrics, cfg)`**

```python
def quality_flags(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
```

**Описание:**
- Сравнивает метрики с пороговыми значениями из конфигурации.

**Флаги:**
- **`too_many_missing`**: Превышен лимит пропусков.
- **`too_many_duplicates`**: Превышен лимит дубликатов.
- **`too_many_bad_dates`**: Превышен лимит некорректных дат.
- **`too_many_invalid`**: Превышен лимит невалидных значений.
- **`any_issue`**: Хотя бы один флаг поднят.

**Возвращает**: Словарь с флагами качества.

---

#### 4. **`clean_batch(df, cfg)`**

```python
def clean_batch(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
```

**Описание:**
- Очищает данные на основе конфигурации.

**Операции очистки:**
1. **Удаление дубликатов** (`drop_duplicates`).
2. **Удаление строк с неправильным порядком дат** (`drop_bad_date_order`).
3. **Удаление строк с пропусками во временных метках** (`drop_missing_time`).
4. **Приведение значений вне диапазона к NaN** (`out_of_range_to_nan`):
   - `PROD_YEAR`: проверка на `prod_year_min` и `prod_year_max`.
   - `SEATS_NUM`: проверка на `seats_min` и `seats_max`.
5. **Приведение отрицательных значений к NaN** (`negative_to_nan`).
6. **Заполнение нулями** для специфичных колонок:
   - `CLAIM_PAID`, `INSURED_VALUE`, `CARRYING_CAPACITY`, `CCM_TON`, `SEATS_NUM`.
7. **Импутация числовых колонок** медианой или средним (`impute_numeric`).
8. **Импутация категориальных колонок** значением "Unknown" (`impute_categorical`).

**Возвращает**: Очищенный DataFrame.

---

#### 5. **`binary_transactions(df, cfg)`**

```python
def binary_transactions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
```

**Описание:**
- Преобразует данные в бинарный формат для алгоритма Apriori.

**Создаваемые признаки:**
- **`premium_gt_0`**: премия больше 0.
- **`premium_high`**: премия в верхнем квартиле.
- **`insured_value_high`**: страховая сумма в верхнем квартиле.
- **`seats_ge_5`**: количество мест >= 5.
- **`duration_ge_365`**: срок страховки >= 365 дней.
- **`{COLUMN}={VALUE}`**: бинарные признаки для топ-K категориальных значений (для `INSR_TYPE`, `TYPE_VEHICLE`, `USAGE`, `MAKE`, `SEX`).
- **`claim_paid_pos`**: наличие выплаты по страховому случаю (опционально).

**Возвращает**: DataFrame с бинарными признаками (0/1).

---

#### 6. **`mine_rules(df, cfg)`**

```python
def mine_rules(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
```

**Описание:**
- Выполняет майнинг ассоциативных правил с помощью алгоритма Apriori.

**Этапы:**
1. Преобразование данных в бинарный формат (`binary_transactions`).
2. Поиск частых наборов элементов (`apriori` из `mlxtend`).
3. Генерация правил (`association_rules`).
4. Фильтрация по `min_lift` из конфигурации.
5. Сортировка по `lift`, `confidence`, `support`.
6. Выбор топ-K правил (`top_k_rules`).

**Параметры из конфигурации:**
- **`min_support`**: Минимальная поддержка.
- **`min_confidence`**: Минимальная уверенность.
- **`min_lift`**: Минимальный лифт.
- **`top_k_rules`**: Количество возвращаемых правил.

**Возвращает**: DataFrame с правилами (столбцы: `support`, `confidence`, `lift`, `antecedents_str`, `consequents_str`).

---

#### 7. **`build_reference_rules_if_missing(parsed_df, cfg)`**

```python
def build_reference_rules_if_missing(parsed_df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Path]:
```

**Описание:**
- Создаёт референсные правила консистентности, если файл не существует.
- Использует первый батч данных для генерации правил.

**Возвращает**: Путь к файлу с правилами или `None`.

---

#### 8. **`evaluate_reference_rules_on_batch(parsed_df, cfg, batch_info)`**

```python
def evaluate_reference_rules_on_batch(parsed_df: pd.DataFrame, cfg: Dict[str, Any], batch_info: Dict[str, Any] = None) -> Dict[str, Any]:
```

**Описание:**
- Проверяет батч данных на соответствие референсным правилам консистентности.
- Обнаруживает дрифт с помощью `DataDriftDetector`.

**Этапы:**
1. Загрузка референсных правил из файла.
2. Преобразование батча в бинарный формат.
3. Для каждого правила:
   - Подсчёт поддержки антецедента и конъюнкции.
   - Вычисление новой поддержки и уверенности.
   - Проверка флагов:
     - **`flag_conf_drop`**: падение уверенности ниже `confidence_drop_ratio * confidence_ref`.
     - **`flag_support_drop`**: падение поддержки ниже `support_drop_ratio * support_ref`.
     - **`flag_conf_abs`**: абсолютная уверенность ниже `min_confidence_abs`.
     - **`low_sample`**: антецедент встречается реже `min_antecedent_count`.
4. Детекция дрифта через `DataDriftDetector`.
5. Сохранение отчётов в `artifacts/dq/` и `artifacts/rules/`.

**Возвращает**: Словарь с результатами проверки консистентности и дрифта:
```python
{
    'enabled': bool,
    'n_rows': int,
    'n_rules_checked': int,
    'any_issue': bool,
    'rules': list,
    'drift_analysis': dict,
    'drift_detected': bool,
    'drift_type': str,
    'drift_confidence': float
}
```

---

#### 9. **`analyze_batch_file(batch_path, config_path)`**

```python
def analyze_batch_file(batch_path: str | Path, config_path: str | Path) -> Dict[str, Any]:
```

**Описание:**
- Полный анализ батча: парсинг, валидация, очистка, майнинг правил, проверка консистентности.

**Этапы:**
1. Загрузка конфигурации.
2. Чтение батча из CSV.
3. Парсинг типов данных.
4. Построение референсных правил (если отсутствуют).
5. Проверка консистентности.
6. Вычисление метрик до очистки.
7. Майнинг правил Apriori.
8. Очистка данных.
9. Вычисление метрик после очистки.
10. Сохранение артефактов:
    - `{batch_id}_dq.json` — метрики качества.
    - `{batch_id}_apriori_rules.csv` — ассоциативные правила.
    - `{batch_id}.parquet` или `.csv` — очищенные данные.
    - `consistency_{batch_id}.json` — отчёт о консистентности.

**Возвращает**: Словарь с путями к сохранённым файлам и флагами качества.

---

### Итоговая структура

1. **Парсинг типов данных** (`parse_types`).
2. **Вычисление метрик качества** (`compute_dq_metrics`).
3. **Проверка пороговых значений** (`quality_flags`).
4. **Очистка данных** (`clean_batch`).
5. **Майнинг ассоциативных правил** (`mine_rules`).
6. **Проверка консистентности** (`evaluate_reference_rules_on_batch`).
7. **Детекция дрифта** через `DataDriftDetector`.
8. **Сохранение артефактов** в JSON/CSV/Parquet форматах.
