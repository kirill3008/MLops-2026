## Документация к модулю "config.py"

Модуль обеспечивает централизованное управление конфигурацией MLOps пайплайна. Все параметры загружаются из единого YAML-файла `unified_config.yaml`.

---

### Класс `Config`

#### **`__init__(config_file)`**
```python
def __init__(self, config_file: str = "unified_config.yaml"):
    self.config_file = config_file
    self._config_data: Optional[Dict[str, Any]] = None
    self._load_config()
```
- **`config_file`**: Путь к YAML-файлу конфигурации. По умолчанию `unified_config.yaml`.
- Автоматически загружает конфигурацию при создании экземпляра.

---

#### **`_load_config()`**
```python
def _load_config(self):
    if not os.path.exists(self.config_file):
        raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
    
    with open(self.config_file, 'r', encoding='utf-8') as f:
        self._config_data = yaml.safe_load(f)
```
- Проверяет наличие конфигурационного файла.
- Загружает данные через `yaml.safe_load()`.
- Выбрасывает `FileNotFoundError`, если файл не найден.

---

### Свойства для доступа к секциям

#### **`data_collection`**
```python
@property
def data_collection(self) -> Dict[str, Any]:
    return self._config_data.get('data_collection', {})
```
- Возвращает настройки сбора данных (источники, размер батча, задержка).

#### **`data_analysis`**
```python
@property
def data_analysis(self) -> Dict[str, Any]:
    return self._config_data.get('data_analysis', {})
```
- Возвращает параметры анализа качества данных и детекции дрифта.

#### **`model_training`**
```python
@property
def model_training(self) -> Dict[str, Any]:
    return self._config_data.get('model_training', {})
```
- Возвращает настройки обучения моделей (CV-фолды, гиперпараметры).

#### **`model_registry`**
```python
@property
def model_registry(self) -> Dict[str, Any]:
    return self._config_data.get('model_registry', {})
```
- Возвращает параметры реестра моделей (путь хранения, формат сериализации).

---

### Дополнительные методы

#### **`get_section(section_name, default)`**
```python
def get_section(self, section_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return self._config_data.get(section_name, default or {})
```
- Возвращает произвольную секцию конфигурации.
- Если секция не найдена, возвращает `default`.

#### **`get_nested(*keys, default)`**
```python
def get_nested(self, *keys: str, default: Any = None) -> Any:
    current = self._config_data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
```
- Безопасный доступ к вложенным параметрам.
- Пример: `config.get_nested('data_analysis', 'dq', 'parsing', 'date_format')`

#### **`reload()`**
```python
def reload(self):
    self._load_config()
```
- Перезагружает конфигурацию из файла.

---

### Глобальный экземпляр

#### **`get_config(config_file)`**
```python
def get_config(config_file: str = "unified_config.yaml") -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance
```
- Возвращает глобальный экземпляр конфигурации (паттерн Singleton).
- При первом вызове создается экземпляр, при последующих — возвращается существующий.

#### **`reload_config()`**
```python
def reload_config():
    global _config_instance
    if _config_instance:
        _config_instance.reload()
```
- Перезагружает глобальную конфигурацию.

---

### Примеры использования

```python
from config import get_config

# Получить конфигурацию
config = get_config()

# Доступ к секциям
batch_size = config.data_collection.get('batch_size')
cv_folds = config.model_training.get('cv_folds')

# Вложенные параметры
date_format = config.get_nested('data_analysis', 'dq', 'parsing', 'date_format')
```
