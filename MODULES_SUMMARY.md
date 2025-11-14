# 📦 New Modules & Features Summary

## ✅ Что добавлено

### 1️⃣ Модуль сбора данных (`data_collectors.py`)

#### **WebScraperCollector** - Парсинг сайтов
```python
scraper = WebScraperCollector(llm_provider)

# Скрапим несколько URL
data = await scraper.scrape_urls(
    urls=["https://example.com/page1", "https://example.com/page2"],
    selector="article.content"
)

# Или краулим весь сайт
data = await scraper.crawl_website(
    start_url="https://example.com/docs",
    max_pages=100,
    link_pattern="docs\\.example\\.com"
)

# Конвертируем в тренировочные примеры
examples = scraper.convert_to_training_examples(data)
```

**Возможности:**
- ✅ Асинхронный парсинг (быстро!)
- ✅ CSS селекторы для точной выборки
- ✅ Краулинг с фильтрацией ссылок
- ✅ Автоматическая конвертация в Q&A с помощью LLM

---

#### **APICollector** - Интеграция с API
```python
collector = APICollector(api_key="YOUR_TOKEN")

# GitHub issues
issues = await collector.fetch_github_issues(
    repo="fastapi/fastapi",
    state="all",
    max_issues=200
)

# GitHub discussions
discussions = await collector.fetch_github_discussions(
    repo="microsoft/vscode",
    max_discussions=50
)

# StackOverflow
questions = await collector.fetch_stackoverflow_questions(
    tag="python",
    max_questions=100
)

# RSS feeds
entries = await collector.fetch_rss_feed(
    feed_url="https://blog.example.com/feed.xml",
    max_entries=50
)

# Конвертация в примеры
examples = collector.convert_to_training_examples(
    issues,
    data_type='github_issues',
    format_type='chat'
)
```

**Поддерживаемые API:**
- ✅ GitHub Issues
- ✅ GitHub Discussions (GraphQL)
- ✅ StackOverflow Questions
- ✅ RSS/Atom Feeds

---

#### **FileParserCollector** - Парсинг файлов
```python
parser = FileParserCollector(llm_provider)

# Парсинг различных форматов
text = parser.parse_text_file("docs/manual.txt")
data = parser.parse_json_file("data/knowledge.json")
markdown = parser.parse_markdown_file("docs/guide.md")

# Разбивка текста на чанки
chunks = parser.chunk_text(text, chunk_size=1000, overlap=100)

# Конвертация документов в Q&A пары
examples = parser.convert_documents_to_qa_pairs(
    documents=chunks,
    examples_per_doc=3
)
```

**Поддерживаемые форматы:**
- ✅ Plain Text (.txt)
- ✅ JSON (.json)
- ✅ JSONL (.jsonl)
- ✅ Markdown (.md)

---

### 2️⃣ Модуль дедупликации и очистки (`deduplication.py`)

#### **DataDeduplicator** - Удаление дубликатов
```python
deduplicator = DataDeduplicator(similarity_threshold=0.9)

# Метод 1: Точные дубликаты (hash)
unique, stats = deduplicator.deduplicate_dataset(
    examples=examples,
    format_type='chat',
    method='hash'
)

# Метод 2: Нечеткие дубликаты (fuzzy)
unique, stats = deduplicator.deduplicate_dataset(
    examples=examples,
    format_type='chat',
    method='fuzzy',
    similarity_threshold=0.88
)

print(f"Удалено дубликатов: {stats['duplicates_removed']}")
print(f"Уникальных: {stats['unique_count']}")
```

**Методы:**
- ✅ `exact` - Точное совпадение текста
- ✅ `hash` - MD5/SHA256 хеширование (быстрее)
- ✅ `fuzzy` - Нечеткое сравнение (лучше для NLP)

---

#### **DataCleaner** - Очистка и валидация
```python
cleaner = DataCleaner()

# Фильтрация датасета
filtered, stats = cleaner.filter_dataset(
    examples=examples,
    format_type='chat',
    min_length=20,      # Мин. длина контента
    max_length=5000,    # Макс. длина
    remove_toxic=True   # Удалить токсичный контент
)

print(f"Невалидные структуры: {stats['invalid_structure']}")
print(f"Слишком короткие: {stats['too_short']}")
print(f"Слишком длинные: {stats['too_long']}")
print(f"Токсичный контент: {stats['toxic_content']}")
print(f"Валидные: {stats['valid_count']}")
```

**Проверки:**
- ✅ Структура (messages, instruction/output)
- ✅ Длина контента
- ✅ Токсичный контент
- ✅ Кодировка
- ✅ Лишние пробелы

---

#### **DataAnalyzer** - Анализ датасета
```python
analyzer = DataAnalyzer()

# Полный анализ
analysis = analyzer.analyze_dataset(
    examples=examples,
    format_type='chat'
)

print(f"Всего примеров: {analysis['total_examples']}")
print(f"Статистика длины: {analysis['length_stats']}")
print(f"Метаданные: {analysis['metadata_analysis']}")
```

---

### 3️⃣ Система конфигураций (`config_loader.py`)

#### Загрузка конфигов из YAML/JSON
```python
from config_loader import load_pipeline_config

# Загрузка конфигурации
config = load_pipeline_config("configs/my_pipeline.yaml")

print(f"Имя пайплайна: {config.name}")
print(f"LLM Provider: {config.llm_provider.provider}")
print(f"Источников данных: {len(config.data_sources)}")
```

#### Пример конфигурации
```yaml
name: "my_pipeline"
description: "My custom pipeline"

llm_provider:
  provider: "openai"
  model: "gpt-5.1"
  api_key: "${OPENAI_API_KEY}"  # Из env переменных

data_sources:
  - type: "synthetic"
    config:
      domain: "support"
      count: 500

  - type: "web"
    config:
      urls: ["https://example.com"]

deduplication:
  enabled: true
  method: "fuzzy"

cleaning:
  enabled: true
  min_length: 20

output:
  dataset_name: "my_dataset"
  format: "jsonl"
```

**Фичи:**
- ✅ YAML и JSON форматы
- ✅ Env переменные (`${VAR}`)
- ✅ Валидация с Pydantic
- ✅ Шаблоны конфигов

---

### 4️⃣ Версионирование (`versioning.py`)

#### **VersionManager** - Управление версиями
```python
from versioning import version_manager

# Создание версии
version = version_manager.create_version(
    dataset_id=1,
    file_path="data/datasets/my_dataset.jsonl",
    description="Исходная версия"
)

# Список всех версий
versions = version_manager.list_versions(dataset_id=1)

# Откат к версии
version_manager.set_current_version(
    dataset_id=1,
    version_id="abc123"
)

# Сравнение версий
diff = version_manager.compare_versions(
    dataset_id=1,
    version_id_1="abc123",
    version_id_2="def456"
)

print(f"Разница примеров: {diff['differences']['example_count_diff']}")
```

**Возможности:**
- ✅ Автоматическое версионирование
- ✅ Откат к любой версии
- ✅ Сравнение версий
- ✅ История изменений
- ✅ Метаданные для каждой версии

---

### 5️⃣ API Extensions (`api_extensions.py`)

#### Новые эндпоинты (v2)

**Сбор данных:**
- `POST /api/v2/collect/web-scrape` - Парсинг сайтов
- `POST /api/v2/collect/web-crawl` - Краулинг сайтов
- `POST /api/v2/collect/github-issues` - GitHub issues
- `POST /api/v2/collect/stackoverflow` - StackOverflow

**Обработка:**
- `POST /api/v2/process/deduplicate` - Дедупликация
- `POST /api/v2/process/clean` - Очистка

**Версионирование:**
- `POST /api/v2/datasets/{id}/versions` - Создать версию
- `GET /api/v2/datasets/{id}/versions` - Список версий
- `POST /api/v2/datasets/{id}/rollback` - Откат

**Конфигурации:**
- `GET /api/v2/configs` - Список конфигов
- `POST /api/v2/configs/execute` - Запустить пайплайн

---

## 📚 Библиотеки (requirements.txt)

### Новые зависимости:
```txt
# Web Scraping & API Collection
aiohttp==3.9.3          # Async HTTP клиент
beautifulsoup4==4.12.3  # HTML парсинг
feedparser==6.0.11      # RSS/Atom feeds
lxml==5.1.0             # XML/HTML обработка

# Configuration
pyyaml==6.0.1           # YAML конфиги

# Progress
tqdm==4.66.2            # Прогресс бары
```

---

## 🎯 Типичные сценарии использования

### Сценарий 1: Чатбот техподдержки IT-компании
```yaml
# configs/it_support.yaml
data_sources:
  # Синтетика - базовые сценарии
  - type: "synthetic"
    config:
      domain: "support"
      subdomain: "technical"
      count: 500

  # Реальные GitHub issues
  - type: "api"
    config:
      api_type: "github"
      params:
        repo: "your-company/product"
        state: "closed"
        max_issues: 300

  # FAQ со сайта
  - type: "web"
    config:
      urls: ["https://yourcompany.com/faq"]
```

### Сценарий 2: Документация → Q&A
```yaml
# configs/docs_qa.yaml
data_sources:
  - type: "web"
    config:
      start_url: "https://docs.yourproduct.com"
      max_pages: 200
      content_selector: "article.documentation"

llm_provider:
  provider: "openai"
  model: "gpt-5.1"  # Для конвертации в Q&A
```

### Сценарий 3: Code Assistant
```yaml
# configs/code_assistant.yaml
data_sources:
  # StackOverflow вопросы
  - type: "api"
    config:
      api_type: "stackoverflow"
      params:
        tag: "python"
        max_questions: 500

  # GitHub discussions
  - type: "api"
    config:
      api_type: "github"
      params:
        repo: "python/cpython"
```

---

## 🚀 Как запустить

### 1. Установка зависимостей
```bash
cd backend
pip install -r requirements.txt
```

### 2. Создание примеров конфигов
```python
from config_loader import create_example_configs
create_example_configs()
```

### 3. Запуск через API
```bash
# Запуск FastAPI
uvicorn main:app --reload

# В другом терминале
curl -X POST http://localhost:8000/api/v2/collect/web-scrape \
  -H "Content-Type: application/json" \
  -d @request.json
```

### 4. Запуск через Python
```python
import asyncio
from data_collectors import WebScraperCollector
from llm_providers import create_provider

async def main():
    llm = create_provider("openai", model="gpt-5.1")
    scraper = WebScraperCollector(llm)

    data = await scraper.scrape_urls(
        urls=["https://example.com"],
        selector="article"
    )

    examples = scraper.convert_to_training_examples(data)
    return examples

examples = asyncio.run(main())
```

---

## 📁 Структура файлов

```
dataset-creator/
├── backend/
│   ├── data_collectors.py      # 🆕 Сбор данных
│   ├── deduplication.py        # 🆕 Дедупликация
│   ├── versioning.py           # 🆕 Версионирование
│   ├── config_loader.py        # 🆕 Конфиги
│   ├── api_extensions.py       # 🆕 API v2
│   ├── requirements.txt        # ✏️ Обновлен
│   └── ...
├── configs/
│   └── examples/               # 🆕 Примеры конфигов
│       ├── basic_synthetic.yaml
│       ├── web_scraping.yaml
│       ├── github_issues.yaml
│       ├── stackoverflow.yaml
│       └── multi_source.yaml
├── UPGRADE_GUIDE.md            # 🆕 Полная документация
└── MODULES_SUMMARY.md          # 🆕 Этот файл
```

---

## 💡 Best Practices

### 1. Всегда используй дедупликацию
```python
# После любого сбора данных
unique, stats = deduplicator.deduplicate_dataset(
    examples=examples,
    method='fuzzy',
    similarity_threshold=0.88
)
```

### 2. Создавай версии перед изменениями
```python
# Перед дедупликацией, очисткой, улучшением
version_manager.create_version(
    dataset_id=1,
    file_path=dataset_path,
    description="Перед дедупликацией"
)
```

### 3. Используй мультисорс для лучшего качества
```
40-60% - Синтетика (разнообразие)
20-30% - Web scraping (реальный контент)
20-30% - API данные (актуальность)
```

### 4. Включай quality control для важных датасетов
```yaml
quality_control:
  enabled: true
  threshold: 7.5
  auto_fix: true
```

---

## 🎓 Следующие шаги

1. ✅ Изучи примеры в `configs/examples/`
2. ✅ Запусти тестовый пайплайн
3. ✅ Создай свой конфиг для своей задачи
4. ✅ Запусти сбор данных
5. ✅ Примени дедупликацию + очистку
6. ✅ Опционально: quality control
7. ✅ Используй версионирование

**Все готово для создания мощных датасетов! 🚀**
