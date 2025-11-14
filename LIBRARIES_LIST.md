# 📚 Список всех библиотек

## Core Framework
```
fastapi==0.109.2          # Современный веб-фреймворк для API
uvicorn==0.27.1           # ASGI сервер для FastAPI
python-multipart==0.0.9   # Поддержка multipart/form-data
sqlalchemy==2.0.27        # ORM для работы с базами данных
psutil==5.9.5             # Системная информация (CPU, память)
python-dotenv==1.0.1      # Загрузка переменных окружения из .env
pydantic==2.6.1           # Валидация данных и настроек
```

## LLM Providers
```
openai==1.12.0            # OpenAI API (GPT-4, GPT-5.1)
anthropic==0.18.1         # Anthropic API (Claude)
```

## Web Scraping & API Collection
```
aiohttp==3.9.3            # Асинхронный HTTP клиент
beautifulsoup4==4.12.3    # Парсинг HTML/XML
feedparser==6.0.11        # Парсинг RSS/Atom feeds
lxml==5.1.0               # Быстрая обработка XML/HTML
```

## Configuration & Data Processing
```
pyyaml==6.0.1             # Парсинг YAML конфигов
```

## Progress Tracking
```
tqdm==4.66.2              # Прогресс-бары для циклов
```

---

## Зачем нужна каждая библиотека

### 🌐 FastAPI + Uvicorn
**Для чего:** Создание REST API для веб-приложения
**Пример:**
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/datasets")
def get_datasets():
    return {"datasets": [...]}
```

### 🗄️ SQLAlchemy
**Для чего:** Работа с PostgreSQL базой данных
**Пример:**
```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/db")
```

### 🔍 Pydantic
**Для чего:** Валидация данных и конфигураций
**Пример:**
```python
from pydantic import BaseModel

class Config(BaseModel):
    name: str
    count: int
```

### 🤖 OpenAI / Anthropic
**Для чего:** Генерация синтетических данных с помощью LLM
**Пример:**
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[{"role": "user", "content": "Generate Q&A"}]
)
```

### 🌍 aiohttp
**Для чего:** Асинхронные HTTP запросы (быстрый парсинг)
**Пример:**
```python
import aiohttp

async with aiohttp.ClientSession() as session:
    async with session.get("https://example.com") as response:
        html = await response.text()
```

### 📄 BeautifulSoup4
**Для чего:** Парсинг HTML страниц
**Пример:**
```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, 'html.parser')
title = soup.find('h1').text
```

### 📡 feedparser
**Для чего:** Парсинг RSS/Atom feeds
**Пример:**
```python
import feedparser

feed = feedparser.parse("https://blog.com/feed.xml")
for entry in feed.entries:
    print(entry.title)
```

### ⚙️ PyYAML
**Для чего:** Загрузка YAML конфигураций
**Пример:**
```yaml
# config.yaml
name: "my_pipeline"
data_sources:
  - type: "web"
    urls: ["https://example.com"]
```
```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)
```

### 📊 tqdm
**Для чего:** Красивые прогресс-бары
**Пример:**
```python
from tqdm import tqdm

for item in tqdm(items, desc="Processing"):
    process(item)
```

---

## Установка

```bash
# Все сразу
cd backend
pip install -r requirements.txt

# Или по отдельности
pip install fastapi uvicorn
pip install openai anthropic
pip install aiohttp beautifulsoup4 feedparser
pip install pyyaml tqdm
```

---

## Опциональные библиотеки

Для расширенных функций можно добавить:

```bash
# PDF парсинг
pip install pypdf2 pdfplumber

# Word документы
pip install python-docx

# Excel файлы
pip install openpyxl pandas

# Продвинутый NLP
pip install spacy transformers

# Детекция дубликатов
pip install fuzzywuzzy python-Levenshtein

# Rate limiting для API
pip install ratelimit
```

---

## Полный requirements.txt с опциональными

```txt
# Core Framework
fastapi==0.109.2
uvicorn==0.27.1
python-multipart==0.0.9
sqlalchemy==2.0.27
psutil==5.9.5
python-dotenv==1.0.1
pydantic==2.6.1

# LLM Providers
openai==1.12.0
anthropic==0.18.1

# Web Scraping & API Collection
aiohttp==3.9.3
beautifulsoup4==4.12.3
feedparser==6.0.11
lxml==5.1.0

# Configuration & Data Processing
pyyaml==6.0.1

# Progress Tracking
tqdm==4.66.2

# Optional: Document Parsing
# pypdf2==3.0.1
# pdfplumber==0.10.3
# python-docx==1.1.0
# openpyxl==3.1.2
# pandas==2.2.0

# Optional: Advanced NLP
# spacy==3.7.4
# transformers==4.38.0

# Optional: Fuzzy Matching
# fuzzywuzzy==0.18.0
# python-Levenshtein==0.25.0

# Optional: Rate Limiting
# ratelimit==2.2.1
```

---

Все библиотеки совместимы с Python 3.8+ 🐍
