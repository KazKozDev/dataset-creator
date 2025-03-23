# 🎯 LLM Dataset Creator

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

> Мощный инструмент для создания и улучшения наборов данных с помощью LLM моделей

## 🎥 Демонстрация

![Demo](docs/demo.gif)

## 🚀 Основные возможности

- 🤖 Интеграция с различными LLM провайдерами (Ollama, OpenAI и др.)
- 📊 Создание и улучшение наборов данных для разных доменов
- 🔄 Асинхронная обработка задач
- 📈 Контроль качества данных
- 🎨 Современный веб-интерфейс на React

## ⚡ Быстрый старт

### Предварительные требования

- Docker и Docker Compose
- Node.js 18+ (для разработки)
- Python 3.9+ (для разработки)

### Установка через Docker

```bash
# Клонируем репозиторий
git clone https://github.com/KazKozDev/dataset-creator.git
cd dataset-creator

# Запускаем через Docker Compose
docker-compose up -d
```

Приложение будет доступно по адресу: http://localhost:3000

### Локальная разработка

1. Настройка backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

2. Настройка frontend:
```bash
cd frontend
npm install
npm start
```

## 💡 Использование

### 1. Настройка провайдера LLM

1. Перейдите в раздел "Settings"
2. Выберите провайдера (например, Ollama)
3. Укажите необходимые параметры подключения

### 2. Создание набора данных

1. Перейдите в раздел "Generator"
2. Выберите домен (например, "Support", "Education")
3. Настройте параметры генерации
4. Запустите процесс

### 3. Улучшение качества

1. Откройте созданный набор данных
2. Используйте инструменты контроля качества
3. Примените автоматические улучшения

## 📚 API Reference

### Основные эндпоинты

- `GET /api/datasets` - Список наборов данных
- `POST /api/datasets` - Создание набора данных
- `GET /api/providers` - Доступные LLM провайдеры
- `GET /api/tasks` - Статус задач

Полная документация API доступна по адресу: http://localhost:8000/docs

## 🤝 Участие в разработке

Мы приветствуем ваш вклад в проект! 

1. Форкните репозиторий
2. Создайте ветку для новой функциональности
3. Внесите изменения
4. Отправьте Pull Request

## 📝 Лицензия

Этот проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).

## 🆘 Поддержка

- 📧 Email: support@example.com
- 💬 GitHub Issues
- 📚 [Wiki](https://github.com/KazKozDev/dataset-creator/wiki)

## 🙏 Благодарности

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Chakra UI](https://chakra-ui.com/)
- [Ollama](https://ollama.ai/)
