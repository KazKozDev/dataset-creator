# 🎯 LLM Dataset Creator

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

> A powerful tool for AI developers and researchers to create, enhance, and manage high-quality training datasets for Large Language Models (LLMs). Generate domain-specific examples, improve data quality, and streamline the dataset creation process - all through an intuitive web interface.

**Perfect for:**
- 🔬 AI Researchers working on LLM fine-tuning
- 💻 ML Engineers building specialized AI models
- 🏢 Companies developing domain-specific AI solutions
- 🎓 Academic teams preparing training data

## 🎥 Key Features

- 🤖 Multi-Provider Support: Works with Ollama, OpenAI, and other LLM providers
- 🎯 Domain-Specific Generation: Create datasets for support, education, healthcare, and more
- 🔄 Async Processing: Efficient batch generation with real-time progress tracking
- 📊 Quality Control: Built-in validation and enhancement tools
- 🎨 Modern UI: Clean, responsive interface built with React and Chakra UI

## ⚡ Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for development)
- Python 3.9+ (for development)

### Docker Installation

```bash
# Clone the repository
git clone https://github.com/KazKozDev/dataset-creator.git
cd dataset-creator

# Launch with Docker Compose
docker-compose up -d
```

Access the application at: http://localhost:3000

### Local Development

1. Backend setup:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

2. Frontend setup:
```bash
cd frontend
npm install
npm start
```

## 💡 Usage Guide

### 1. Configure LLM Provider

1. Go to "Settings"
2. Select your provider (e.g., Ollama)
3. Set up connection parameters

### 2. Generate Dataset

1. Navigate to "Generator"
2. Choose domain (e.g., "Support", "Education")
3. Configure generation parameters
4. Start the process

### 3. Quality Enhancement

1. Open your dataset
2. Use quality control tools
3. Apply automatic improvements

## 📚 API Reference

### Core Endpoints

- `GET /api/datasets` - List datasets
- `POST /api/datasets` - Create dataset
- `GET /api/providers` - Available LLM providers
- `GET /api/tasks` - Task status

Full API documentation: http://localhost:8000/docs

## 🤝 Contributing

We welcome contributions! 

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🆘 Support

- 📧 Email: support@example.com
- 💬 GitHub Issues
- 📚 [Wiki](https://github.com/KazKozDev/dataset-creator/wiki)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Chakra UI](https://chakra-ui.com/)
- [Ollama](https://ollama.ai/)
