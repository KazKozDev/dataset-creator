# 📊 LLM Dataset Creator

A Python-based application for generating, managing, and validating training datasets for Large Language Models.

## 1. Project Introduction

**Project Type**: Web application with REST API backend  
**Version**: 1.0.0-beta

LLM Dataset Creator enables AI researchers and ML engineers to generate high-quality, domain-specific training datasets for language models. It automates the creation process through an intuitive web interface and provides tools for quality control and validation.

![Application Interface](docs/interface.png)

### Problem Solved
Manual dataset creation for LLM training is time-consuming and error-prone. This tool automates the process while ensuring data quality and consistency across different domains.

### Target Users
- AI Researchers working on LLM fine-tuning
- ML Engineers developing specialized models
- Organizations building domain-specific AI solutions
- Academic teams preparing training data

## 2. Core Features

### Dataset Generation
- **Function**: `create_dataset(domain: str, size: int, params: dict) -> Dataset`
- **Input**: Domain specification, size requirements, and generation parameters
- **Output**: Structured dataset in JSONL format
- **Configuration**:
  ```python
  {
    "domain": "support",
    "size": 100,
    "params": {
      "language": "en",
      "model": "llama2",
      "temperature": 0.7
    }
  }
  ```

### Quality Control
- **Function**: `validate_dataset(dataset_id: int, criteria: dict) -> ValidationReport`
- **Input**: Dataset ID and validation criteria
- **Output**: Quality metrics and validation report
- **Parameters**: Customizable quality thresholds and validation rules

### Batch Processing
- **Function**: `process_batch(tasks: List[Task]) -> BatchResult`
- **Input**: List of generation tasks
- **Output**: Processing status and results
- **Monitoring**: Real-time progress tracking

## 3. Installation & Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Node.js 18+
- PostgreSQL 13+

### Quick Start (Docker)
```bash
# Clone repository
git clone https://github.com/KazKozDev/dataset-creator.git
cd dataset-creator

# Launch application
docker-compose up -d
```

Access the application at http://localhost:3000 (setup time: ~2 minutes)

### Development Setup
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

## 4. API Documentation

### Dataset Operations

#### Create Dataset
```python
POST /api/datasets
{
    "domain": "support",
    "size": 100,
    "params": {
        "language": "en",
        "model": "llama2"
    }
}
```

#### List Datasets
```python
GET /api/datasets
Response: List[Dataset]
```

### Task Management

#### Get Task Status
```python
GET /api/tasks/{task_id}
Response: TaskStatus
```

Full API documentation available at http://localhost:8000/docs

## 5. Project Structure
```
dataset-creator/
├── backend/
│   ├── main.py           # FastAPI application
│   ├── generator.py      # Dataset generation logic
│   ├── quality.py        # Quality control
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   └── services/     # API integration
│   └── package.json
├── docker/
│   └── docker-compose.yml
└── README.md
```

## 6. Contributing

1. Fork the repository
2. Create a feature branch
3. Follow code style guidelines:
   - Backend: PEP 8
   - Frontend: ESLint configuration
4. Submit a Pull Request

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 7. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 8. Contact & Support

- **GitHub Issues**: Bug reports and feature requests
- **Email**: kazkozdev@gmail.com
- **Documentation**: [Project Wiki](https://github.com/KazKozDev/dataset-creator/wiki)

## Dependencies

### Backend
- FastAPI
- SQLAlchemy
- Python LLM libraries

### Frontend
- React
- Chakra UI
- React Query
