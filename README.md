# Dataset Creator

Tool for creating and managing training datasets for LLM. Generates domain-specific examples through a web interface.

Designed for AI researchers and ML engineers who need to create high-quality training datasets for language models. The platform provides an environment for generating, validating, and managing domain-specific data examples.

![screen](https://github.com/user-attachments/assets/3dedfdaa-0ea2-4667-b43d-ba6fe68fa985)

## Who is it for?

- AI Researchers seeking comprehensive dataset generation capabilities
- ML Engineers who need to create domain-specific training data
- Organizations developing specialized models
- Academic research teams requiring quality training datasets

## Features

- **Multi-Provider Support**: Seamless integration with Ollama, OpenAI, and other LLM providers
- **Domain-Specific Generation**: Create datasets tailored to specific fields and applications
- **Batch Processing**: Generate and process multiple entries with progress tracking
- **Data Validation**: Ensure dataset quality with integrated validation tools
- **Modern Interface**: React/Chakra UI for intuitive dataset management

## Architecture

The application follows a modern microservices architecture:

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend  | React, Chakra UI | Responsive UI for dataset creation and management |
| Backend   | Python FastAPI | Efficient API services and LLM integration |
| Database  | PostgreSQL | Secure storage for datasets and metadata |
| Deployment| Docker | Consistent environments across systems |

## Setup

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for development)
- Python 3.9+ (for development)

### Docker Setup

```bash
git clone https://github.com/KazKozDev/dataset-creator.git
cd dataset-creator
docker-compose up -d
```

Access the application at: http://localhost:3000

### Development Setup

1. Backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

2. Frontend:

```bash
cd frontend
npm install
npm start
```

## Usage

### Provider Configuration
1. Settings → Select provider → Configure parameters

### Dataset Generation
1. Generator → Select domain → Set parameters → Start generation

### Quality Management
1. Open dataset → Run quality tools → Apply improvements

## API Integration

Access comprehensive API documentation at http://localhost:8000/docs after startup.

Key endpoints:
- `GET /api/datasets` - List datasets
- `POST /api/datasets` - Create dataset
- `GET /api/providers` - List providers
- `GET /api/tasks` - Task status

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Submit Pull Request

## License

MIT License. See LICENSE file for details.

## Support

- GitHub Issues
- Wiki
- Email: kazkozdev@gmail.com

## 📚 Dependencies

- FastAPI
- React
- Chakra UI
- Ollama
