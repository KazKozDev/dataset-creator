
<p align="center">
  <img src="https://github.com/user-attachments/assets/503b816e-f2b7-4e75-8c10-8a13b1e755ca" alt="logo" width="150"/>
</p>
  
<p align="center">
  <a href="https://github.com/KazKozDev/dataset-creator">
    <img src="https://img.shields.io/badge/version-1.1.1-blue.svg" alt="Version">
  </a>
  <a href="https://github.com/KazKozDev/dataset-creator/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
  <a href="https://github.com/KazKozDev/dataset-creator/blob/main/docker-compose.yml">
    <img src="https://img.shields.io/badge/docker-ready-brightgreen.svg" alt="Docker">
  </a>
</p>

<p align="center">
  A synthetic data generation platform for creating and managing training datasets for LLM fine-tuning. <br><br>
    SynthGen. Data Reimagined<br><br>
  Leverages foundation models to generate domain-specific examples through an intuitive web interface.<br> Helping ML engineers and organizations easily generate high-quality data for custom AI solutions.

![Dataset Creator Interface](https://github.com/user-attachments/assets/3dedfdaa-0ea2-4667-b43d-ba6fe68fa985)

![generator](https://github.com/user-attachments/assets/d6654f21-dd34-4c2c-a76a-cc5b4bd14712)

## □ Core Features

### Foundation Model Integration
- **Multi-Provider Support**: Unified API framework for seamless integration with Ollama, OpenAI, and other LLM providers

### Training Data Engineering
- **Domain-Specific Generation Pipeline**: Create datasets tailored to vertical applications with configurable quality parameters
- **Batch Processing Orchestration**: Generate and process multiple entries with distributed task management

### Quality Assurance System
- **Data Validation Framework**: Ensure dataset quality through comprehensive validation protocols
- **Modern Interface Architecture**: React/Chakra UI implementation with advanced state management

### 🤖 LLM Agent System (NEW!)
- **Multi-Agent Architecture**: Specialized agents collaborate to create superior quality datasets
- **Quality Enhancement Pipeline**: Generator → Critic → Refiner → Diversity → Domain Expert agents
- **Automated Quality Control**: Iterative refinement ensures 8+ quality scores
- **Diversity Guarantee**: Prevents repetitive examples through intelligent comparison
- **Domain Validation**: Expert agents validate accuracy for medical, legal, financial, and other critical domains
- **Configurable Workflow**: Customize agent behavior, thresholds, and iteration limits
- **Detailed Metrics**: Track success rates, quality scores, and agent performance

> 📚 **See [AGENT_SYSTEM.md](AGENT_SYSTEM.md) for complete documentation on the agent architecture**

### Production Deployment
1. Clone the repository:
```bash
git clone https://github.com/KazKozDev/dataset-creator.git
```

2. Change to the project directory:
```bash
cd dataset-creator
```

3. Deploy with containerization:
```bash
docker-compose up -d
```

Access the application at: http://localhost:3000

### Development Environment

#### Backend Services:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend Application:
```bash
cd frontend
npm install
npm start
```

## □ Architecture

The application implements a cloud-native architecture with emphasis on scalability:

| Component | Technology | Implementation Details |
|-----------|------------|---------|
| Frontend  | React, Chakra UI | Responsive SPA with comprehensive state management |
| API Layer | Python FastAPI | RESTful services with asynchronous processing capabilities |
| Database  | PostgreSQL | Optimized schema for dataset versioning and metadata |
| Deployment| Docker | Containerized services with environment isolation |

## □ Usage

### Provider Configuration
1. Settings → Select provider → Configure authentication parameters

### Dataset Generation Workflow
1. Generator → Select domain → Define quality parameters → Execute pipeline

### Quality Management
1. Open dataset → Run validation suite → Apply improvements → Export production-ready dataset

## □ API Integration

Access comprehensive API documentation at http://localhost:8000/docs after deployment.

Key service endpoints:
- `GET /api/datasets` - List datasets with quality metrics
- `POST /api/datasets` - Create dataset with configuration parameters
- `GET /api/providers` - List available LLM providers with capabilities
- `GET /api/tasks` - Monitor task execution status

## □ License

MIT License. See [LICENSE](https://github.com/KazKozDev/dataset-creator/blob/main/LICENSE) file for details.

---


If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | [GitHub Issues](https://github.com/KazKozDev/dataset-creator/issues)
