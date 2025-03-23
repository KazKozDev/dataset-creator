# LLM Dataset Creator

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/KazKozDev/dataset-creator)

A powerful tool for creating high-quality datasets for training and fine-tuning Large Language Models (LLMs).

## Features

- 🤖 Multi-LLM Support: Works with Anthropic Claude, OpenAI GPT, and Ollama models
- 🎯 Domain-Specific Generation: Specialized content across multiple domains (Support, Medical, Legal, etc.)
- 🌐 Multilingual Support: Generate datasets in English and Russian
- 🔄 Asynchronous Processing: Efficient batch generation with real-time progress tracking
- 📊 Quality Control: Built-in quality assessment and filtering
- 🎨 Modern UI: Clean and responsive interface built with Chakra UI
- 🚀 Real-time Updates: Live progress tracking and status updates

## Tech Stack

- Frontend:
  - React
  - Chakra UI
  - React Query
  - TypeScript

- Backend:
  - FastAPI
  - SQLite
  - Python 3.9+
  - Anthropic, OpenAI, and Ollama SDKs

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- Ollama (optional, for local LLM support)
- API keys for Anthropic and/or OpenAI (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KazKozDev/dataset-creator.git
cd dataset-creator
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Set up environment variables:
```bash
# Backend (.env)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
OLLAMA_URL=http://localhost:11434

# Frontend (.env)
REACT_APP_API_URL=http://localhost:8000
```

### Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at http://localhost:3000

## Usage

1. Select a domain and subdomain for your dataset
2. Configure generation parameters:
   - Output format (Chat/Instruction)
   - Language (English/Russian)
   - Number of examples
   - Temperature
   - LLM provider and model
3. Start generation and monitor progress
4. Access generated datasets in the Datasets tab

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**KazKozDev** (kazkozdev@gmail.com)

## Acknowledgments

- Thanks to Anthropic, OpenAI, and Ollama for their amazing LLM technologies
- Special thanks to all contributors and users of this tool
