# VietnamWarQA

A comprehensive Question-Answering system focused on Vietnam War history, powered by advanced AI technology and retrieval-augmented generation (RAG) capabilities.

## Overview

VietnamWarQA is an intelligent conversational AI system designed to provide accurate, contextual, and comprehensive answers about Vietnam War history. The project leverages state-of-the-art language models from NVIDIA's API ecosystem, combined with retrieval-augmented generation techniques to deliver historically accurate and well-sourced responses.

The system is built with a modular architecture that supports multiple data sources, advanced natural language processing capabilities, and seamless integration with various AI model providers. It serves as both an educational tool and a research assistant for anyone interested in understanding the complexities of the Vietnam War period.

Key capabilities include:
- Historical fact verification and cross-referencing
- Multi-perspective analysis of events and decisions
- Timeline-based question answering
- Primary source document analysis
- Interactive conversational interface with memory persistence
- Real-time streaming responses with citation support

## Project Structure



## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git version control
- Docker (optional, for containerized deployment)
- 4GB+ RAM recommended for optimal performance

### Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/VietnamWarQA.git
cd VietnamWarQA
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env file with your API keys and configuration
```

5. Initialize the database and embeddings:
```bash
python scripts/setup_environment.py
python scripts/build_embeddings.py
```

## Usage

### Basic Chat Interface
```bash
python src/main.py
```

### Advanced Configuration
```bash
python src/main.py --config custom_config.yml --model nvidia/deepseek-r1
```

### API Server Mode
```bash
python src/main.py --server --port 8080
```

### Batch Processing
```bash
python examples/batch_processing.py --input questions.txt --output results.json
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete details.

Copyright (c) 2024 VietnamWarQA Project Contributors. All rights reserved.