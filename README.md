# FuseLLM: Multi-Modal AI Assistant

FuseLLM is a modular, multi-pipeline AI assistant that intelligently processes text, images, and audio inputs using Hugging Face's powerful models. It automatically detects input types, routes tasks to appropriate pipelines, and delivers coherent, context-aware responses.

## Features

- **Multi-Modal Input** - Handles text, images, audio, and multimodal inputs
- **Automatic Task Routing** - Smart intent detection and pipeline selection
- **Context-Aware** - Retrieves and incorporates relevant context
- **Ethical AI** - Built-in content filtering and validation
- **Dual Interface** - Use via CLI or Gradio web UI with dark mode support
- **Modular Design** - Easy to extend with new models and features

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA for GPU acceleration

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FuseLLM.git
   cd FuseLLM
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For development:
   ```bash
   pip install -e .[dev]
   ```

## Usage

### Command Line Interface

```bash
# Process text input
python -m src.main --text "Your input text here"

# Process an image
python -m src.main --image path/to/image.jpg

# Process an audio file
python -m src.main --audio path/to/audio.wav
```

### Web Interface

Launch the Gradio web interface:

```bash
python -m src.main --web
```

Then open your browser to `http://localhost:7860`

## Project Structure

```
FuseLLM/
├── src/
│   ├── core/               # Core logic components
│   │   ├── orchestrator.py      # Central coordination
│   │   ├── task_router.py       # Input type and intent detection
│   │   ├── pipeline_manager.py  # Hugging Face pipeline management
│   │   ├── fuser.py             # Output fusion
│   │   ├── retriever.py         # Context retrieval
│   │   ├── validator.py         # Output validation
│   │   └── ethics_checker.py    # Content filtering
│   │
│   └── utils/              # Utility functions
│       ├── io_utils.py
│       ├── log_utils.py
│       └── text_utils.py
│
├── tests/                  # Unit tests
├── data/                   # Sample data
└── requirements.txt        # Dependencies
```

## Core Components

### 1. Task Router (`task_router.py`)
- Detects input type (text/image/audio)
- Classifies user intent
- Routes to appropriate processing pipeline

### 2. Pipeline Manager (`pipeline_manager.py`)
- Manages Hugging Face pipelines
- Handles model loading and caching
- Processes inputs through appropriate models

### 3. Retriever (`retriever.py`)
- Fetches relevant context
- Supports multiple file types
- Implements keyword and semantic search

### 4. Fuser (`fuser.py`)
- Combines model outputs with context
- Handles different task types
- Ensures coherent final output

### 5. Validator (`validator.py`)
- Validates output quality
- Filters low-confidence results
- Ensures response coherence

### 6. Ethics Checker (`ethics_checker.py`)
- Filters harmful content
- Implements safety checks
- Customizable filters

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses `black` for code formatting and `flake8` for linting. Run these before committing:

```bash
black .
flake8
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Contact

For questions or feedback, please open an issue or contact [Your Email].

---

<p align="center">
  Made with ❤️ by Shashank
</p>
