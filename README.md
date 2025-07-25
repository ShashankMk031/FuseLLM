#  FuseLLM: Multi-Pipeline, Modular AI Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models)

FuseLLM is a modular AI assistant that automatically detects input types (text, audio, image, multimodal), routes tasks to appropriate Hugging Face pipelines, retrieves relevant context, fuses results, validates outputs, and returns well-formatted responses.

## Features

- **Multi-Modal Input** - Handles text, images, and audio inputs
- **Smart Task Routing** - Auto-detects input type and intent
- **Contextual Understanding** - Retrieves and incorporates relevant context
- **Pipeline Fusion** - Combines multiple model outputs intelligently
- **Output Validation** - Ensures responses are valid and coherent
- **Ethics & Safety** - Built-in content filtering
- **Dual Interface** - CLI and Gradio web UI with dark mode
- **Extensible Architecture** - Easy to add new models and capabilities

## How It Works

FuseLLM follows this workflow:

1. **Input Detection** - Identifies input type (text/audio/image)
2. **Task Routing** - Determines the appropriate processing pipeline
3. **Context Retrieval** - Fetches relevant context when available
4. **Pipeline Execution** - Runs the selected Hugging Face pipeline
5. **Result Fusion** - Combines pipeline output with context
6. **Validation & Ethics** - Ensures output quality and safety
7. **Formatting** - Presents results in a user-friendly way

## Models Used

FuseLLM leverages the following Hugging Face models:

### Text Processing
- **Text Generation**: `gpt2` (default) - For general text generation tasks
- **Question Answering**: `deepset/roberta-base-squad2` - For extracting answers from context
- **Text Classification**: `distilbert-base-uncased` - For intent classification

### Image Processing
- **Image Classification**: `google/vit-base-patch16-224` - For identifying image content
- **Image Captioning**: `nlpconnect/vit-gpt2-image-captioning` - For generating descriptions of images

### Audio Processing
- **Speech Recognition**: `facebook/wav2vec2-base-960h` - For transcribing speech to text

### Note on Response Quality
 **Important Note on Response Quality**: 
- The current implementation uses relatively small, general-purpose models to ensure broad compatibility and fast response times.
- As a result, the quality of responses may vary and might not always meet production-grade expectations.
- Some responses may be generic, off-topic, or contain inaccuracies.
- This is an experimental implementation and should be used for demonstration and development purposes only.
- For production use, consider fine-tuning models on your specific domain or using larger, more capable models.

## Current Status

### ✅ Implemented Features
- Basic intent detection and input classification
- Multi-modal input handling (text, image, audio)
- Contextual retrieval from files
- Pipeline execution with Hugging Face models
- Basic response validation and ethics checking
- CLI and Gradio interfaces
- Dark mode UI

### In Progress
- Enhanced intent classification
- Improved context retrieval
- Better error handling and user feedback

### Upcoming Features

#### Model & Features
- Confidence score integration
- Multilingual support
- Advanced intent classification model

#### Pipeline Enhancements
- Text-to-image generation
- Object detection visualization
- Code generation and analysis

#### Memory & Context
- Conversation memory
- Contextual chaining across turns

#### Deployment
- REST API with FastAPI
- Cloud deployment (Hugging Face Spaces, Docker)

#### UI/UX
- Voice input/output
- Enhanced file previews
- Chat history sidebar
- Audio waveform visualization

##  Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs** - Open an issue with detailed reproduction steps
2. **Suggest Features** - Share your ideas for new features
3. **Submit Pull Requests** - Help us improve the codebase
4. **Improve Documentation** - Help make the project more accessible
5. **Test and Report** - Try out the latest features and report any issues

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints for better code clarity
- Write docstrings for public functions and classes
- Keep commits atomic and well-described

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their amazing Transformers library
- The open-source community for their contributions and support
- All contributors who help make this project better

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA for GPU acceleration

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShashankMk031/FuseLLM.git
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

### Usage

#### Web Interface (Recommended)
```bash
python -m src.main  # Launches Gradio web UI
```

#### Command Line Interface
```bash
python -m src.main --cli  # Run in CLI mode
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

## Contributing

Contributions are welcome! and is open for all.

## Contact

For questions or feedback, please open an issue or contact [Your Email].

---

<p align="center">
  Made with ❤️ by Shashank
</p>
