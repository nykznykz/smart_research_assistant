# Smart Research Assistant

A ReAct-style agent that uses tools to conduct research and provide well-structured summaries with citations.

## Features

- ReAct-style reasoning and action selection
- Document search and retrieval
- Information extraction and summarization
- Citation management
- Chain-of-thought reasoning

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

```python
from src.agent import ResearchAssistant

agent = ResearchAssistant()
response = agent.ask("What are the top 3 threats to AI alignment raised in recent research?")
print(response)
```

## Project Structure

```
smart_research_assistant/
├── src/                # Source code
├── tests/             # Test cases
├── data/              # Data and document storage
├── config/            # Configuration files
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Development

- Run tests: `pytest tests/`
- Format code: `black src/`
- Type checking: `mypy src/` 