# Smart Research Assistant

A research assistant that uses AI to search, summarize, and extract citations from research materials. Built with ReAct-style reasoning for intelligent tool selection and information processing.

## Features

- ReAct-style reasoning and action selection
- Web search using Bing
- Text summarization
- Citation extraction
- Interactive Gradio interface
- Memory of previous interactions

## Technical Details

The assistant uses a ReAct (Reasoning and Acting) framework to:
1. Reason about what information is needed
2. Select appropriate tools (search, summarize, citations)
3. Process and combine information
4. Generate well-structured answers with citations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nykznykz/smart_research_assistant.git
cd smart_research_assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers:
```bash
playwright install
```

## Usage

### Command Line Interface

Run the example script:
```bash
python -m src.example
```

### Web Interface

Launch the Gradio interface:
```bash
python -m src.gradio_interface
```

This will start a web server and open the interface in your default browser. You can:
- Enter research questions
- Get AI-generated answers with citations
- Try example questions
- Share the interface with others via URL

## Development

### Testing

Run the test suite:
```bash
pytest
```

### Project Structure

- `src/agent.py`: Main agent implementation with ReAct-style reasoning
- `src/tools.py`: Tool implementations (search, summarize, citations)
- `src/gradio_interface.py`: Web interface
- `tests/`: Test files
- `requirements.txt`: Python dependencies

## License

MIT License 