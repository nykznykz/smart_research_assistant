import gradio as gr
from .agent import ResearchAssistant
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the research assistant
assistant = ResearchAssistant()

def process_question(question: str) -> str:
    """Process a question using the research assistant."""
    try:
        logger.info(f"Processing question: {question}")
        answer = assistant.ask(question)
        logger.info(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"Error: {str(e)}"

# Create the Gradio interface
interface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(
        label="Question",
        placeholder="Enter your research question here...",
        lines=3
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="Research Assistant",
    description="Ask questions about any topic and get well-researched answers with citations.",
    examples=[
        "What are the main threats to AI alignment?",
        "What is the current state of quantum computing?",
        "How does climate change affect biodiversity?"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    interface.launch(share=True) 