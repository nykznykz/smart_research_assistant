from src.agent import ResearchAssistant

def main():
    # Initialize the research assistant
    assistant = ResearchAssistant()
    
    # Ask a question
    question = "What are the main threats to AI alignment?"
    print(f"Question: {question}")
    
    # Get the answer
    answer = assistant.ask(question)
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main() 