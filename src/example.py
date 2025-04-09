from agent import ResearchAssistant

def main():
    # Initialize the research assistant
    assistant = ResearchAssistant()
    
    # Example research question
    question = "What price will bitcoin reach in 2025?"
    
    print(f"Question: {question}\n")
    print("Processing...\n")
    
    # Get the answer
    answer = assistant.ask(question)
    
    print("Answer:")
    print(answer)
    
    # Print the reasoning steps
    print("\nReasoning steps:")
    for step in assistant.memory:
        print(f"- Used {step['tool']}: {step['result']}")

if __name__ == "__main__":
    main() 