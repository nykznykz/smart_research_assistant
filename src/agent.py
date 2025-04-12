from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from .tools import SearchTool, SummarizerTool, CitationTool
import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    
class ResearchAssistant:
    """A research assistant that can search, summarize, and extract citations."""
    
    def __init__(self):
        self.llm = OllamaLLM(model="gemma3:4b")
        self.memory = ChatMessageHistory()
        
        # Initialize tools
        self.search_tool = SearchTool()
        self.summarizer_tool = SummarizerTool()
        self.citation_tool = CitationTool()
        
        # Define available tools
        self.tools = [
            {
                "name": "search",
                "instance": self.search_tool,
                "description": "Search for information on a topic"
            },
            {
                "name": "summarize",
                "instance": self.summarizer_tool,
                "description": "Summarize text"
            },
            {
                "name": "extract_citations",
                "instance": self.citation_tool,
                "description": "Extract citations from text"
            }
        ]
        self.reasoning_chain = None
        
    def _initialize_tools(self):
        """Initialize the available tools for the agent."""
        self.tools = [
            Tool(
                name="search",
                description="Search for relevant documents or information",
                parameters={"query": "str", "max_results": "int"}
            ),
            Tool(
                name="summarize",
                description="Summarize a document or text",
                parameters={"text": "str", "max_length": "int"}
            ),
            Tool(
                name="extract_citations",
                description="Extract citations from a document",
                parameters={"text": "str"}
            )
        ]
    
    def _get_reasoning_prompt(self) -> PromptTemplate:
        """Get the reasoning prompt template."""
        return PromptTemplate(
            input_variables=["question", "tools", "chat_history"],
            template="""
            You are a research assistant that helps answer questions by using available tools.
            You have access to the following tools:
            {tools}
            
            Previous conversation history:
            {chat_history}
            
            Current question: {question}
            
            Think step by step about how to answer the question.
            First, reason about what information you need and which tools to use.
            Then, decide on the next action to take.
            
            Format your response as:
            Thought: <your reasoning>
            Action: <tool name>
            Action Input: <tool parameters as JSON>
            
            Example Action Input for search:
            {{"query": "your search query", "max_results": 5}}
            
            Example Action Input for summarize:
            {{"text": "text to summarize", "max_length": 500}}
            
            Example Action Input for extract_citations:
            {{"text": "text with citations"}}
            
            Important rules:
            1. Use the exact tool names: 'search', 'summarize', or 'extract_citations'
            2. After gathering information, use the summarize tool to create a concise summary
            3. After summarizing, provide a Final Answer that directly addresses the question
            4. Do not continue searching after you have enough information to answer the question
            5. Use the extract_citations tool to ensure your answer includes proper citations
            """
        )
    
    def _process_tool_result(self, tool_name: str, result: Any) -> None:
        """Process the result of a tool execution."""
        # Add the tool result to memory
        self.memory.add_user_message(f"Used {tool_name} tool")
        self.memory.add_ai_message(str(result))
    
    def _parse_action_input(self, action_input: str) -> Dict[str, Any]:
        """Parse the action input into a dictionary of parameters."""
        # Try to find JSON in the input
        json_match = re.search(r'\{.*\}', action_input, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, try to extract parameters manually
        params = {}
        for line in action_input.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                params[key.strip()] = value.strip().strip('"\'')
        
        return params
    
    def ask(self, question: str) -> str:
        """Ask a question and get an answer."""
        logger.info(f"Starting to process question: {question}")
        
        # Get the reasoning prompt
        prompt = self._get_reasoning_prompt()
        
        # Format the chat history for the prompt
        chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in self.memory.messages])
        
        # Get the next action from the LLM
        logger.info("Getting next action from LLM...")
        response = self.llm.invoke(
            prompt.format(
                question=question,
                tools=self.tools,
                chat_history=chat_history
            )
        )
        logger.info(f"LLM Response:\n{response}")
        
        # Parse the response
        thought = response.split("Action:")[0].strip().replace("Thought:", "").strip()
        action = response.split("Action:")[1].split("Action Input:")[0].strip()
        action_input = response.split("Action Input:")[1].strip()
        
        logger.info(f"Thought: {thought}")
        logger.info(f"Action: {action}")
        logger.info(f"Action Input: {action_input}")
        
        # Execute the tool
        logger.info(f"Executing tool: {action}")
        result = self._execute_tool(action, action_input)
        logger.info(f"Tool result: {result}")
        
        # Process the result
        self._process_tool_result(action, result)
        
        # Get the final answer
        final_prompt = PromptTemplate(
            input_variables=["question", "tools", "chat_history", "summary"],
            template="""
            Based on the following summary of research findings, please provide a comprehensive answer to the question.
            
            Question: {question}
            
            Research findings:
            {summary}
            
            Previous conversation history:
            {chat_history}
            
            Please provide a clear, concise answer that directly addresses the question.
            Include relevant citations and sources where appropriate.
            """
        )
        
        final_response = self.llm.invoke(
            final_prompt.format(
                question=question,
                tools=self.tools,
                chat_history=chat_history,
                summary=result
            )
        )
        
        return final_response
    
    def _execute_tool(self, tool_name: str, params: str) -> Any:
        """Execute a tool with the given parameters."""
        try:
            # Parse parameters
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    # For invalid JSON, raise ValueError
                    raise ValueError(f"Invalid JSON parameters: {params}")
            
            # Find the tool instance
            tool_info = next((t for t in self.tools if t["name"] == tool_name), None)
            if not tool_info:
                raise ValueError(f"Tool {tool_name} not found")
            
            tool = tool_info["instance"]
            
            # Execute the tool with correct parameters
            if tool_name == "search":
                # For search, we need to handle the async operation
                return tool.search(query=params.get("query", ""), max_results=params.get("max_results", 3))
            elif tool_name == "summarize":
                return tool.summarize(text=params.get("text", ""), max_length=params.get("max_length", 500))
            elif tool_name == "extract_citations":
                return tool.extract_citations(text=params.get("text", ""))
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise 