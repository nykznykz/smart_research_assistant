from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferMemory
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
    """Agent that uses tools to answer research questions."""
    
    def __init__(self):
        self.llm = Ollama(model="gemma3:4b")
        # Initialize tool instances
        self.search_tool = SearchTool()
        self.summarizer_tool = SummarizerTool()
        self.citation_tool = CitationTool()
        # Store tools in a list with their metadata
        self.tools = [
            {"name": "search", "instance": self.search_tool},
            {"name": "summarize", "instance": self.summarizer_tool},
            {"name": "extract_citations", "instance": self.citation_tool}
        ]
        self.memory = ConversationBufferMemory()
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
        """Create the ReAct-style reasoning prompt."""
        return PromptTemplate(
            input_variables=["question", "tools", "memory"],
            template="""
            You are a research assistant that uses tools to answer questions.
            You have access to the following tools:
            {tools}
            
            Previous steps in memory:
            {memory}
            
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
    
    def _process_tool_result(self, action: str, result: Any) -> None:
        """Process the result of a tool execution."""
        self.memory.save_context(
            {"input": f"Action: {action}"},
            {"output": str(result)}
        )
    
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
        """Process a research question using ReAct-style reasoning."""
        logger.info(f"Starting to process question: {question}")
        
        # Initialize the reasoning chain
        reasoning_chain = LLMChain(
            llm=self.llm,
            prompt=self._get_reasoning_prompt()
        )
        
        # Start the reasoning loop
        iteration = 0
        max_iterations = 3  # Limit to 3 search iterations
        search_results = []  # Store all search results
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\nIteration {iteration}")
            
            # Get next action from LLM
            logger.info("Getting next action from LLM...")
            response = reasoning_chain.run(
                question=question,
                tools=self.tools,
                memory=self.memory
            )
            logger.info(f"LLM Response:\n{response}")
            
            # Parse the response
            try:
                thought = response.split("Action:")[0].strip()
                action = response.split("Action:")[1].split("Action Input:")[0].strip()
                action_input = response.split("Action Input:")[1].strip()
                
                logger.info(f"Thought: {thought}")
                logger.info(f"Action: {action}")
                logger.info(f"Action Input: {action_input}")
                
                # Execute the action
                if action.lower() in ["final answer", "answer the question"]:
                    logger.info("Received Final Answer")
                    return action_input
                
                # Validate the action is a defined tool
                if action not in [tool["name"] for tool in self.tools]:
                    logger.warning(f"Invalid action: {action}. Available tools: {[tool['name'] for tool in self.tools]}")
                    continue
                
                # Execute the tool
                logger.info(f"Executing tool: {action}")
                result = self._execute_tool(action, action_input)
                logger.info(f"Tool result: {result}")
                
                # Store search results
                if action == "search" and result:
                    search_results.extend(result)
                
                # Process the result
                self._process_tool_result(action, result)
                
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                logger.error(f"Full response: {response}")
                continue  # Continue to next iteration instead of raising
        
        # After max iterations, summarize the findings
        logger.info("Reached maximum iterations, summarizing findings...")
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Create a summary of all search results
        search_summary = "\n".join([
            f"Title: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}\n"
            for result in search_results
        ])
        
        # First, use the summarizer tool to create a concise summary
        logger.info("Creating summary of search results...")
        try:
            summary = self._execute_tool("summarize", json.dumps({
                "text": search_summary,
                "max_length": 1000
            }))
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            summary = search_summary  # Fall back to raw search results
        
        # Then, ask the LLM to provide a final answer based on the summary
        logger.info("Generating final answer...")
        final_response = reasoning_chain.run(
            question=question,
            tools=self.tools,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["question", "tools", "memory", "summary"],
                template="""
                Based on the following summary of research findings, please provide a comprehensive answer to the question.
                
                Question: {question}
                
                Summary of Research:
                {summary}
                
                Please provide a well-structured answer that:
                1. Directly addresses the question
                2. Provides key findings from the research
                3. Includes relevant citations or sources
                4. Is clear and concise
                
                Final Answer:
                """
            ),
            summary=summary
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