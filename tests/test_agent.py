import pytest
from src.agent import ResearchAssistant
from src.tools import SearchTool, SummarizerTool, CitationTool
import json

@pytest.fixture
def agent():
    return ResearchAssistant()

def test_agent_initialization(agent):
    assert agent.llm is not None
    assert len(agent.tools) > 0
    assert len(agent.memory) == 0
    assert agent.search_tool is not None
    assert agent.summarizer_tool is not None
    assert agent.citation_tool is not None

def test_tool_initialization(agent):
    tool_names = [tool.name for tool in agent.tools]
    assert "search" in tool_names
    assert "summarize" in tool_names
    assert "extract_citations" in tool_names

def test_reasoning_prompt(agent):
    prompt = agent._get_reasoning_prompt()
    assert "tools" in prompt.input_variables
    assert "memory" in prompt.input_variables
    assert "question" in prompt.input_variables

def test_tool_execution_search(agent):
    params = json.dumps({"query": "test", "max_results": 3})
    result = agent._execute_tool("search", params)
    assert isinstance(result, list)
    # Since we don't have documents loaded, it should fall back to web search
    assert len(result) <= 3

def test_tool_execution_summarize(agent):
    params = json.dumps({"text": "This is a test text to summarize.", "max_length": 50})
    result = agent._execute_tool("summarize", params)
    assert isinstance(result, str)
    assert len(result) > 0

def test_tool_execution_citations(agent):
    params = json.dumps({"text": "According to Smith (2020), this is important. Jones et al. (2021) disagree."})
    result = agent._execute_tool("extract_citations", params)
    assert isinstance(result, list)

def test_tool_execution_invalid_tool(agent):
    params = json.dumps({"query": "test"})
    with pytest.raises(ValueError):
        agent._execute_tool("invalid_tool", params)

def test_tool_execution_invalid_params(agent):
    with pytest.raises(ValueError):
        agent._execute_tool("search", "invalid json")

def test_process_tool_result(agent):
    result = agent._process_tool_result("search", "test result")
    assert len(agent.memory) == 1
    assert agent.memory[0]["tool"] == "search"
    assert agent.memory[0]["result"] == "test result" 