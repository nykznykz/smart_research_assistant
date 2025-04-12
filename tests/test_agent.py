import pytest
import json
from src.agent import ResearchAssistant

def test_agent_initialization():
    """Test that the agent initializes correctly."""
    agent = ResearchAssistant()
    assert agent is not None
    assert hasattr(agent, 'llm')
    assert hasattr(agent, 'tools')
    assert hasattr(agent, 'memory')
    assert hasattr(agent, 'search_tool')
    assert hasattr(agent, 'summarizer_tool')
    assert hasattr(agent, 'citation_tool')

def test_tool_initialization():
    """Test that tools are properly initialized."""
    agent = ResearchAssistant()
    tool_names = [tool["name"] for tool in agent.tools]
    assert "search" in tool_names
    assert "summarize" in tool_names
    assert "extract_citations" in tool_names

def test_tool_execution_search():
    """Test search tool execution."""
    agent = ResearchAssistant()
    result = agent._execute_tool("search", json.dumps({
        "query": "test query",
        "max_results": 1
    }))
    assert isinstance(result, list)

def test_tool_execution_summarize():
    """Test summarizer tool execution."""
    agent = ResearchAssistant()
    result = agent._execute_tool("summarize", json.dumps({
        "text": "This is a test text to summarize.",
        "max_length": 100
    }))
    assert isinstance(result, str)

def test_tool_execution_citations():
    """Test citation tool execution."""
    agent = ResearchAssistant()
    result = agent._execute_tool("extract_citations", json.dumps({
        "text": "This is a test text with citations [1]."
    }))
    assert isinstance(result, list)

def test_tool_execution_invalid_tool():
    """Test handling of invalid tool name."""
    agent = ResearchAssistant()
    with pytest.raises(ValueError):
        agent._execute_tool("invalid_tool", json.dumps({}))

def test_tool_execution_invalid_params():
    """Test handling of invalid parameters."""
    agent = ResearchAssistant()
    with pytest.raises(ValueError):
        agent._execute_tool("search", "invalid json")

def test_process_tool_result():
    """Test processing of tool results."""
    agent = ResearchAssistant()
    agent._process_tool_result("search", "test result")
    # Verify memory was updated
    assert len(agent.memory.messages) == 2  # One user message and one AI message
    assert agent.memory.messages[0].type == "human"
    assert agent.memory.messages[1].type == "ai"
    assert "Used search tool" in agent.memory.messages[0].content
    assert "test result" in agent.memory.messages[1].content 