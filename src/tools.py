from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from playwright.async_api import async_playwright
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class SearchTool:
    """Tool for searching documents and web content."""
    
    def __init__(self):
        self.llm = Ollama(model="gemma3:4b")
        self.browser = None
        self.context = None
        self.page = None
    
    async def _init_browser(self):
        """Initialize the browser if not already initialized."""
        if not self.browser:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials'
                ]
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
            )
            self.page = await self.context.new_page()
    
    async def _close_browser(self):
        """Close the browser and cleanup resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        self.browser = None
        self.context = None
        self.page = None
    
    async def web_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Perform a web search using Bing."""
        try:
            await self._init_browser()
            
            # Navigate to Bing
            await self.page.goto(f"https://www.bing.com/search?q={query}")
            
            # Wait for search results to load
            await self.page.wait_for_selector("ol#b_results", timeout=10000)
            
            # Extract search results
            results = []
            result_elements = await self.page.query_selector_all("ol#b_results > li.b_algo")
            
            for element in result_elements[:max_results]:
                try:
                    title_element = await element.query_selector("h2")
                    link_element = await element.query_selector("h2 a")
                    snippet_element = await element.query_selector("div.b_caption p")
                    
                    if title_element and link_element and snippet_element:
                        title = await title_element.inner_text()
                        url = await link_element.get_attribute("href")
                        snippet = await snippet_element.inner_text()
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        })
                except Exception as e:
                    logger.error(f"Error extracting result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
        finally:
            await self._close_browser()
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Synchronous wrapper for web search."""
        return asyncio.run(self.web_search(query, max_results))
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the document repository."""
        # TODO: Implement document search
        return []

class SummarizerTool:
    def __init__(self):
        self.llm = Ollama(model="gemma3:4b")
        
    def summarize(self, text: str, max_length: int = 500) -> str:
        """Summarize the given text."""
        prompt = f"""
        Please summarize the following text in {max_length} characters or less:
        
        {text}
        
        Summary:
        """
        
        return self.llm(prompt)

class CitationTool:
    def __init__(self):
        self.llm = Ollama(model="gemma3:4b")
        
    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from the given text."""
        prompt = f"""
        Extract all citations from the following text. Format each citation as a JSON object with 'author', 'year', and 'title' fields.
        
        Text:
        {text}
        
        Citations:
        """
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [] 