from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from playwright.sync_api import sync_playwright
import json
import logging

logger = logging.getLogger(__name__)

class SearchTool:
    """Tool for searching documents and the web."""
    
    def __init__(self):
        self.llm = Ollama(model="gemma3:4b")
        self.browser = None
        self.context = None
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the document repository."""
        # TODO: Implement document search
        return []
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using Bing."""
        if not self.browser:
            self.browser = sync_playwright().start()
            self.context = self.browser.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials'
                ]
            ).new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
            )
        
        try:
            page = self.context.new_page()
            page.goto(f"https://www.bing.com/search?q={query}")
            
            # Wait for search results to load
            page.wait_for_selector("ol#b_results", timeout=10000)
            
            # Extract search results
            results = []
            result_elements = page.query_selector_all("ol#b_results > li.b_algo")
            
            for element in result_elements[:max_results]:
                try:
                    title_element = element.query_selector("h2")
                    link_element = element.query_selector("h2 a")
                    snippet_element = element.query_selector("div.b_caption p")
                    
                    if title_element and link_element and snippet_element:
                        title = title_element.inner_text()
                        url = link_element.get_attribute("href")
                        snippet = snippet_element.inner_text()
                        
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
            logger.error(f"Error during Bing search: {str(e)}")
            return []
            
        finally:
            if page:
                page.close()
    
    def __del__(self):
        """Clean up browser resources."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.stop()

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