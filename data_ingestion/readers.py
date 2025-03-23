"""
Data source readers that extract data from various sources.
"""
import os
import logging
from typing import Dict, Any, BinaryIO
from pathlib import Path
from datetime import datetime
import chardet
from .interfaces import DataSourceReader

logger = logging.getLogger(__name__)


class CloudStorageReader(DataSourceReader):
    """Reader for cloud storage."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """Initialize the reader with credentials."""
        self.credentials = credentials
    
    def _download_resource(self, source_path: str) -> bytes:
        """Download the resource from cloud storage."""
        pass

    def read(self, source_path: str) -> bytes:
        """Read data from cloud storage."""
        pass

    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """Get metadata for the resource."""
        pass

    def get_text(self, text: str) -> str:
        """Post-process the text."""
        pass


import requests
from bs4 import BeautifulSoup


class CMSReader(CloudStorageReader):
    """Reader for Content Management Systems (websites)."""
    
    def __init__(self, credentials: Dict[str, Any] = None):
        """Initialize the reader with optional credentials for authenticated sites."""
        super().__init__(credentials or {})
        self.session = requests.Session()
        if credentials:
            # Setup authentication if credentials are provided
            # This could include setting cookies, headers, etc.
            pass
    
    def read(self, source_path: str) -> bytes:
        """
        Read data from a website URL.
        
        Args:
            source_path: The URL of the webpage to fetch
            
        Returns:
            The raw HTML content as bytes
        """
        try:
            response = self.session.get(source_path, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.content
        except requests.RequestException as e:
            logger.error(f"Error fetching content from {source_path}: {e}")
            raise
    
    def get_text(self, html_content: bytes or str, target_tags=None) -> str:
        """
        Extract readable text content from HTML using BeautifulSoup, keeping only text 
        from specified tags.
        
        Args:
            html_content: HTML content either as bytes or string
            target_tags: List of HTML tags to extract text from (e.g., ['div', 'p', 'ul', 'table'])
                         If None, extracts text from all tags (default behavior)
            
        Returns:
            Cleaned text with HTML tags and links removed
        """
        if isinstance(html_content, bytes):
            # Detect encoding
            encoding = 'utf-8'
            html_text = html_content.decode(encoding, errors='replace')
        else:
            html_text = html_content
            
        # Parse HTML
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Remove script, style, and navigation elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Add spacing between block elements to prevent text from sticking together
        for tag in soup.find_all(['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr']):
            if tag.string:
                tag.string.replace_with(tag.string + " ")
        
        if target_tags:
            # Extract text only from specified tags
            extracted_text = []
            
            for tag_name in target_tags:
                for tag in soup.find_all(tag_name):
                    # Replace images with simple placeholder
                    for img in tag.find_all('img'):
                        img.replace_with(' ')
                    
                    # Keep only the text from links (remove URLs)
                    for a in tag.find_all('a'):
                        a.replace_with(a.get_text())
                    
                    # Get text from this tag
                    tag_text = tag.get_text()
                    if tag_text.strip():
                        extracted_text.append(tag_text)
            
            # Join extracted texts
            text = '\n\n'.join(extracted_text)
        else:
            # Original behavior - extract text from all tags
            # Replace images with simple placeholder
            for img in soup.find_all('img'):
                img.replace_with(' ')
            
            # Keep only the text from links (remove URLs)
            for a in soup.find_all('a'):
                a.replace_with(a.get_text())
            
            # Get text
            text = soup.get_text()
        
        # Better whitespace cleaning
        # First normalize all whitespace
        text = ' '.join(text.split())
        
        # Restore proper paragraph breaks
        text = text.replace('. ', '.\n')
        
        # Fix spacing between sentences
        text = text.replace('.\n', '.\n\n')
        text = text.replace('!\n', '!\n\n')
        text = text.replace('?\n', '?\n\n')
        
        # Cleanup any remaining whitespace issues
        lines = [line.strip() for line in text.splitlines()]
        text = '\n\n'.join(line for line in lines if line)
        
        return text
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the webpage using BeautifulSoup.
        
        Args:
            source_path: The URL of the webpage
            
        Returns:
            Dictionary containing metadata
        """
        html_content = self.read(source_path)
        
        if isinstance(html_content, bytes):
            # Detect encoding
            detected = chardet.detect(html_content)
            encoding = detected['encoding'] or 'utf-8'
            html_text = html_content.decode(encoding, errors='replace')
        else:
            html_text = html_content
            
        soup = BeautifulSoup(html_text, 'html.parser')
        
        metadata = {
            'url': source_path,
            'fetch_time': datetime.now().isoformat()
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.text.strip()
            
        # Extract meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '')
            
        # Extract Open Graph metadata
        for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            property_name = meta.get('property')[3:]  # Remove 'og:' prefix
            if property_name:
                metadata[f'og_{property_name}'] = meta.get('content', '')
                
        # Try to extract publish date
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta:
            metadata['published_date'] = date_meta.get('content', '')
            
        return metadata