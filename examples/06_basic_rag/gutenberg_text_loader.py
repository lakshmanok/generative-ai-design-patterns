import os
import re
import time
import hashlib
import requests
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Any
from urllib.parse import urlparse
from llama_index.core import Document
from abc import ABC, abstractmethod

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages the local cache for downloaded files.
    
    Attributes:
        cache_dir (Path): Path to the cache directory.
    """
    
    def __init__(self, cache_dir: str = "./.cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir (str): Path to the cache directory. Defaults to "./.cache".
        """
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Create the cache directory if it doesn't exist."""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
            logger.info(f"Created cache directory at {self.cache_dir}")
    
    def _get_cache_filename(self, url: str) -> str:
        """
        Generate a unique filename for a URL.
        
        Args:
            url (str): The URL to generate a filename for.
            
        Returns:
            str: A unique filename based on the URL.
        """
        # Extract the filename from the URL if possible
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        original_filename = path_parts[-1] if path_parts[-1] else "index"
        
        # Create a hash of the URL to ensure uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        
        # Combine original filename with hash
        if '.' in original_filename:
            name_parts = original_filename.split('.')
            extension = name_parts[-1]
            base_name = '.'.join(name_parts[:-1])
            return f"{base_name}_{url_hash}.{extension}"
        else:
            return f"{original_filename}_{url_hash}.txt"
    
    def get_cache_path(self, url: str) -> Path:
        """
        Get the cache path for a URL.
        
        Args:
            url (str): The URL to get the cache path for.
            
        Returns:
            Path: The path where the cached file would be stored.
        """
        filename = self._get_cache_filename(url)
        return self.cache_dir / filename
    
    def is_cached(self, url: str) -> bool:
        """
        Check if a URL is already cached.
        
        Args:
            url (str): The URL to check.
            
        Returns:
            bool: True if the URL is cached, False otherwise.
        """
        cache_path = self.get_cache_path(url)
        return cache_path.exists()
    
    def get_cached_content(self, url: str) -> Optional[str]:
        """
        Get the cached content for a URL.
        
        Args:
            url (str): The URL to get the cached content for.
            
        Returns:
            Optional[str]: The cached content if available, None otherwise.
        """
        if not self.is_cached(url):
            return None
        
        cache_path = self.get_cache_path(url)
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading cached file for {url}: {e}")
            return None
    
    def cache_content(self, url: str, content: str) -> bool:
        """
        Cache content for a URL.
        
        Args:
            url (str): The URL the content was downloaded from.
            content (str): The content to cache.
            
        Returns:
            bool: True if caching was successful, False otherwise.
        """
        self._ensure_cache_dir()
        cache_path = self.get_cache_path(url)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Cached content for {url} at {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error caching content for {url}: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear all cached files.
        
        Returns:
            bool: True if clearing was successful, False otherwise.
        """
        try:
            if self.cache_dir.exists():
                for file_path in self.cache_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_size(self) -> Tuple[int, str]:
        """
        Get the total size of the cache.
        
        Returns:
            Tuple[int, str]: A tuple containing the size in bytes and a human-readable size.
        """
        total_size = 0
        
        if self.cache_dir.exists():
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        # Convert to human-readable format
        units = ['B', 'KB', 'MB', 'GB']
        size_human = total_size
        unit_index = 0
        
        while size_human > 1024 and unit_index < len(units) - 1:
            size_human /= 1024
            unit_index += 1
        
        human_readable = f"{size_human:.2f} {units[unit_index]}"
        return total_size, human_readable
    
    def list_cached_files(self) -> List[Dict[str, Any]]:
        """
        List all cached files with metadata.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing file information.
        """
        files_info = []
        
        if self.cache_dir.exists():
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append({
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size_bytes': stat.st_size,
                        'last_modified': time.ctime(stat.st_mtime)
                    })
        
        return files_info



class GutenbergTextLoadError(Exception):
    """Exception raised for errors in loading Gutenberg text files."""
    pass

class DocumentSource(ABC):
    @abstractmethod
    def load_from_url(self, url) -> Document:
        pass

class GutenbergSource(DocumentSource):
    """
    A class to load text files from Project Gutenberg as a LlamaIndex Document.
    
    This class handles fetching text content from URLs, processing Gutenberg-specific
    formatting, and creating a document store indexed by BM25.
    
    Attributes:
        cache_manager (CacheManager): Manager for the local cache.
    """
    
    def __init__(
        self,
        cache_dir: str = "./.cache",
    ):
        self.cache_manager = CacheManager(cache_dir)
   
    def _fetch_text_from_url(self, url: str) -> str:
        """
        Fetch text content from a URL with caching
        
        Args:
            url (str): URL to fetch text from.
            
        Returns:
            str: Text content from the URL.
            
        Raises:
            GutenbergTextLoadError: If there's an error fetching or processing the URL.
        """
        if self.cache_manager.is_cached(url):
            logger.info(f"Loading {url} from cache")
            cached_content = self.cache_manager.get_cached_content(url)
            if cached_content:
                return cached_content
            logger.warning(f"Cached content for {url} could not be read, downloading again")
        
        try:
            logger.info(f"Fetching text from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if content is text
            content_type = response.headers.get('Content-Type', '')
            if 'text/plain' not in content_type and 'text/html' not in content_type:
                raise GutenbergTextLoadError(f"URL does not contain text content: {content_type}")
            
            # Detect encoding or use utf-8 as fallback
            encoding = response.encoding or 'utf-8'
            content = response.content.decode(encoding)
        
            # Cache the downloaded content
            self.cache_manager.cache_content(url, content)
            
            return content
        except requests.RequestException as e:
            raise GutenbergTextLoadError(f"Error fetching URL {url}: {str(e)}")
        except UnicodeDecodeError as e:
            raise GutenbergTextLoadError(f"Error decoding content from {url}: {str(e)}")
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """
        Clean Project Gutenberg text by removing headers, footers, and license information.
        
        Args:
            text (str): Raw text from Project Gutenberg.
            
        Returns:
            str: Cleaned text with Gutenberg-specific content removed.
        """
        # Pattern to find the start of the actual content (after header)
        start_markers = [
            r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .+? \*\*\*",
            r"\*\*\* START OF THE PROJECT GUTENBERG .+? \*\*\*",
            r"\*\*\*START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
            r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK"
        ]
        
        # Pattern to find the end of the content (before footer)
        end_markers = [
            r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .+? \*\*\*",
            r"\*\*\* END OF THE PROJECT GUTENBERG .+? \*\*\*", 
            r"\*\*\*END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
            r"END OF (THIS|THE) PROJECT GUTENBERG EBOOK"
        ]
        
        # Find start of content
        start_pos = 0
        for marker in start_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                start_pos = match.end()
                break
        
        # Find end of content
        end_pos = len(text)
        for marker in end_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                end_pos = match.start()
                break
        
        # Extract and clean the content
        content = text[start_pos:end_pos].strip()
        
        # Remove extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        logger.info(f"Cleaned Gutenberg text: removed {start_pos} chars from start, "
                   f"{len(text) - end_pos} chars from end")
        
        return content
    
    def load_from_url(self, url) -> Document:
        """
        Load text from a URL and return a LlamaIndex Document.
        
        Args:
            url (str, optional): URL to load text from. If None, uses the default URL.
            
        Returns:
            Document
            
        Raises:
            GutenbergTextLoadError: If there's an error loading or processing the text.
        """
        url = url or self.default_url
        
        try:
            # Fetch and clean the text
            raw_text = self._fetch_text_from_url(url)
            cleaned_text = self._clean_gutenberg_text(raw_text)
            
            # Create a document with metadata
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            document = Document(
                text=cleaned_text,
                metadata={
                    "source": url,
                    "filename": filename,
                    "date_loaded": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            
            logger.info(f"Successfully loaded text from {url}.")
            
            return document
 
        except Exception as e:
            raise GutenbergTextLoadError(f"Error loading from URL {url}: {str(e)}")