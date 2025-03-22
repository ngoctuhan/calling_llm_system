"""
Data source readers that extract data from various sources.
"""
import os
import logging
from typing import Dict, Any, BinaryIO
from pathlib import Path
from datetime import datetime

from .interfaces import DataSourceReader

logger = logging.getLogger(__name__)


class BaseReader(DataSourceReader):
    """Base implementation for data source readers."""
    
    def read(self, source_path: str) -> bytes:
        """
        Read binary data from a local file source.
        
        Args:
            source_path: Path to the data source
            
        Returns:
            Raw binary data from the source
        """
        try:
            with open(source_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {source_path}: {str(e)}")
            raise
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata from the data source.
        
        Args:
            source_path: Path to the data source
            
        Returns:
            Dictionary containing metadata
        """
        try:
            path = Path(source_path)
            stats = path.stat()
            
            return {
                'filename': path.name,
                'file_path': str(path.absolute()),
                'file_extension': path.suffix.lower(),
                'size_bytes': stats.st_size,
                'creation_time': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modification_time': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'content_type': self._get_content_type(path.suffix.lower()),
                'source_type': 'local_file'
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {source_path}: {str(e)}")
            return {
                'filename': os.path.basename(source_path),
                'file_path': source_path,
                'file_extension': os.path.splitext(source_path)[1].lower(),
                'error': str(e),
                'source_type': 'local_file'
            }
    
    def _get_content_type(self, extension: str) -> str:
        """Get content type based on file extension."""
        mapping = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.txt': 'text',
            '.md': 'text',
            '.xlsx': 'spreadsheet',
            '.xls': 'spreadsheet',
            '.csv': 'table',
            '.pptx': 'presentation',
            '.ppt': 'presentation',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.m4a': 'audio',
            '.ogg': 'audio'
        }
        return mapping.get(extension, 'unknown')


class PDFReader(BaseReader):
    """Reader for PDF files."""
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(source_path)
        metadata['content_type'] = 'document'
        
        # Add PDF-specific metadata
        # In a real implementation, you would use a library like PyPDF2 or pdfminer
        # to extract more PDF-specific metadata
        
        return metadata


class ExcelReader(BaseReader):
    """Reader for Excel and CSV files."""
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(source_path)
        
        ext = Path(source_path).suffix.lower()
        if ext == '.csv':
            metadata['content_type'] = 'table'
        else:
            metadata['content_type'] = 'spreadsheet'
            
        # In a real implementation, you would use openpyxl or pandas
        # to extract more Excel-specific metadata like sheet names
        
        return metadata


class PowerPointReader(BaseReader):
    """Reader for PowerPoint files."""
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(source_path)
        metadata['content_type'] = 'presentation'
        
        # In a real implementation, you would use python-pptx
        # to extract more PowerPoint-specific metadata
        
        return metadata


class AudioReader(BaseReader):
    """Reader for audio files."""
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(source_path)
        metadata['content_type'] = 'audio'
        
        # In a real implementation, you would use a library like pydub or librosa
        # to extract more audio-specific metadata like duration, sample rate, etc.
        
        return metadata


class TextReader(BaseReader):
    """Reader for text files."""
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(source_path)
        metadata['content_type'] = 'text'
        return metadata


class CloudStorageReader(DataSourceReader):
    """Base class for cloud storage readers."""
    
    def read(self, source_path: str) -> bytes:
        """
        Read binary data from a cloud storage source.
        
        Args:
            source_path: URI to the data source
            
        Returns:
            Raw binary data from the source
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a cloud storage source.
        
        Args:
            source_path: URI to the data source
            
        Returns:
            Dictionary containing metadata
        """
        raise NotImplementedError("Subclasses must implement this method")


class SharePointReader(CloudStorageReader):
    """Reader for SharePoint files."""
    
    def read(self, source_path: str) -> bytes:
        """
        Read binary data from a SharePoint document.
        
        Args:
            source_path: SharePoint URI (sharepoint://site/library/path/to/file)
            
        Returns:
            Raw binary data from the source
        """
        # In a real implementation, you would use the Office 365 REST API
        # or Microsoft Graph API to download the file
        logger.info(f"Reading SharePoint document: {source_path}")
        
        # Placeholder implementation
        return b"SharePoint document content"
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a SharePoint document.
        
        Args:
            source_path: SharePoint URI
            
        Returns:
            Dictionary containing metadata
        """
        # Parse the URI to extract site, library, and file path
        parts = source_path.replace('sharepoint://', '').split('/')
        site = parts[0] if len(parts) > 0 else ""
        library = parts[1] if len(parts) > 1 else ""
        file_path = '/'.join(parts[2:]) if len(parts) > 2 else ""
        
        filename = file_path.split('/')[-1] if file_path else ""
        file_extension = os.path.splitext(filename)[1].lower() if filename else ""
        
        return {
            'filename': filename,
            'file_path': source_path,
            'file_extension': file_extension,
            'content_type': self._get_content_type(file_extension),
            'source_type': 'sharepoint',
            'site': site,
            'library': library,
            'relative_path': file_path
        }
    
    def _get_content_type(self, extension: str) -> str:
        """Get content type based on file extension."""
        # Reuse the mapping from BaseReader
        mapping = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.txt': 'text',
            '.md': 'text',
            '.xlsx': 'spreadsheet',
            '.xls': 'spreadsheet',
            '.csv': 'table',
            '.pptx': 'presentation',
            '.ppt': 'presentation',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.m4a': 'audio',
            '.ogg': 'audio'
        }
        return mapping.get(extension, 'unknown')


class OneDriveReader(CloudStorageReader):
    """Reader for OneDrive files."""
    
    def read(self, source_path: str) -> bytes:
        """
        Read binary data from a OneDrive document.
        
        Args:
            source_path: OneDrive URI (onedrive://path/to/file)
            
        Returns:
            Raw binary data from the source
        """
        # In a real implementation, you would use the Microsoft Graph API
        # to download the file
        logger.info(f"Reading OneDrive document: {source_path}")
        
        # Placeholder implementation
        return b"OneDrive document content"
    
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a OneDrive document.
        
        Args:
            source_path: OneDrive URI
            
        Returns:
            Dictionary containing metadata
        """
        # Parse the URI to extract the file path
        file_path = source_path.replace('onedrive://', '')
        
        filename = file_path.split('/')[-1] if file_path else ""
        file_extension = os.path.splitext(filename)[1].lower() if filename else ""
        
        return {
            'filename': filename,
            'file_path': source_path,
            'file_extension': file_extension,
            'content_type': self._get_content_type(file_extension),
            'source_type': 'onedrive',
            'relative_path': file_path
        }
    
    def _get_content_type(self, extension: str) -> str:
        """Get content type based on file extension."""
        # Reuse the mapping from BaseReader
        mapping = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.txt': 'text',
            '.md': 'text',
            '.xlsx': 'spreadsheet',
            '.xls': 'spreadsheet',
            '.csv': 'table',
            '.pptx': 'presentation',
            '.ppt': 'presentation',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.m4a': 'audio',
            '.ogg': 'audio'
        }
        return mapping.get(extension, 'unknown') 