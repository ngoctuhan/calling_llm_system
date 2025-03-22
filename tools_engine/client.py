import logging
import grpc
import os
import tempfile
import subprocess
import json
import time
from typing import Dict, Any, List, Optional, Union

# Import the generated gRPC code
# Note: These imports will work after running generate_protos.py
# from app.tools_engine.tools_pb2 import *
# from app.tools_engine.tools_pb2_grpc import *

# For now, use placeholders
TextProcessingServiceStub = object
DataTransformationServiceStub = object
SpeechProcessingServiceStub = object

from app.core.config import settings

logger = logging.getLogger(__name__)

class ToolsClient:
    """Client for interacting with the tools gRPC services"""
    
    def __init__(self):
        """Initialize the client and create the stubs"""
        self.channel = None
        self.text_processing_stub = None
        self.data_transformation_stub = None
        self.speech_processing_stub = None
        self._connect()
    
    def _connect(self):
        """Establish connection to the gRPC server"""
        server_address = f"{settings.GRPC_HOST}:{settings.GRPC_PORT}"
        self.channel = grpc.insecure_channel(server_address)
        
        # Create service stubs
        self.text_processing_stub = TextProcessingServiceStub(self.channel)
        self.data_transformation_stub = DataTransformationServiceStub(self.channel)
        self.speech_processing_stub = SpeechProcessingServiceStub(self.channel)
        
        logger.info(f"Connected to gRPC server at {server_address}")
    
    def close(self):
        """Close the gRPC channel"""
        if self.channel:
            self.channel.close()
            logger.info("Closed gRPC channel")
    
    def extract_text_from_image(self, image_data: bytes, image_format: str = "jpeg", enhance_image: bool = False) -> Dict[str, Any]:
        """
        Extract text from an image using OCR
        
        Args:
            image_data: Binary image data
            image_format: Image format (jpeg, png, etc.)
            enhance_image: Whether to enhance the image before OCR
            
        Returns:
            dict: Extracted text and metadata
        """
        try:
            # In a real implementation, you would create the appropriate request
            # request = TextExtractionRequest(
            #     image_data=image_data,
            #     image_format=image_format,
            #     enhance_image=enhance_image
            # )
            # response = self.text_processing_stub.ExtractText(request)
            
            # For now, return a mock response
            response = {
                "extracted_text": "Placeholder text extracted from image",
                "success": True,
                "error_message": "",
                "confidence_score": 0.95
            }
            return response
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return {
                "extracted_text": "",
                "success": False,
                "error_message": str(e),
                "confidence_score": 0.0
            }
    
    def convert_pdf_to_text(self, pdf_data: bytes, start_page: int = 0, end_page: int = -1, extract_tables: bool = False) -> Dict[str, Any]:
        """
        Convert PDF to text
        
        Args:
            pdf_data: Binary PDF data
            start_page: Starting page (0-indexed)
            end_page: Ending page (-1 for all pages)
            extract_tables: Whether to extract tables
            
        Returns:
            dict: Extracted text and table data
        """
        try:
            # In a real implementation, you would create the appropriate request
            # request = PDFConversionRequest(
            #     pdf_data=pdf_data,
            #     start_page=start_page,
            #     end_page=end_page,
            #     extract_tables=extract_tables
            # )
            # response = self.text_processing_stub.ConvertPDFToText(request)
            
            # For now, return a mock response
            response = {
                "extracted_text": "Placeholder text extracted from PDF",
                "success": True,
                "error_message": "",
                "table_data": ["Table 1 data", "Table 2 data"] if extract_tables else []
            }
            return response
            
        except Exception as e:
            logger.error(f"Error converting PDF to text: {str(e)}")
            return {
                "extracted_text": "",
                "success": False,
                "error_message": str(e),
                "table_data": []
            }
    
    def convert_data_format(self, input_data: str, input_format: str, output_format: str, options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Convert data between formats
        
        Args:
            input_data: Input data as string
            input_format: Source format (json, csv, xml, etc.)
            output_format: Target format
            options: Conversion options
            
        Returns:
            dict: Converted data and status
        """
        try:
            # In a real implementation, you would create the appropriate request
            # request = DataConversionRequest(
            #     input_data=input_data,
            #     input_format=input_format,
            #     output_format=output_format,
            #     conversion_options=options or {}
            # )
            # response = self.data_transformation_stub.ConvertData(request)
            
            # For now, return a mock response
            response = {
                "converted_data": f"Placeholder {input_format} to {output_format} conversion",
                "success": True,
                "error_message": ""
            }
            return response
            
        except Exception as e:
            logger.error(f"Error converting data format: {str(e)}")
            return {
                "converted_data": "",
                "success": False,
                "error_message": str(e)
            }
    
    def execute_python_script(self, script_content: str, input_data: str = "", parameters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute a Python script
        
        Args:
            script_content: Python script as string
            input_data: Input data for the script
            parameters: Additional parameters
            
        Returns:
            dict: Script output and execution info
        """
        try:
            # In a real implementation, you would create the appropriate request
            # request = PythonScriptRequest(
            #     script_content=script_content,
            #     input_data=input_data,
            #     parameters=parameters or {}
            # )
            # response = self.data_transformation_stub.ExecutePythonScript(request)
            
            # For now, return a mock response
            response = {
                "output_data": "Placeholder script execution result",
                "success": True,
                "error_message": "",
                "execution_time": 0.5
            }
            return response
            
        except Exception as e:
            logger.error(f"Error executing Python script: {str(e)}")
            return {
                "output_data": "",
                "success": False,
                "error_message": str(e),
                "execution_time": 0.0
            }
    
    def transcribe_speech(self, audio_data: bytes, audio_format: str = "wav", 
                         language_code: str = "en-US", enhanced_model: bool = False,
                         sample_rate_hertz: int = 16000, 
                         phrases: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Transcribe audio to text using Google Cloud Speech-to-Text
        
        Args:
            audio_data: Binary audio data
            audio_format: Audio format (wav, mp3, ogg, etc.)
            language_code: Language code (en-US, fr-FR, etc.)
            enhanced_model: Whether to use an enhanced recognition model
            sample_rate_hertz: Audio sample rate in Hertz
            phrases: List of phrases to boost recognition accuracy
            
        Returns:
            dict: Transcription and metadata
        """
        try:
            # In a real implementation, you would create the appropriate request
            # request = SpeechTranscriptionRequest(
            #     audio_data=audio_data,
            #     audio_format=audio_format,
            #     language_code=language_code,
            #     enhanced_model=enhanced_model,
            #     sample_rate_hertz=sample_rate_hertz,
            #     phrases=phrases or []
            # )
            # response = self.speech_processing_stub.TranscribeSpeech(request)
            
            # For now, return a mock response
            response = {
                "transcript": "Placeholder speech transcription",
                "success": True,
                "error_message": "",
                "confidence_score": 0.92,
                "alternatives": [
                    {"transcript": "Placeholder speech transcription", "confidence": 0.92},
                    {"transcript": "Place holder speech transcription", "confidence": 0.85}
                ],
                "duration_seconds": 5.2
            }
            return response
            
        except Exception as e:
            logger.error(f"Error transcribing speech: {str(e)}")
            return {
                "transcript": "",
                "success": False,
                "error_message": str(e),
                "confidence_score": 0.0,
                "alternatives": [],
                "duration_seconds": 0.0
            }
    
    def execute_python_file(self, file_path: str, input_data: str = "", 
                           arguments: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a Python file from a local path
        
        Args:
            file_path: Path to the Python file
            input_data: Input data to pass to the script's stdin
            arguments: Command line arguments to pass to the script
            
        Returns:
            dict: Script output and execution info
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Python file not found: {file_path}")
            
            # Build the command
            cmd = ["python", file_path]
            if arguments:
                cmd.extend(arguments)
            
            # Execute the script in a subprocess
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Provide input if needed
            stdout, stderr = process.communicate(input=input_data)
            
            execution_time = time.time() - start_time
            
            # Check if the command executed successfully
            if process.returncode != 0:
                logger.error(f"Error executing Python file: {stderr}")
                return {
                    "output_data": stdout,
                    "success": False,
                    "error_message": stderr,
                    "execution_time": execution_time,
                    "return_code": process.returncode
                }
            
            return {
                "output_data": stdout,
                "success": True,
                "error_message": stderr if stderr else "",
                "execution_time": execution_time,
                "return_code": process.returncode
            }
            
        except Exception as e:
            logger.error(f"Error executing Python file: {str(e)}")
            return {
                "output_data": "",
                "success": False,
                "error_message": str(e),
                "execution_time": 0.0,
                "return_code": -1
            }

# Singleton instance
_tools_client = None

def get_tools_client():
    """Get or create the tools client singleton"""
    global _tools_client
    if _tools_client is None:
        _tools_client = ToolsClient()
    return _tools_client 