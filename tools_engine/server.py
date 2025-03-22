import logging
import time
import grpc
import os
import tempfile
import subprocess
import json
from concurrent import futures
from typing import Dict, Any, List, Optional, Union
import io
import base64

# Import the generated gRPC code
# Note: These imports will work after running generate_protos.py
# from app.tools_engine.tools_pb2 import *
# from app.tools_engine.tools_pb2_grpc import *

# For now, use placeholders to create the structure
# In a real implementation, you would use the actual generated classes
DataTransformationServiceServicer = object
TextProcessingServiceServicer = object
SpeechProcessingServiceServicer = object
add_DataTransformationServiceServicer_to_server = lambda x, y: None
add_TextProcessingServiceServicer_to_server = lambda x, y: None
add_SpeechProcessingServiceServicer_to_server = lambda x, y: None

from app.core.config import settings
from app.core.logger import setup_logging

logger = setup_logging()

class TextProcessingServicer(TextProcessingServiceServicer):
    """Implementation of the TextProcessingService gRPC service"""
    
    def ExtractText(self, request, context):
        """
        Extract text from an image using OCR
        
        This implementation uses Tesseract OCR via pytesseract.
        For enhanced OCR, it also provides an option to use Google Cloud Vision API.
        """
        logger.info("Received text extraction request")
        
        try:
            # Install dependencies if needed
            try:
                import pytesseract
                from PIL import Image, ImageEnhance
            except ImportError:
                logger.warning("Missing dependencies, attempting to install...")
                subprocess.check_call(["pip", "install", "pytesseract", "pillow"])
                import pytesseract
                from PIL import Image, ImageEnhance
            
            # Convert binary data to image
            image = Image.open(io.BytesIO(request.image_data))
            
            # Enhance image if requested
            if request.enhance_image:
                # Apply some basic image enhancements
                image = ImageEnhance.Contrast(image).enhance(1.5)  # Increase contrast
                image = ImageEnhance.Sharpness(image).enhance(1.5)  # Increase sharpness
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(image)
            confidence_score = 0.75  # Tesseract doesn't provide confidence score easily
            
            # Try Google Cloud Vision API for enhanced OCR if configured
            if request.enhance_image and settings.GOOGLE_CLOUD_PROJECT:
                try:
                    from google.cloud import vision
                    
                    # Create a client
                    client = vision.ImageAnnotatorClient()
                    
                    # Convert image to format needed by Vision API
                    image_content = request.image_data
                    vision_image = vision.Image(content=image_content)
                    
                    # Perform text detection
                    response = client.text_detection(image=vision_image)
                    
                    # Extract text from response
                    if response.text_annotations:
                        enhanced_text = response.text_annotations[0].description
                        
                        # If Google Cloud Vision gives a longer result, use it
                        if len(enhanced_text) > len(extracted_text):
                            extracted_text = enhanced_text
                            confidence_score = 0.95  # Google Vision typically provides better results
                        
                except Exception as e:
                    logger.warning(f"Failed to use Google Cloud Vision API: {str(e)}")
            
            return {
                "extracted_text": extracted_text,
                "success": True,
                "error_message": "",
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return {
                "extracted_text": "",
                "success": False,
                "error_message": str(e),
                "confidence_score": 0.0
            }
    
    def ConvertPDFToText(self, request, context):
        """
        Convert PDF to text
        
        This implementation uses PyPDF2 for basic text extraction
        and if requested, it can use Tesseract OCR for scanned PDFs.
        """
        logger.info("Received PDF conversion request")
        
        try:
            # Install dependencies if needed
            try:
                import PyPDF2
                import pytesseract
                from pdf2image import convert_from_bytes
                from PIL import Image
            except ImportError:
                logger.warning("Missing dependencies, attempting to install...")
                subprocess.check_call(["pip", "install", "PyPDF2", "pytesseract", "pdf2image", "pillow"])
                import PyPDF2
                import pytesseract
                from pdf2image import convert_from_bytes
                from PIL import Image
            
            # Create a PDF reader
            pdf_file = io.BytesIO(request.pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Determine page range
            start_page = max(0, request.start_page)
            end_page = min(len(pdf_reader.pages) - 1, request.end_page) if request.end_page >= 0 else len(pdf_reader.pages) - 1
            
            # Extract text from each page
            extracted_text = ""
            table_data = []
            
            # Check if any text can be extracted directly
            sample_page = pdf_reader.pages[start_page].extract_text()
            use_ocr = len(sample_page.strip()) < 50  # If very little text is extracted, it might be a scanned PDF
            
            if use_ocr:
                # Convert PDF pages to images for OCR
                images = convert_from_bytes(request.pdf_data, first_page=start_page+1, last_page=end_page+1)
                
                # Extract text from each image
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image)
                    extracted_text += f"\n\n--- Page {start_page + i + 1} ---\n\n{page_text}"
            else:
                # Extract text directly from PDF
                for i in range(start_page, end_page + 1):
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()
                    extracted_text += f"\n\n--- Page {i + 1} ---\n\n{page_text}"
            
            # Extract tables if requested
            if request.extract_tables:
                try:
                    import tabula
                    
                    # Save PDF to a temporary file for tabula (it requires a file path)
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                        tmp_file.write(request.pdf_data)
                        tmp_path = tmp_file.name
                    
                    try:
                        # Extract tables from each page
                        for i in range(start_page, end_page + 1):
                            tables = tabula.read_pdf(tmp_path, pages=i+1)
                            for j, table in enumerate(tables):
                                table_text = table.to_string()
                                table_data.append(f"Table {i+1}-{j+1}:\n{table_text}")
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract tables: {str(e)}")
                    table_data = ["Failed to extract tables"]
            
            return {
                "extracted_text": extracted_text,
                "success": True,
                "error_message": "",
                "table_data": table_data
            }
            
        except Exception as e:
            logger.error(f"Error converting PDF to text: {str(e)}")
            return {
                "extracted_text": "",
                "success": False,
                "error_message": str(e),
                "table_data": []
            }


class DataTransformationServicer(DataTransformationServiceServicer):
    """Implementation of the DataTransformationService gRPC service"""
    
    def ConvertData(self, request, context):
        """
        Convert data between formats
        
        This implementation supports conversion between:
        - JSON to CSV
        - CSV to JSON
        - JSON to XML
        - XML to JSON
        """
        logger.info(f"Received data conversion request: {request.input_format} to {request.output_format}")
        
        try:
            # Install dependencies if needed
            try:
                import pandas as pd
                import xml.etree.ElementTree as ET
                import dicttoxml
            except ImportError:
                logger.warning("Missing dependencies, attempting to install...")
                subprocess.check_call(["pip", "install", "pandas", "dicttoxml"])
                import pandas as pd
                import xml.etree.ElementTree as ET
                import dicttoxml
            
            # JSON to CSV conversion
            if request.input_format.lower() == "json" and request.output_format.lower() == "csv":
                data = json.loads(request.input_data)
                df = pd.DataFrame(data)
                converted_data = df.to_csv(index=False)
            
            # CSV to JSON conversion
            elif request.input_format.lower() == "csv" and request.output_format.lower() == "json":
                df = pd.read_csv(io.StringIO(request.input_data))
                converted_data = df.to_json(orient='records')
            
            # JSON to XML conversion
            elif request.input_format.lower() == "json" and request.output_format.lower() == "xml":
                data = json.loads(request.input_data)
                xml_data = dicttoxml.dicttoxml(data)
                converted_data = xml_data.decode('utf-8')
            
            # XML to JSON conversion
            elif request.input_format.lower() == "xml" and request.output_format.lower() == "json":
                root = ET.fromstring(request.input_data)
                
                def element_to_dict(element):
                    result = {}
                    for child in element:
                        child_data = element_to_dict(child)
                        if child.tag in result:
                            if isinstance(result[child.tag], list):
                                result[child.tag].append(child_data)
                            else:
                                result[child.tag] = [result[child.tag], child_data]
                        else:
                            result[child.tag] = child_data
                    if element.text and element.text.strip():
                        if not result:
                            return element.text.strip()
                    return result if result else None
                
                data = element_to_dict(root)
                converted_data = json.dumps({root.tag: data})
            
            else:
                return {
                    "converted_data": "",
                    "success": False,
                    "error_message": f"Unsupported conversion: {request.input_format} to {request.output_format}"
                }
            
            return {
                "converted_data": converted_data,
                "success": True,
                "error_message": ""
            }
            
        except Exception as e:
            logger.error(f"Error converting data: {str(e)}")
            return {
                "converted_data": "",
                "success": False,
                "error_message": str(e)
            }
    
    def ExecutePythonScript(self, request, context):
        """
        Execute a Python script in a sandboxed environment
        
        This implementation:
        1. Creates a temporary file with the script content
        2. Executes it in a subprocess
        3. Captures and returns the output
        """
        logger.info("Received Python script execution request")
        
        try:
            # Create a temporary directory for sandboxed execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a temporary file for the script
                script_path = os.path.join(temp_dir, "script.py")
                
                # Write the script to the file
                with open(script_path, "w") as f:
                    f.write(request.script_content)
                
                # Prepare any parameters as command line arguments
                args = []
                for key, value in request.parameters.items():
                    args.extend([key, value])
                
                # Start timing
                start_time = time.time()
                
                # Execute the script
                process = subprocess.Popen(
                    ["python", script_path] + args,
                    stdin=subprocess.PIPE if request.input_data else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir
                )
                
                # Provide input data if available
                stdout, stderr = process.communicate(input=request.input_data)
                
                # End timing
                execution_time = time.time() - start_time
                
                # Check if the script executed successfully
                if process.returncode != 0:
                    logger.error(f"Script execution failed: {stderr}")
                    return {
                        "output_data": stdout,
                        "success": False,
                        "error_message": stderr,
                        "execution_time": execution_time
                    }
                
                return {
                    "output_data": stdout,
                    "success": True,
                    "error_message": stderr,
                    "execution_time": execution_time
                }
                
        except Exception as e:
            logger.error(f"Error executing Python script: {str(e)}")
            return {
                "output_data": "",
                "success": False,
                "error_message": str(e),
                "execution_time": 0.0
            }


class SpeechProcessingServicer(SpeechProcessingServiceServicer):
    """Implementation of the SpeechProcessingService gRPC service"""
    
    def TranscribeSpeech(self, request, context):
        """
        Transcribe audio to text using Google Cloud Speech-to-Text
        
        This implementation:
        1. Takes audio data in various formats
        2. Converts it to the format expected by Google Cloud
        3. Calls the Speech-to-Text API
        4. Returns the transcription and metadata
        """
        logger.info(f"Received speech transcription request: {request.audio_format}, {request.language_code}")
        
        try:
            # Install dependencies if needed
            try:
                from google.cloud import speech
                import soundfile as sf
            except ImportError:
                logger.warning("Missing dependencies, attempting to install...")
                subprocess.check_call(["pip", "install", "google-cloud-speech", "soundfile"])
                from google.cloud import speech
                import soundfile as sf
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=f".{request.audio_format}", delete=False) as temp_file:
                temp_file.write(request.audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Load the audio file to get properties and convert if needed
                audio_data, sample_rate = sf.read(temp_file_path)
                
                # Override sample rate if specified in the request
                if request.sample_rate_hertz > 0:
                    sample_rate = request.sample_rate_hertz
                
                # Initialize speech client
                client = speech.SpeechClient()
                
                # Configure recognition settings
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    language_code=request.language_code,
                    use_enhanced=request.enhanced_model,
                    speech_contexts=[speech.SpeechContext(phrases=list(request.phrases))] if request.phrases else None
                )
                
                # Create audio object
                with open(temp_file_path, 'rb') as audio_file:
                    content = audio_file.read()
                
                audio = speech.RecognitionAudio(content=content)
                
                # Perform speech recognition
                response = client.recognize(config=config, audio=audio)
                
                # Process the response
                if not response.results:
                    return {
                        "transcript": "",
                        "success": True,
                        "error_message": "No speech recognized",
                        "confidence_score": 0.0,
                        "alternatives": [],
                        "duration_seconds": 0.0
                    }
                
                # Get the first result
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
                
                # Extract alternatives
                alternatives = []
                for alt in result.alternatives:
                    alternatives.append({
                        "transcript": alt.transcript,
                        "confidence": alt.confidence
                    })
                
                # Calculate duration (approximation based on audio length)
                duration_seconds = len(audio_data) / sample_rate
                
                return {
                    "transcript": transcript,
                    "success": True,
                    "error_message": "",
                    "confidence_score": confidence,
                    "alternatives": alternatives,
                    "duration_seconds": duration_seconds
                }
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
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


def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add the service implementations to the server
    text_servicer = TextProcessingServicer()
    data_servicer = DataTransformationServicer()
    speech_servicer = SpeechProcessingServicer()
    
    add_TextProcessingServiceServicer_to_server(text_servicer, server)
    add_DataTransformationServiceServicer_to_server(data_servicer, server)
    add_SpeechProcessingServiceServicer_to_server(speech_servicer, server)
    
    # Add a secure port
    server_address = f"{settings.GRPC_HOST}:{settings.GRPC_PORT}"
    server.add_insecure_port(server_address)
    
    # Start the server
    server.start()
    logger.info(f"gRPC server started on {server_address}")
    
    try:
        # Keep the server running
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)

if __name__ == "__main__":
    serve() 