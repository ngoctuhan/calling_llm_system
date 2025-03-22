# Tools Engine

The Tools Engine is a gRPC-based service that provides various utilities for text processing, data transformation, and speech-to-text capabilities. It serves as a microservice within the larger Q&A system architecture, providing specialized functionality that can be consumed by other components.

## Features

The Tools Engine offers the following services:

### Text Processing Service

- **Extract Text from Images**: Uses OCR (Optical Character Recognition) to extract text from images, with optional image enhancement.
- **Convert PDF to Text**: Extracts text and tables from PDF documents, supporting both native PDFs and scanned documents through OCR.

### Data Transformation Service

- **Convert Data Formats**: Transforms data between different formats (JSON, CSV, XML).
- **Execute Python Scripts**: Runs Python scripts in a secure sandbox, providing input data and capturing the output.
- **Execute Python Files**: Executes Python files with command-line arguments and input data.

### Speech Processing Service

- **Transcribe Speech**: Converts audio to text using Google Cloud Speech-to-Text, supporting various audio formats and languages.

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Generate the gRPC code from the proto definitions:
   ```
   python tools_engine/generate_protos.py
   ```

## Configuration

The Tools Engine relies on configuration settings defined in `app/core/config.py`. Make sure the following settings are available:

- `GRPC_HOST`: Host address for the gRPC server (default: "localhost")
- `GRPC_PORT`: Port for the gRPC server (default: 50051)
- `DEBUG`: Enable debug mode (affects logging)
- `GOOGLE_CLOUD_PROJECT`: (Optional) Google Cloud project ID for enhanced OCR and speech-to-text
- `GOOGLE_APPLICATION_CREDENTIALS`: (Optional) Path to Google Cloud credentials file

## Running the Server

To start the gRPC server:

```bash
python tools_engine/server.py
```

The server will listen on the configured host and port.

## Client Usage

The Tools Engine provides a `ToolsClient` class that makes it easy to interact with the gRPC services. Here's a quick example:

```python
from tools_engine.client import ToolsClient

# Create a client
client = ToolsClient()

# Extract text from an image
result = client.extract_text_from_image(image_data=image_bytes, enhance_image=True)
print(f"Extracted text: {result['extracted_text']}")

# Convert PDF to text
pdf_result = client.convert_pdf_to_text(pdf_data=pdf_bytes, start_page=0, end_page=5, extract_tables=True)
print(f"PDF text: {pdf_result['extracted_text']}")
print(f"Tables: {pdf_result['table_data']}")

# Convert data format
conversion = client.convert_data_format(input_data=json_data, input_format="json", output_format="csv")
print(f"Converted data: {conversion['converted_data']}")

# Execute a Python script
script_result = client.execute_python_script(
    script_content="import sys\nprint('Hello, world!')\nprint(sys.argv)",
    parameters={"--arg1": "value1"}
)
print(f"Script output: {script_result['output_data']}")

# Transcribe speech
transcription = client.transcribe_speech(
    audio_data=audio_bytes,
    audio_format="wav",
    language_code="en-US",
    enhanced_model=True
)
print(f"Transcript: {transcription['transcript']}")
```

## Example Scripts

The `examples` directory contains scripts demonstrating how to use each service:

- `speech_processing_example.py`: Demonstrates speech-to-text capabilities
- `python_execution_example.py`: Shows how to execute Python scripts and files

To run an example:

```bash
python examples/speech_processing_example.py audio_file.wav
python examples/python_execution_example.py execute --script "print('Hello, world!')"
```

## Development

### Adding New Services

To add a new service:

1. Define the service in `tools_engine/protos/tools.proto`
2. Generate the gRPC code using `generate_protos.py`
3. Implement the service in `server.py`
4. Add client methods in `client.py`

### Testing

Run unit tests with:

```bash
pytest tools_engine/tests/
```

## Architecture

The Tools Engine uses a gRPC-based client-server architecture:

1. **Proto Definitions**: Service interfaces are defined in `.proto` files using Protocol Buffers.
2. **Server**: Implements the services defined in the proto files.
3. **Client**: Provides a Python API for interacting with the server.
4. **Configuration**: Configures logging, server settings, and external services.

## License

[MIT License](LICENSE) 