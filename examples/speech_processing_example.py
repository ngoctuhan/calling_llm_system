#!/usr/bin/env python
"""
Example script demonstrating how to use the speech processing functionality
of the tools_engine gRPC service for speech-to-text transcription.
"""

import os
import sys
import base64
import argparse
from pprint import pprint

# Add the parent directory to the path so we can import from tools_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tools client
from tools_engine.client import ToolsClient

def read_audio_file(file_path):
    """Read an audio file and return its contents as bytes"""
    with open(file_path, 'rb') as f:
        return f.read()

def get_audio_format(file_path):
    """Get the audio format from the file extension"""
    _, ext = os.path.splitext(file_path)
    return ext.lstrip('.').lower()

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio to text using the tools_engine service')
    parser.add_argument('audio_file', help='Path to the audio file to transcribe')
    parser.add_argument('--language', default='en-US', help='Language code (default: en-US)')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced speech recognition model')
    parser.add_argument('--sample-rate', type=int, default=0, help='Sample rate in Hz (default: auto-detect)')
    parser.add_argument('--phrases', nargs='+', help='Boost recognition of specific phrases')
    args = parser.parse_args()
    
    # Check if the audio file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        return 1
    
    # Read the audio file
    try:
        audio_data = read_audio_file(args.audio_file)
        audio_format = get_audio_format(args.audio_file)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return 1
    
    # Create a client
    client = ToolsClient()
    
    print(f"Transcribing audio file: {args.audio_file}")
    print(f"File size: {len(audio_data)} bytes")
    print(f"Format: {audio_format}")
    print(f"Language: {args.language}")
    print(f"Enhanced model: {'Yes' if args.enhanced else 'No'}")
    if args.sample_rate:
        print(f"Sample rate: {args.sample_rate} Hz")
    if args.phrases:
        print(f"Boosted phrases: {', '.join(args.phrases)}")
    
    print("\nSending request to server...")
    
    # Call the service
    response = client.transcribe_speech(
        audio_data=audio_data,
        audio_format=audio_format,
        language_code=args.language,
        enhanced_model=args.enhanced,
        sample_rate_hertz=args.sample_rate,
        phrases=args.phrases or []
    )
    
    print("\nTranscription Results:")
    print("=====================")
    
    if response.get('success'):
        print(f"\nTranscript:\n{response.get('transcript')}")
        print(f"\nConfidence: {response.get('confidence_score', 0) * 100:.1f}%")
        print(f"Duration: {response.get('duration_seconds', 0):.2f} seconds")
        
        if response.get('alternatives'):
            print("\nAlternative transcripts:")
            for i, alt in enumerate(response.get('alternatives')):
                print(f"{i+1}. {alt.get('transcript')} (conf: {alt.get('confidence', 0) * 100:.1f}%)")
    else:
        print(f"Error: {response.get('error_message', 'Unknown error')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 