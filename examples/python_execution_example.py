#!/usr/bin/env python
"""
Example script demonstrating how to use the Python script execution functionality
of the tools_engine gRPC service to execute Python code dynamically.
"""

import os
import sys
import argparse
from pprint import pprint

# Add the parent directory to the path so we can import from tools_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tools client
from tools_engine.client import ToolsClient

def read_file(file_path):
    """Read a file and return its contents as string"""
    with open(file_path, 'r') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description='Execute Python scripts using the tools_engine service')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Execute Python script content
    execute_parser = subparsers.add_parser('execute', help='Execute Python script content')
    execute_parser.add_argument('--script', required=True, help='Python script content or file path')
    execute_parser.add_argument('--is-file', action='store_true', help='Script argument is a file path')
    execute_parser.add_argument('--input', help='Input data to pass to the script')
    execute_parser.add_argument('--params', nargs='+', help='Parameters in format key=value')
    
    # Execute Python file
    execute_file_parser = subparsers.add_parser('execute-file', help='Execute a Python file')
    execute_file_parser.add_argument('file', help='Path to the Python file to execute')
    execute_file_parser.add_argument('--input', help='Input data to pass to the script')
    execute_file_parser.add_argument('--args', nargs='+', help='Command line arguments to pass to the script')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create a client
    client = ToolsClient()
    
    if args.command == 'execute':
        # If script is a file path and --is-file is set, read the file
        script_content = args.script
        if args.is_file:
            if not os.path.isfile(args.script):
                print(f"Error: Script file '{args.script}' not found.")
                return 1
            script_content = read_file(args.script)
        
        # Parse parameters
        params = {}
        if args.params:
            for param in args.params:
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key] = value
                else:
                    print(f"Warning: Ignoring invalid parameter format: {param}")
        
        print("Executing Python script:")
        print("=======================")
        print(f"Script length: {len(script_content)} characters")
        if params:
            print(f"Parameters: {params}")
        if args.input:
            print(f"Input data: {args.input}")
        
        print("\nSending request to server...")
        
        # Call the service
        response = client.execute_python_script(
            script_content=script_content,
            input_data=args.input or "",
            parameters=params
        )
        
        print("\nExecution Results:")
        print("=================")
        if response.get('success'):
            print(f"\nOutput:\n{response.get('output_data')}")
            print(f"\nExecution time: {response.get('execution_time', 0):.3f} seconds")
            if response.get('error_message'):
                print(f"\nStandard Error:\n{response.get('error_message')}")
        else:
            print(f"Error: {response.get('error_message', 'Unknown error')}")
            if response.get('output_data'):
                print(f"\nPartial output:\n{response.get('output_data')}")
    
    elif args.command == 'execute-file':
        if not os.path.isfile(args.file):
            print(f"Error: Python file '{args.file}' not found.")
            return 1
        
        print("Executing Python file:")
        print("=====================")
        print(f"File: {args.file}")
        if args.args:
            print(f"Arguments: {args.args}")
        if args.input:
            print(f"Input data: {args.input}")
        
        print("\nSending request to server...")
        
        # Call the service
        response = client.execute_python_file(
            file_path=args.file,
            input_data=args.input or "",
            args=args.args or []
        )
        
        print("\nExecution Results:")
        print("=================")
        if response.get('success'):
            print(f"\nOutput:\n{response.get('output_data')}")
            print(f"\nExecution time: {response.get('execution_time', 0):.3f} seconds")
            if response.get('error_message'):
                print(f"\nStandard Error:\n{response.get('error_message')}")
        else:
            print(f"Error: {response.get('error_message', 'Unknown error')}")
            if response.get('output_data'):
                print(f"\nPartial output:\n{response.get('output_data')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 