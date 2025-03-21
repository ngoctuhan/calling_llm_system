#!/usr/bin/env python
import os
import subprocess
import sys

def generate_proto_code():
    """Generate Python code from protobuf definitions"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    proto_dir = os.path.join(current_dir, "protos")
    
    # Get a list of all .proto files
    proto_files = [f for f in os.listdir(proto_dir) if f.endswith('.proto')]
    
    if not proto_files:
        print("No .proto files found in the protos directory.")
        return
    
    # Create output directories
    os.makedirs(os.path.join(current_dir, "generated"), exist_ok=True)
    
    # Generate Python code for each .proto file
    for proto_file in proto_files:
        proto_path = os.path.join(proto_dir, proto_file)
        output_dir = current_dir
        
        print(f"Generating Python code for {proto_file}...")
        
        try:
            # Run protoc command
            subprocess.run([
                "python", "-m", "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                proto_path
            ], check=True)
            
            print(f"Successfully generated code for {proto_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error generating code for {proto_file}: {e}")
            sys.exit(1)
    
    print("All protobuf code generated successfully.")

if __name__ == "__main__":
    generate_proto_code() 