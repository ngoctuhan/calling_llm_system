import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools_engine.server import DataTransformationServicer


class TestPythonExecution(unittest.TestCase):
    """Test cases for the Python script execution functionality"""
    
    def setUp(self):
        """Set up the test case"""
        self.data_servicer = DataTransformationServicer()
        
        # Create a temporary directory for files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir.name
    
    def tearDown(self):
        """Clean up after the test"""
        self.temp_dir.cleanup()
    
    def test_execute_python_script_simple(self):
        """Test executing a simple Python script"""
        # Create a mock request
        mock_request = MagicMock()
        mock_request.script_content = "print('Hello, world!')"
        mock_request.input_data = ""
        mock_request.parameters = {}
        
        # Call the service method
        response = self.data_servicer.ExecutePythonScript(mock_request, MagicMock())
        
        # Verify the response
        self.assertTrue(response.get("success"))
        self.assertEqual(response.get("output_data").strip(), "Hello, world!")
        self.assertTrue(response.get("execution_time") > 0)
    
    def test_execute_python_script_with_input(self):
        """Test executing a Python script with input data"""
        # Create a mock request
        mock_request = MagicMock()
        mock_request.script_content = "import sys\nprint(input('Enter name: '))\nprint('Done')"
        mock_request.input_data = "Test User"
        mock_request.parameters = {}
        
        # Call the service method
        response = self.data_servicer.ExecutePythonScript(mock_request, MagicMock())
        
        # Verify the response
        self.assertTrue(response.get("success"))
        self.assertEqual(response.get("output_data").strip(), "Test User\nDone")
    
    def test_execute_python_script_with_params(self):
        """Test executing a Python script with parameters"""
        # Create a mock request
        mock_request = MagicMock()
        mock_request.script_content = "import sys\nprint(sys.argv[1:])"
        mock_request.input_data = ""
        mock_request.parameters = {"--name": "Test User", "--age": "30"}
        
        # Call the service method
        response = self.data_servicer.ExecutePythonScript(mock_request, MagicMock())
        
        # Verify the response
        self.assertTrue(response.get("success"))
        # The output will be the command line arguments as a list
        self.assertTrue("--name" in response.get("output_data"))
        self.assertTrue("Test User" in response.get("output_data"))
        self.assertTrue("--age" in response.get("output_data"))
        self.assertTrue("30" in response.get("output_data"))
    
    def test_execute_python_script_with_error(self):
        """Test executing a Python script that raises an error"""
        # Create a mock request with a script that raises an error
        mock_request = MagicMock()
        mock_request.script_content = "print(1/0)"  # Division by zero error
        mock_request.input_data = ""
        mock_request.parameters = {}
        
        # Call the service method
        response = self.data_servicer.ExecutePythonScript(mock_request, MagicMock())
        
        # Verify the response
        self.assertFalse(response.get("success"))
        self.assertEqual(response.get("output_data"), "")
        self.assertTrue("ZeroDivisionError" in response.get("error_message"))
    
    def test_execute_complex_script(self):
        """Test executing a more complex Python script"""
        # Script that computes Fibonacci numbers
        fib_script = """
import sys
import json

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

# Get number from command line args or use default
n = 10
if len(sys.argv) > 1:
    try:
        n = int(sys.argv[1])
    except ValueError:
        pass

# Compute and print Fibonacci numbers
fib_numbers = fibonacci(n)
print(json.dumps({"fibonacci": fib_numbers}))
"""
        
        # Create a mock request
        mock_request = MagicMock()
        mock_request.script_content = fib_script
        mock_request.input_data = ""
        mock_request.parameters = {"15": ""}  # Pass 15 as an argument
        
        # Call the service method
        response = self.data_servicer.ExecutePythonScript(mock_request, MagicMock())
        
        # Verify the response
        self.assertTrue(response.get("success"))
        self.assertTrue("fibonacci" in response.get("output_data"))
        self.assertTrue("610" in response.get("output_data"))  # 15th Fibonacci number is 610


if __name__ == '__main__':
    unittest.main() 