import os
import unittest
import tempfile
import wave
import struct
import numpy as np
from unittest.mock import MagicMock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools_engine.server import SpeechProcessingServicer


class TestSpeechProcessing(unittest.TestCase):
    """Test cases for the SpeechProcessingService"""
    
    def setUp(self):
        """Set up the test case"""
        self.speech_servicer = SpeechProcessingServicer()
        
        # Create a simple WAV file for testing
        self.temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_wav_path = self.temp_wav.name
        self._create_test_wav(self.temp_wav_path)
    
    def tearDown(self):
        """Clean up after the test"""
        if os.path.exists(self.temp_wav_path):
            os.unlink(self.temp_wav_path)
    
    def _create_test_wav(self, file_path, duration=1.0, sample_rate=16000):
        """Create a simple test WAV file with a sine wave"""
        amp = 32767  # Max amplitude for 16-bit audio
        freq = 440.0  # Frequency of the sine wave (A4 note)
        
        # Generate a sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = amp * np.sin(2 * np.pi * freq * t)
        
        # Convert to 16-bit PCM
        audio_data = audio_data.astype(np.int16)
        
        # Create a WAV file
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write frames
            for sample in audio_data:
                wav_file.writeframes(struct.pack('<h', int(sample)))
    
    @patch('google.cloud.speech.SpeechClient')
    @patch('soundfile.read')
    def test_transcribe_speech(self, mock_sf_read, mock_speech_client):
        """Test the TranscribeSpeech method"""
        # Read the test WAV file
        with open(self.temp_wav_path, 'rb') as f:
            audio_data = f.read()
        
        # Create a mock request
        mock_request = MagicMock()
        mock_request.audio_data = audio_data
        mock_request.audio_format = "wav"
        mock_request.language_code = "en-US"
        mock_request.enhanced_model = True
        mock_request.sample_rate_hertz = 16000
        mock_request.phrases = ["hello", "world"]
        
        # Mock soundfile.read to return sample data
        sample_array = np.ones(16000, dtype=np.float32)  # 1 second of audio
        mock_sf_read.return_value = (sample_array, 16000)
        
        # Mock the speech client's recognize method
        mock_client_instance = mock_speech_client.return_value
        mock_recognize = mock_client_instance.recognize
        
        # Create mock response
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "Hello, world!"
        mock_alternative.confidence = 0.95
        mock_result.alternatives = [mock_alternative]
        
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        
        # Set up the mock recognize method to return our mock response
        mock_recognize.return_value = mock_response
        
        # Call the service method
        response = self.speech_servicer.TranscribeSpeech(mock_request, MagicMock())
        
        # Verify the response
        self.assertEqual(response.get("transcript"), "Hello, world!")
        self.assertEqual(response.get("confidence_score"), 0.95)
        self.assertTrue(response.get("success"))
        self.assertEqual(len(response.get("alternatives")), 1)
        self.assertEqual(response.get("alternatives")[0].get("transcript"), "Hello, world!")
        
        # Verify the correct calls were made
        mock_speech_client.assert_called_once()
        mock_recognize.assert_called_once()
        
        # Verify the configuration was correct
        call_args = mock_recognize.call_args
        self.assertIn('config', call_args[1])
        self.assertIn('audio', call_args[1])
        
        config = call_args[1]['config']
        self.assertEqual(config.language_code, "en-US")
        self.assertEqual(config.sample_rate_hertz, 16000)
        self.assertTrue(config.use_enhanced)


if __name__ == '__main__':
    unittest.main() 