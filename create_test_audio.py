import torch
import torchaudio
import os

def create_dummy_audio_files():
    """Create dummy audio files for testing"""
    os.makedirs("/tmp", exist_ok=True)
    
    for i in range(30):  # Create enough for train + val
        # Generate random audio: 1 second of noise at 16kHz
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        
        # Generate random audio data
        audio_data = torch.randn(samples) * 0.1  # Small amplitude to avoid clipping
        
        # Save as WAV file
        filename = f"/tmp/sample_audio_{i}.wav"
        torchaudio.save(filename, audio_data.unsqueeze(0), sample_rate)
        print(f"Created {filename}")
    
    print("Dummy audio files created successfully!")

if __name__ == "__main__":
    create_dummy_audio_files()
