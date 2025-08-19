import json
import os

# Create sample manifests for testing
# In practice, you'd replace these with real audio files and transcripts

def create_sample_manifest(filename, num_samples=10):
    """Create a sample manifest with dummy data for testing"""
    samples = []
    for i in range(num_samples):
        # Dummy audio path - replace with real paths
        audio_path = f"/tmp/sample_audio_{i}.wav"
        # Dummy transcript
        transcript = f"This is sample utterance number {i} for testing"
        # Random label 0-3 (angry, happy, sad, neutral)
        label = i % 4
        
        sample = {
            "audio": audio_path,
            "text": transcript,
            "label": label
        }
        samples.append(sample)
    
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {filename} with {num_samples} samples")

if __name__ == "__main__":
    create_sample_manifest("train_manifest.jsonl", 20)
    create_sample_manifest("val_manifest.jsonl", 10)
    print("Sample manifests created. Replace audio paths with real files before training.")
