import os
import json
import glob
from typing import Dict, List, Tuple
import re


class SERDatasetLoader:
    """Loader for multiple SER datasets with unified label mapping"""
    
    # Unified emotion labels (mapping to 0-5 for 6-class CREMA model)
    # CREMA emotions: 0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Fear, 5=Disgust
    EMOTION_MAP = {
        # RAVDESS emotions (mapped to 6-class system)
        '01': 0,  # neutral
        '02': 1,  # calm -> happy
        '03': 1,  # happy
        '04': 2,  # sad
        '05': 3,  # angry
        '06': 4,  # fearful
        '07': 5,  # disgust
        '08': 1,  # surprised -> happy
        
        # CREMA-D emotions (from filename) - 6 classes
        'NEU': 0,  # neutral
        'HAP': 1,  # happy
        'SAD': 2,  # sad
        'ANG': 3,  # angry
        'FEA': 4,  # fearful
        'DIS': 5,  # disgust
        
        # Unified mapping to 6 classes
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'disgust': 5,
    }
    
    # Map emotions to 6-class system (for CREMA compatibility)
    # 0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Fear, 5=Disgust
    EMOTION_TO_6CLASS = {
        0: 0,  # neutral -> neutral
        1: 1,  # calm -> happy
        2: 1,  # happy -> happy
        3: 2,  # sad -> sad
        4: 3,  # angry -> angry
        5: 4,  # fearful -> fearful
        6: 5,  # disgust -> disgust
        7: 1,  # surprised -> happy
    }
    
    def __init__(self, datasets_path: str = "datasets"):
        self.datasets_path = datasets_path
        
    def load_ravdess(self, split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Load RAVDESS dataset"""
        print("Loading RAVDESS dataset...")
        samples = []
        
        # RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        # Example: 03-01-03-01-02-02-01.wav
        # 03 = modality (03 = audio)
        # 01 = vocal channel (01 = speech)
        # 03 = emotion (03 = happy)
        # 01 = intensity (01 = normal, 02 = strong)
        # 02 = statement (01-02)
        # 02 = repetition (01-02)
        # 01 = actor (01-24)
        
        actor_dirs = glob.glob(os.path.join(self.datasets_path, "ravdess", "Actor_*"))
        
        for actor_dir in actor_dirs:
            actor = os.path.basename(actor_dir)
            wav_files = glob.glob(os.path.join(actor_dir, "*.wav"))
            
            for wav_file in wav_files:
                filename = os.path.basename(wav_file)
                parts = filename.replace('.wav', '').split('-')
                
                if len(parts) == 7:
                    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor_num = parts
                    
                    # Only use speech modality
                    if modality == '03' and vocal_channel == '01':
                        emotion_label = self.EMOTION_MAP.get(emotion_code, 0)
                        # Map to 6-class
                        emotion_label = self.EMOTION_TO_6CLASS.get(emotion_label, 0)
                        
                        # Create dummy transcript (in real scenario, you'd use ASR)
                        transcript = f"Statement {statement} by actor {actor_num}"
                        
                        samples.append({
                            'audio': wav_file,
                            'text': transcript,
                            'label': emotion_label,
                            'dataset': 'ravdess',
                            'actor': actor_num,
                            'intensity': intensity
                        })
        
        print(f"Loaded {len(samples)} RAVDESS samples")
        return self._split_samples(samples, split_ratio)
    
    def load_crema(self, split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Load CREMA-D dataset"""
        print("Loading CREMA-D dataset...")
        samples = []
        
        # CREMA-D filename format: ID_Modality_VocalChannel_Emotion_Intensity_Statement_Repetition_Actors.wav
        # Example: 1091_TSI_HAP_XX.wav
        # 1091 = ID
        # TSI = Modality_VocalChannel (TSI = Text_Speech_Intensity)
        # HAP = Emotion (HAP, SAD, ANG, FEA, DIS, NEU)
        # XX = Intensity (XX = normal, LO = low, MD = medium, HI = high)
        # (Statement and Repetition are not in filename)
        
        wav_files = glob.glob(os.path.join(self.datasets_path, "crema", "AudioWAV", "*.wav"))
        
        for wav_file in wav_files:
            # Check file size - skip files smaller than 1KB (likely corrupted)
            if os.path.getsize(wav_file) < 1024:
                continue
                
            filename = os.path.basename(wav_file)
            parts = filename.replace('.wav', '').split('_')
            
            if len(parts) >= 3:
                speaker_id, modality_channel, emotion = parts[0], parts[1], parts[2]
                
                # Map emotion to label
                emotion_label = self.EMOTION_MAP.get(emotion, 0)
                # Map to 6-class
                emotion_label = self.EMOTION_TO_6CLASS.get(emotion_label, 0)
                
                # Create dummy transcript
                transcript = f"Speaker {speaker_id} utterance"
                
                samples.append({
                    'audio': wav_file,
                    'text': transcript,
                    'label': emotion_label,
                    'dataset': 'crema',
                    'speaker': speaker_id,
                    'modality': modality_channel
                })
        
        print(f"Loaded {len(samples)} CREMA-D samples (filtered corrupted files)")
        return self._split_samples(samples, split_ratio)
    
    def load_iemocap(self, split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Load IEMOCAP dataset (placeholder - implement based on your IEMOCAP structure)"""
        print("Loading IEMOCAP dataset...")
        samples = []
        
        # IEMOCAP structure varies - implement based on your specific format
        # This is a placeholder
        iemocap_path = os.path.join(self.datasets_path, "iemocap")
        if os.path.exists(iemocap_path):
            print("IEMOCAP dataset found but loader not implemented yet")
            # TODO: Implement IEMOCAP loading based on your structure
        
        return self._split_samples(samples, split_ratio)
    
    def _split_samples(self, samples: List[Dict], split_ratio: float) -> Tuple[List[Dict], List[Dict]]:
        """Split samples into train/val based on speaker/actor to avoid data leakage"""
        # Group by speaker/actor
        speaker_groups = {}
        for sample in samples:
            speaker = sample.get('actor', sample.get('speaker', 'unknown'))
            if speaker not in speaker_groups:
                speaker_groups[speaker] = []
            speaker_groups[speaker].append(sample)
        
        # Split speakers
        speakers = list(speaker_groups.keys())
        split_idx = int(len(speakers) * split_ratio)
        train_speakers = speakers[:split_idx]
        val_speakers = speakers[split_idx:]
        
        # Create train/val splits
        train_samples = []
        val_samples = []
        
        for speaker in train_speakers:
            train_samples.extend(speaker_groups[speaker])
        
        for speaker in val_speakers:
            val_samples.extend(speaker_groups[speaker])
        
        print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")
        return train_samples, val_samples
    
    def create_manifests(self, output_dir: str = ".", split_ratio: float = 0.8):
        """Create train and validation manifests from all datasets"""
        print("Creating manifests from all datasets...")
        
        all_train = []
        all_val = []
        
        # Load all datasets
        datasets = [
            ('ravdess', self.load_ravdess),
            ('crema', self.load_crema),
            ('iemocap', self.load_iemocap),
        ]
        
        for dataset_name, loader_func in datasets:
            try:
                train_samples, val_samples = loader_func(split_ratio)
                all_train.extend(train_samples)
                all_val.extend(val_samples)
                print(f"Added {len(train_samples)} train, {len(val_samples)} val from {dataset_name}")
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
        
        # Save manifests
        train_manifest = os.path.join(output_dir, "train_manifest.jsonl")
        val_manifest = os.path.join(output_dir, "val_manifest.jsonl")
        
        with open(train_manifest, 'w') as f:
            for sample in all_train:
                f.write(json.dumps(sample) + '\n')
        
        with open(val_manifest, 'w') as f:
            for sample in all_val:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Created manifests:")
        print(f"  Train: {train_manifest} ({len(all_train)} samples)")
        print(f"  Val: {val_manifest} ({len(all_val)} samples)")
        
        # Print label distribution
        self._print_label_distribution(all_train, "Train")
        self._print_label_distribution(all_val, "Val")
        
        return train_manifest, val_manifest
    
    def _print_label_distribution(self, samples: List[Dict], split_name: str):
        """Print label distribution for debugging"""
        label_counts = {}
        for sample in samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"{split_name} label distribution:")
        for label, count in sorted(label_counts.items()):
            emotion_name = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 4: 'fear', 5: 'disgust'}.get(label, f'unknown_{label}')
            print(f"  {emotion_name}: {count}")


if __name__ == "__main__":
    loader = SERDatasetLoader()
    loader.create_manifests()
