from torch.utils.data import Dataset
import json
import os
from .preprocess import load_audio

class SERDataset(Dataset):
    def __init__(self, manifest_path):
        """
        manifest_path: jsonl file, each line: {"audio" or "audio_path": path, "text": str (optional), "label": int}
        """
        self.items = []
        with open(manifest_path) as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_key = 'audio_path' if 'audio_path' in item else 'audio'
        path = item[audio_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest references missing file: {path}")
        audio = load_audio(path)
        text = item.get('text', '')
        label = item['label']
        return audio, text, label
