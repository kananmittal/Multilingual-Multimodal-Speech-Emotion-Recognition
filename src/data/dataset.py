from torch.utils.data import Dataset
import json
from .preprocess import load_audio

class SERDataset(Dataset):
    def __init__(self, manifest_path):
        """
        manifest_path: jsonl file, each line: {"audio": path, "text": "...", "label": int}
        """
        self.items = []
        with open(manifest_path) as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio = load_audio(item['audio'])
        text = item['text']
        label = item['label']
        return audio, text, label
