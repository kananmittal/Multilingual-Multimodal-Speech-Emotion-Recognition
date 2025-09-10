import os
import json
from pathlib import Path
from typing import Dict, List, Set


DATA_ROOT = Path("datasets/IEMOCAP")
# Sessions 1-4 -> train, Session 5 -> val (common split)
TRAIN_SESSIONS = {"Session1", "Session2", "Session3", "Session4"}
VAL_SESSIONS = {"Session5"}

# Build a global label map dynamically so we include ALL emotions present
EMOTION_TO_ID: Dict[str, int] = {}


def is_wav_file(file_name: str) -> bool:
    return file_name.lower().endswith(".wav")


def parse_dialog_label_file(label_file: Path):
    # Returns mapping: utterance_id -> emotion_tag (string)
    # Lines look like: [start - end]\tSes01F_impro01_F000\tneu\t[2.5,2.5,2.5]
    mapping = {}
    try:
        with open(label_file, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("["):
                    continue
                # Split on whitespace/tabs
                parts = line.split()
                if len(parts) < 3:
                    continue
                # parts[0] is [start, parts[1] might be -, keep robust by finding the utterance token starting with Ses
                utt_id = None
                emo = None
                # Find first token that startswith Ses and next token after that is emotion
                for i, token in enumerate(parts):
                    if token.startswith("Ses"):
                        utt_id = token
                        # Emotion is next non-bracket token
                        if i + 1 < len(parts):
                            emo = parts[i + 1]
                        break
                if not utt_id or not emo:
                    continue
                emo = emo.strip().lower()
                mapping[utt_id] = emo
    except FileNotFoundError:
        return {}
    return mapping


def build_label_map() -> Dict[str, int]:
    global EMOTION_TO_ID
    unique_emotions: Set[str] = set()
    for session_dir in DATA_ROOT.glob("Session*/"):
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        if not eval_dir.exists():
            continue
        for txt in eval_dir.glob("*.txt"):
            utt_to_emo = parse_dialog_label_file(txt)
            unique_emotions.update(utt_to_emo.values())
    emotions_sorted: List[str] = sorted(unique_emotions)
    EMOTION_TO_ID = {emo: idx for idx, emo in enumerate(emotions_sorted)}
    with open("iemocap_label_map.json", "w") as f:
        json.dump(EMOTION_TO_ID, f, indent=2)
    return EMOTION_TO_ID


def collect_items():
    train_items = []
    val_items = []

    # IEMOCAP wav files under SessionX/sentences/wav/**/<file>.wav
    for session_dir in DATA_ROOT.glob("Session*/"):
        session_name = session_dir.name
        wav_root = session_dir / "sentences" / "wav"
        if not wav_root.exists():
            continue

        for subdir, _, files in os.walk(wav_root):
            subdir_path = Path(subdir)
            # Load dialog label mapping once per dialog directory
            dialog_name = subdir_path.name
            label_file = session_dir / "dialog" / "EmoEvaluation" / f"{dialog_name}.txt"
            utt_to_emo = parse_dialog_label_file(label_file)

            for file in files:
                if not is_wav_file(file):
                    continue
                wav_path = subdir_path / file

                utt_id = Path(file).stem  # e.g., Ses01F_impro01_F000
                emo = utt_to_emo.get(utt_id)
                if emo is None:
                    continue
                # Map every observed emotion to an id
                if emo not in EMOTION_TO_ID:
                    # This should not happen after build_label_map, but be safe
                    EMOTION_TO_ID[emo] = max(EMOTION_TO_ID.values(), default=-1) + 1
                label = EMOTION_TO_ID[emo]

                item = {"audio_path": str(wav_path), "label": int(label)}
                if session_name in TRAIN_SESSIONS:
                    train_items.append(item)
                elif session_name in VAL_SESSIONS:
                    val_items.append(item)

    return train_items, val_items


def write_jsonl(path: Path, items):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main():
    label_map = build_label_map()
    train_items, val_items = collect_items()
    print(f"Collected train: {len(train_items)}, val: {len(val_items)}")
    print("Label map:")
    try:
        print(json.dumps(label_map, indent=2))
    except Exception:
        print(label_map)
    write_jsonl(Path("train_manifest.jsonl"), train_items)
    write_jsonl(Path("val_manifest.jsonl"), val_items)


if __name__ == "__main__":
    main()


