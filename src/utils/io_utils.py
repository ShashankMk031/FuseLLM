from pathlib import Path
from PIL import Image
import torchaudio

def read_text_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    return path.read_text(encoding="utf-8")

def read_image_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    return Image.open(path)

def read_audio_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate

import csv
from datetime import datetime
from pathlib import Path

def save_io_pair(input_type, user_input, response, filename="data/ui_logs.csv"):
    Path("data").mkdir(exist_ok=True)

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), input_type, user_input, response
        ])
