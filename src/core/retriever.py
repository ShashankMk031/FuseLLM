"""Search text, images, audio, or metadata-tagged documents.

Return relevant content (or paths/payloads) to the main pipeline for fusion and response.

Support multimodal fusion, e.g., combining image captions with relevant text.

"""

import os
import json
from typing import List, Dict, Union


class Retriever:
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return []

    def retrieve_text(self, query: str) -> List[str]:
        results = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".txt"):
                with open(os.path.join(self.data_dir, file), "r") as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        results.append(content)
        return results

    def retrieve_image(self, keyword: str) -> List[str]:
        return [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and keyword.lower() in f.lower()
        ]

    def retrieve_audio(self, keyword: str) -> List[str]:
        return [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.lower().endswith((".wav", ".mp3")) and keyword.lower() in f.lower()
        ]

    def retrieve_by_metadata(self, key: str, value: Union[str, List[str]]) -> List[Dict]:
        if not self.metadata:
            return []

        values = [value] if isinstance(value, str) else value
        return [entry for entry in self.metadata if entry.get(key) in values]

    def multimodal_fuse(self, query: str) -> Dict[str, List[str]]:
        return {
            "text": self.retrieve_text(query),
            "image": self.retrieve_image(query),
            "audio": self.retrieve_audio(query),
        }
