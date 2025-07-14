# test_retriever for the retriver.py 

from core.retriever import Retriever
import os

# Initialize
retriever = Retriever(data_dir="data/")

# Sample tests (assuming you have some sample .txt, .png, .wav files in /data)
print("=== Text Retrieval ===")
text_results = retriever.retrieve_text("robot")
print(f"Found {len(text_results)} matching text documents.")
for i, r in enumerate(text_results[:2]):
    print(f"-> Result {i+1}:\n{r[:150]}...\n")

print("=== Image Retrieval ===")
image_results = retriever.retrieve_image("sample")
print(f"Found {len(image_results)} matching image files: {image_results}")

print("=== Audio Retrieval ===")
audio_results = retriever.retrieve_audio("sample")
print(f"Found {len(audio_results)} matching audio files: {audio_results}")

print("=== Multimodal Fusion ===")
multi = retriever.multimodal_fuse("sample")
print(f"Multimodal result: {multi}")
