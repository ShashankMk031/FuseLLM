# test/test_fuser.py

from core.fuser import Fuser

def test_text_generation():
    fuser = Fuser()
    output = [{"generated_text": "Hello from the model!"}]
    response = fuser.fuse("text-generation", output)
    assert "Generated Response" in response
    print("✅ text-generation test passed")

def test_summarization():
    fuser = Fuser()
    output = [{"summary_text": "This is a summary."}]
    context = "Original text content."
    response = fuser.fuse("summarization", output, context)
    assert "Summary" in response and "Original" in response
    print("✅ summarization test passed")

def test_image_classification():
    fuser = Fuser()
    output = [
        {"label": "cat", "score": 0.91},
        {"label": "dog", "score": 0.07}
    ]
    response = fuser.fuse("image-classification", output)
    assert "Image Classification Result" in response
    print("✅ image-classification test passed")

def test_audio_transcription():
    fuser = Fuser()
    output = {"text": "Transcribed speech here."}
    response = fuser.fuse("automatic-speech-recognition", output)
    assert "Transcribed Audio" in response
    print("✅ audio transcription test passed")

def test_multimodal():
    fuser = Fuser()
    output = "A multimodal result text"
    context = "Image caption or text+image reasoning."
    response = fuser.fuse("image-text-to-text", output, context)
    assert "Multimodal Output" in response
    print("✅ multimodal fusion test passed")

def test_unknown():
    fuser = Fuser()
    response = fuser.fuse("nonexistent-task", "some output")
    assert "Unsupported task type" in response
    print("✅ unknown task test passed")


if __name__ == "__main__":
    test_text_generation()
    test_summarization()
    test_image_classification()
    test_audio_transcription()
    test_multimodal()
    test_unknown()
