# from core.task_router import route_task

# test_inputs = [
#     "Summarize this long article about transformers.",
#     "translate this to spanish",
#     "What is the sentiment of this sentence?",
#     "data/sample.jpg",
#     {"text": "What’s in this image?", "image": "data/sample.jpg"},
#     "data/audio_sample.mp3"
# ]

# for i, input_item in enumerate(test_inputs, 1):
#     input_type, intent = route_task(input_item)
#     print(f"Test {i}: Type = {input_type} | Intent = {intent}")


# from core.pipeline_manager import run_pipeline

# test_cases = [
#     ("text-generation", "Tell me a story about a robot"),
#     ("summarization", "Transformers are models that learn to predict..."),
#     ("image-classification", "data/sample.jpg"),
#     ("automatic-speech-recognition", "data/audio_sample.wav")
# ]

# for task, data in test_cases:
#     print(f"\nRunning: {task}")
#     output = run_pipeline(task, data)
#     print(output)


# # Example usage in main.py
# from core.fuser import Fuser

# fuser = Fuser()
# final_output = fuser.fuse(
#     task_type="text-generation",
#     pipeline_output=[{"generated_text": "Once upon a time..."}],
#     retrieved_context="This is some background context from a .txt file."
# )
# print(final_output)

from core.orchestrator import orchestrate

if __name__ == "__main__":
    inputs = [
        {"input": "Summarize: Transformers are models that...", "type": "text"},
        {"input": "Translate to French: I love pizza", "type": "text"},
        {"input": "data/sample.txt", "type": "text"},
        {"input": "data/sample.jpg", "type": "image"},
        {"input": "data/sample.wav", "type": "audio"},
    ]

    for i, item in enumerate(inputs, 1):
        print(f"\nTest {i}:")
        output = orchestrate(item["input"], input_type=item["type"])
        print(output)
