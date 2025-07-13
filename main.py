from core.task_router import route_task

test_inputs = [
    "Summarize this long article about transformers.",
    "translate this to spanish",
    "What is the sentiment of this sentence?",
    "data/sample.jpg",
    {"text": "What’s in this image?", "image": "data/sample.jpg"},
    "data/audio_sample.mp3"
]

for i, input_item in enumerate(test_inputs, 1):
    input_type, intent = route_task(input_item)
    print(f"Test {i}: Type = {input_type} | Intent = {intent}")


from core.pipeline_manager import run_pipeline

test_cases = [
    ("text-generation", "Tell me a story about a robot"),
    ("summarization", "Transformers are models that learn to predict..."),
    ("image-classification", "data/sample.jpg"),
    ("automatic-speech-recognition", "data/audio_sample.wav")
]

for task, data in test_cases:
    print(f"\nRunning: {task}")
    output = run_pipeline(task, data)
    print(output)
