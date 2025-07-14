from typing import Dict, Union

class Fuser:
    def __init__(self):
        pass  # Optional: initialize config or metadata if needed later

    def fuse(
        self,
        task_type: str,
        pipeline_output: Union[str, list, dict],
        retrieved_context: Union[str, dict, None] = None
    ) -> str:
        """
        Combines pipeline output with retrieved context into a single coherent response.
        
        Parameters:
            task_type: str
                The pipeline type executed (e.g. "text-generation", "image-classification")
            pipeline_output: str, list, or dict
                Raw output from the Hugging Face pipeline
            retrieved_context: str, dict, or None
                Optional: Content retrieved from local files (e.g. .txt, image captions)

        Returns:
            str: Final assistant response.
        """
        if task_type == "text-generation":
            return self._fuse_text_generation(pipeline_output, retrieved_context)
        elif task_type == "summarization":
            return self._fuse_summarization(pipeline_output, retrieved_context)
        elif task_type == "image-classification":
            return self._fuse_image_classification(pipeline_output, retrieved_context)
        elif task_type == "automatic-speech-recognition":
            return self._fuse_audio_transcription(pipeline_output, retrieved_context)
        elif task_type == "image-text-to-text":
            return self._fuse_multimodal_output(pipeline_output, retrieved_context)
        else:
            return f"Unsupported task type: {task_type}"

    # === Internal fusion methods ===

    def _fuse_text_generation(self, pipeline_output, context):
        text = pipeline_output[0]['generated_text'] if isinstance(pipeline_output, list) else str(pipeline_output)
        return f"Generated Response:\n{text}"

    def _fuse_summarization(self, pipeline_output, context):
        summary = pipeline_output[0]['summary_text'] if isinstance(pipeline_output, list) else str(pipeline_output)
        if context:
            return f"Summary:\n{summary}\n\nContext Source:\n{context}"
        return f"Summary:\n{summary}"

    def _fuse_image_classification(self, pipeline_output, context):
        labels = [f"- {item['label']} ({item['score']:.2f})" for item in pipeline_output]
        label_text = "\n".join(labels)
        if context:
            return f"Image Classification Result:\n{label_text}\n\nRelated Info:\n{context}"
        return f"Image Classification Result:\n{label_text}"

    def _fuse_audio_transcription(self, pipeline_output, context):
        transcription = pipeline_output.get("text", "")
        return f"Transcribed Audio:\n{transcription}"

    def _fuse_multimodal_output(self, pipeline_output, context):
        # Custom fusion rule for multimodal tasks
        response = f"Multimodal Output:\n{pipeline_output}"
        if context:
            response += f"\n\nContext Used:\n{context}"
        return response
