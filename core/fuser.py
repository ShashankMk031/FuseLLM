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
        # Normalize task type
        task_type = task_type.lower()
        
        # Map similar task types
        if task_type in ["text", "general"]:
            task_type = "text-generation"
            
        try:
            if task_type == "text-generation":
                return self._fuse_text_generation(pipeline_output, retrieved_context)
            elif task_type == "summarization":
                return self._fuse_summarization(pipeline_output, retrieved_context)
            elif task_type == "image-classification":
                return self._fuse_image_classification(pipeline_output, retrieved_context)
            elif task_type == "automatic-speech-recognition":
                return self._fuse_audio_transcription(pipeline_output, retrieved_context)
            elif task_type in ["image-text-to-text", "multimodal"]:
                return self._fuse_multimodal_output(pipeline_output, retrieved_context)
            elif task_type == "text-classification":
                # Handle text classification results
                if isinstance(pipeline_output, list) and pipeline_output and 'label' in pipeline_output[0]:
                    return f"Classification Result: {pipeline_output[0]['label']}"
                return f"Classification: {str(pipeline_output)}"
            else:
                # For unknown task types, try to return something sensible
                if isinstance(pipeline_output, (str, int, float, bool)):
                    return str(pipeline_output)
                elif isinstance(pipeline_output, dict):
                    return "\n".join(f"{k}: {v}" for k, v in pipeline_output.items())
                elif isinstance(pipeline_output, list):
                    return "\n".join(str(item) for item in pipeline_output)
                else:
                    return f"Result: {str(pipeline_output)}"
        except Exception as e:
            return f"Error processing {task_type} output: {str(e)}\nRaw output: {str(pipeline_output)[:500]}"

    # === Internal fusion methods ===

    def _fuse_text_generation(self, pipeline_output, context=None):
        """
        Handle text generation pipeline output.
        
        Args:
            pipeline_output: Output from the text generation pipeline
            context: Optional context to include in the response
            
        Returns:
            Formatted text response
        """
        try:
            # Handle different output formats
            if isinstance(pipeline_output, dict):
                text = pipeline_output.get('generated_text', str(pipeline_output))
            elif isinstance(pipeline_output, list):
                if not pipeline_output:
                    return "No text was generated."
                # Handle list of dictionaries
                if isinstance(pipeline_output[0], dict):
                    if 'generated_text' in pipeline_output[0]:
                        text = pipeline_output[0]['generated_text']
                    else:
                        text = str(pipeline_output[0])
                else:
                    text = "\n".join(str(item) for item in pipeline_output)
            elif isinstance(pipeline_output, str):
                text = pipeline_output
            else:
                text = str(pipeline_output)
                
            # Clean up the text
            text = text.strip()
            if not text:
                return "No text was generated."
                
            # Add context if provided
            if context:
                if isinstance(context, dict):
                    context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
                else:
                    context_str = str(context)
                text = f"{text}\n\nContext:\n{context_str}"
                
            return text
            
        except Exception as e:
            error_msg = f"Error processing text generation output: {str(e)}"
            raw_output = str(pipeline_output)[:500]  # Limit length of raw output
            return f"{error_msg}\n\nRaw output: {raw_output}"

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
