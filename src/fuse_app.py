"""Gradio web interface for FuseLLM."""
import os
import gradio as gr
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from src.core.orchestrator import run_fuse_pipeline
from src.utils.log_utils import log_event


CSS = """
#title { text-align: center; }
#description { text-align: center; }
"""


def process_input(
    user_text: str,
    user_image: Optional[str],
    user_audio: Optional[str],
    history: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], str, str, str]:
    """Process user input and generate a response.
    
    Args:
        user_text: User's text input
        user_image: Path to uploaded image or None
        user_audio: Path to uploaded audio file or None
        history: Chat history as list of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Updated history and cleared input fields
    """
    # Determine input type and value
    input_type = "text"
    input_value = user_text.strip()
    
    if user_image is not None:
        input_type = "image"
        input_value = user_image
    elif user_audio is not None:
        input_type = "audio"
        input_value = user_audio
    elif not input_value:
        return history, "", "", ""
    
    try:
        # Log the input
        log_event("web_ui", f"Processing {input_type} input")
        
        # Process the input
        response = run_fuse_pipeline(input_value, input_type=input_type)
        
        # Create user message
        if input_type == "text":
            user_message = {"role": "user", "content": user_text}
        else:
            user_message = {"role": "user", "content": f"[{input_type.upper()} UPLOADED]"}
        
        # Create bot response
        bot_message = {"role": "assistant", "content": response}
        
        # Update history
        history.append(user_message)
        history.append(bot_message)
            
    except Exception as e:
        error_msg = f"❌ Error processing your request: {str(e)}"
        log_event("web_ui", f"Error: {str(e)}", level="error")
        
        error_message = {
            "role": "assistant",
            "content": error_msg
        }
        history.append(error_message)
    
    return history, "", "", ""


def create_web_interface() -> gr.Blocks:
    """Create and return the Gradio web interface."""
    with gr.Blocks(
        title="FuseLLM - Multimodal AI Assistant",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS
    ) as demo:
        # Initialize chat history if not exists
        chat_history = gr.State([])
        # Header
        gr.Markdown(
            """
            # 🤖 FuseLLM
            ### A Multimodal AI Assistant
            """,
            elem_id="title"
        )
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat with FuseLLM",
            height=500,
            show_label=False,
            container=True,
            bubble_full_width=False,
            type="messages",  # Use the new messages format
            value=[]  # Initialize with empty list
        )
        
        # Input components
        with gr.Row():
            text_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                container=False,
                scale=4,
                min_width=200,
            )
            
            with gr.Column(scale=1, min_width=100):
                image_upload = gr.Image(
                    type="filepath",
                    label="Upload Image",
                    visible=True,
                    height=40,
                    sources=["upload"],
                    interactive=True
                )
                audio_upload = gr.Audio(
                    type="filepath",
                    label="Upload Audio",
                    visible=True,
                    sources=["upload"]
                )
        
        # Buttons
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            
        # Set up event handlers
        def submit_message(history, message, image, audio):
            # Convert history to the correct format if needed
            if history and isinstance(history[0], (list, tuple)):
                history = []
                
            # Process the input and get updated history
            updated_history, _, _, _ = process_input(message, image, audio, history)
            return updated_history, "", None, None
            
        # Connect the submit button
        submit_btn.click(
            fn=submit_message,
            inputs=[chatbot, text_input, image_upload, audio_upload],
            outputs=[chatbot, text_input, image_upload, audio_upload],
            queue=False
        )
        
        # Also submit on Enter key
        text_input.submit(
            fn=submit_message,
            inputs=[chatbot, text_input, image_upload, audio_upload],
            outputs=[chatbot, text_input, image_upload, audio_upload],
            queue=False
        )
        
        clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        submit_btn.click(
            fn=process_input,
            inputs=[text_input, image_upload, audio_upload, chatbot],
            outputs=[chatbot, text_input, image_upload, audio_upload],
        )
        
        text_input.submit(
            fn=process_input,
            inputs=[text_input, image_upload, audio_upload, chatbot],
            outputs=[chatbot, text_input, image_upload, audio_upload],
        )
        
        clear_btn.click(
            fn=lambda: ([], "", "", ""),
            inputs=[],
            outputs=[chatbot, text_input, image_upload, audio_upload],
        )
        
        # Add some examples
        gr.Examples(
            examples=[
                ["Tell me a joke"],
                ["Explain quantum computing in simple terms"],
                ["What's the weather like today?"]
            ],
            inputs=text_input,
            label="Try these examples:",
        )
    
    return demo


def launch_web_interface(server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch the Gradio web interface."""
    demo = create_web_interface()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        debug=True,
        favicon_path=None
    )


if __name__ == "__main__":
    launch_web_interface()

    history = gr.State([])

    send_btn.click(
        fn=chat_with_fuse,
        inputs=[user_text, user_image, user_audio, history],
        outputs=[chatbot],
        show_progress=True
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, user_text, user_image, user_audio]
    )

demo.launch()
