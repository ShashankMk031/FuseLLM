"""Gradio web interface for FuseLLM."""
import gradio as gr
import traceback
import os
import logging
from typing import Optional, List, Tuple, Dict, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.core.orchestrator import run_fuse_pipeline
from src.utils.log_utils import log_event

# Custom CSS for the web interface
custom_css = """
:root {
    --primary-color: #4f46e5;
    --secondary-color: #6b7280;
    --accent-color: #8b5cf6;
    --background-color: #f9fafb;
    --surface-color: #ffffff;
    --text-color: #111827;
    --border-color: #e5e7eb;
}

/* Dark theme support */
.dark {
    --primary-color: #6366f1;
    --secondary-color: #9ca3af;
    --accent-color: #a78bfa;
    --background-color: #111827;
    --surface-color: #1f2937;
    --text-color: #f9fafb;
    --border-color: #374151;
}

/* Main container */
.contain {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

/* Chat container */
#chatbot {
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    background-color: var(--surface-color);
    height: 70vh;
    overflow-y: auto;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Message bubbles */
.message {
    margin: 0.5rem 0;
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    max-width: 80%;
    line-height: 1.5;
}

.user-message {
    background-color: var(--primary-color);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 0.25rem;
}

.assistant-message {
    background-color: var(--surface-color);
    border: 1px solid var(--border-color);
    margin-right: auto;
    border-bottom-left-radius: 0.25rem;
}

/* Input area */
.input-area {
    display: flex;
    gap: 0.5rem;
    padding: 1rem;
    background-color: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    margin-top: 1rem;
}

/* Buttons */
button {
    transition: all 0.2s ease-in-out;
}

button:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

button:active {
    transform: translateY(0);
}

/* Loading spinner */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
    margin-right: 8px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .message {
        max-width: 90%;
    }
    
    .input-area {
        flex-direction: column;
    }
    
    button {
        width: 100%;
    }
}
"""

CSS = """
/* General Styles */
#title { 
    text-align: center; 
    margin: 10px 0 5px 0;
    font-size: 1.8em;
    font-weight: 600;
    color: #2c3e50;
}
#description { 
    text-align: center; 
    margin-bottom: 20px;
    color: #7f8c8d;
    font-size: 0.95em;
}

/* Chat Container */
#chatbot {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    min-height: 60vh;
    max-height: 60vh;
    overflow-y: auto;
    padding: 15px;
    background: #fafafa;
    margin-bottom: 15px;
}

/* Messages */
.message {
    margin: 10px 0;
    padding: 12px 15px;
    border-radius: 18px;
    line-height: 1.4;
    position: relative;
    max-width: 85%;
    word-wrap: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.user-message {
    background: #4285f4;
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 5px;
}

.bot-message {
    background: #f1f3f4;
    color: #202124;
    margin-right: auto;
    border-bottom-left-radius: 5px;
}

/* Input Area */
#text_input, #image_input, #audio_input {
    border-radius: 20px;
    padding: 12px 20px;
    border: 1px solid #dfe1e5;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    transition: all 0.3s ease;
}

#text_input:focus, #image_input:focus, #audio_input:focus {
    border-color: #4285f4;
    box-shadow: 0 2px 8px rgba(66, 133, 244, 0.2) !important;
}

/* Buttons */
button {
    border-radius: 20px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    border: none !important;
    cursor: pointer;
}

button.primary {
    background: #4285f4 !important;
    color: white !important;
    box-shadow: 0 2px 5px rgba(66, 133, 244, 0.3);
}

button.primary:hover {
    background: #3367d6 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(66, 133, 244, 0.3);
}

button.secondary {
    background: #f1f3f4 !important;
    color: #5f6368 !important;
    margin: 5px;
    padding: 6px 12px !important;
    font-size: 0.85em !important;
}

button.secondary:hover {
    background: #e8eaed !important;
    transform: translateY(-1px);
}

/* Tabs */
.tab-nav {
    margin-bottom: 10px !important;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 5px;
}

.tab-nav button {
    border-radius: 10px 10px 0 0 !important;
    margin-right: 5px !important;
    padding: 8px 15px !important;
    background: #f1f3f4 !important;
    color: #5f6368 !important;
    border: 1px solid #dfe1e5 !important;
    border-bottom: none !important;
}

.tab-nav button.selected {
    background: white !important;
    border-bottom: 1px solid white !important;
    margin-bottom: -1px !important;
    color: #1a73e8 !important;
    font-weight: 600 !important;
}

/* Suggestions */
.suggestions {
    margin: 15px 0;
    text-align: center;
}

/* Accordion */
.accordion {
    margin: 15px 0;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
}

.accordion-header {
    background: #f8f9fa;
    padding: 10px 15px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 500;
    color: #5f6368;
}

.accordion-content {
    padding: 15px;
    background: white;
    border-top: 1px solid #e0e0e0;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px 15px;
    color: #7f8c8d;
    font-size: 0.9em;
    border-top: 1px solid #eee;
    background: #f8f9fa;
    border-radius: 0 0 10px 10px;
}

.footer a {
    color: #5f6368;
    text-decoration: none;
    transition: color 0.2s ease;
}

.footer a:hover {
    color: #1a73e8;
    text-decoration: underline;
}

.footer span {
    margin: 0 5px;
    color: #dadce0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .message {
        max-width: 90%;
        font-size: 0.95em;
    }
    
    #chatbot {
        height: 50vh;
    }
    
    button {
        padding: 6px 15px !important;
        font-size: 0.9em !important;
    }
    
    .footer {
        padding: 15px 10px;
        font-size: 0.85em;
    }
}"""





def create_web_interface() -> gr.Blocks:
    """Create and return the Gradio web interface."""
    # ===== Component Definitions =====
    # State
    loading_state = gr.State(False)
    
    # Input Components
    text_input = gr.Textbox(
        label="Message",
        placeholder="Type your message here...",
        lines=2,
        max_lines=5,
        container=False,
        show_label=False,
        scale=7,
        min_width=100
    )
    
    image_input = gr.Image(
        label="Upload Image",
        type="filepath",
        visible=False,
        container=False
    )
    
    audio_input = gr.Audio(
        label="Upload Audio",
        type="filepath",
        visible=False,
        container=False
    )
    
    # Output Components
    chatbot = gr.Chatbot(
        label="FuseLLM",
        height=600,
        show_label=False,
        container=True,
        show_copy_button=True,
        avatar_images=(None, "https://avatars.githubusercontent.com/u/12345678?s=200&v=4"),
        render=False
    )
    
    # Buttons
    submit_btn = gr.Button(
        value="Send",
        variant="primary",
        size="lg",
        scale=1,
        min_width=100
    )
    
    clear_btn = gr.Button(
        value="Clear",
        variant="secondary",
        size="lg",
        scale=1,
        min_width=100
    )
    
    # Loading Indicator
    loading = gr.HTML(
        "<div style='text-align: center; margin: 20px;'>"
        "<div class='spinner' style='width: 40px; height: 40px; margin: 0 auto; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;'></div>"
        "<p style='margin-top: 10px; color: #666;'>Generating response...</p>"
        "</div>"
        "<style>"
        "@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }"
        "</style>",
        visible=False
    )
    
    # ===== Helper Functions =====
    
    def toggle_loading(loading_state: bool):
        """Show or hide loading indicator and update UI elements.
        
        Returns:
            Tuple of component updates in the order they're expected by the event handler
        """
        try:
            logger.debug(f"Toggling loading state: {loading_state}")
            
            # Return updates as a tuple in the correct order
            return (
                # Loading indicator visibility
                gr.update(visible=loading_state),
                # Submit button state
                gr.update(
                    interactive=not loading_state,
                    variant="secondary" if loading_state else "primary"
                ),
                # Text input state
                gr.update(interactive=not loading_state),
                # Image input state
                gr.update(interactive=not loading_state),
                # Audio input state
                gr.update(interactive=not loading_state),
                # Clear button state
                gr.update(interactive=not loading_state)
            )
            
        except Exception as e:
            logger.error(f"Error in toggle_loading: {str(e)}", exc_info=True)
            # Return minimal updates on error in the correct order
            return (
                gr.update(visible=loading_state),
                gr.update(interactive=not loading_state),
                gr.update(interactive=not loading_state),
                gr.update(interactive=not loading_state),
                gr.update(interactive=not loading_state),
                gr.update(interactive=not loading_state)
            )
    
    with gr.Blocks(
        title="FuseLLM - AI Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            spacing_size="sm",
            radius_size="md",
            font=["Inter", "sans-serif"]
        ),
        css=custom_css
    ) as demo:
        # All components are now defined at the module level
        pass
        # Initialize chat history if not exists
        chat_history = gr.State([])
        
        # Header
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div style='text-align: center; margin-bottom: 20px;'>
                        <h1 style='margin-bottom: 5px;'>🤖 FuseLLM</h1>
                        <p style='color: #666;'>A simple yet powerful AI assistant</p>
                    </div>
                    """
                )
        
        # Chat interface with improved layout
        with gr.Row(equal_height=True):
            with gr.Column(scale=8, min_width=600):
                # Chatbot display with improved styling and message format
                with gr.Group(elem_classes=["chat-container"]):
                    chatbot = gr.Chatbot(
                        value=[],
                        elem_id="chatbot",
                        elem_classes=["chatbot"],
                        show_label=False,
                        height=500,
                        container=True,
                        render_markdown=True,  # Enable markdown rendering
                        show_copy_button=True,  # Add copy button to messages
                        bubble_full_width=False,  # Better control over message width
                        # Removed unsupported parameters: likeable, show_share_button, show_retry_button, avatar_images
                        type="messages",  # Use the new message format
                        sanitize_html=True  # Allow HTML in messages but sanitize for security
                    )
                
                # Input area with improved layout
                with gr.Row():
                    with gr.Column(scale=7):
                        # Text input with placeholder
                        text_input = gr.Textbox(
                            placeholder="Type your message here...",
                            label=" ",
                            show_label=False,
                            container=False,
                            elem_id="text_input",
                            scale=8
                        )
                    
                    # Action buttons in a row (already defined at the component level)
                    with gr.Column(scale=1, min_width=200):
                        with gr.Row():
                            submit_btn
                            clear_btn
                
                # Loading indicator with better styling
                loading.render()
                
                # Add custom CSS for chat interface
                gr.HTML("""
                <style>
                    /* Chat container styling */
                    .chat-container {
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background: #f9f9f9;
                        min-height: 500px;
                        max-height: 70vh;
                        overflow-y: auto;
                    }
                    
                    /* Chat bubbles */
                    .user, .assistant {
                        margin: 8px 0;
                        padding: 10px 15px;
                        border-radius: 18px;
                        max-width: 85%;
                        word-wrap: break-word;
                        line-height: 1.5;
                    }
                    
                    .user {
                        background-color: #4f46e5;
                        color: white;
                        margin-left: auto;
                        border-bottom-right-radius: 4px;
                    }
                    
                    .assistant {
                        background-color: #f0f0f0;
                        color: #333;
                        margin-right: auto;
                        border-bottom-left-radius: 4px;
                    }
                    
                    /* Loading indicator */
                    .spinner {
                        width: 30px;
                        height: 30px;
                        border: 4px solid #f3f3f3;
                        border-top: 4px solid #4f46e5;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin: 0 auto;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    /* Input area */
                    #text_input textarea {
                        border-radius: 20px !important;
                        padding: 10px 15px !important;
                        min-height: 45px !important;
                        max-height: 150px !important;
                        resize: vertical !important;
                    }
                    
                    /* Buttons */
                    button {
                        border-radius: 20px !important;
                        padding: 8px 20px !important;
                        font-weight: 500 !important;
                        transition: all 0.2s ease !important;
                    }
                    
                    button:hover {
                        transform: translateY(-1px);
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                    
                    /* Responsive design */
                    @media (max-width: 768px) {
                        .chat-container {
                            max-height: 60vh;
                        }
                        
                        .user, .assistant {
                            max-width: 90%;
                        }
                    }
                </style>
                """)
                
        # Define process_inputs function to handle text input submission
        def process_inputs(text, image, audio, chat_history):
            """Process user input from the text input field.
            
            Args:
                text: The text input from the user
                image: The image input (if any)
                audio: The audio input (if any)
                chat_history: The current chat history in the new message format
                
            Returns:
                Tuple of (cleared_text, None, None, updated_history, loading_state)
            """
            try:
                # If no input, return current state
                if not text and not image and not audio:
                    logger.warning("No input provided to process_inputs")
                    return "", None, None, chat_history or [], False
                
                # Create user message based on input type
                if image:
                    user_message = text.strip() if text.strip() else "[Image]"
                    logger.info(f"Processing image input: {image}")
                elif audio:
                    user_message = text.strip() if text.strip() else "[Audio]"
                    logger.info(f"Processing audio input: {audio}")
                else:
                    user_message = text.strip()
                    logger.info(f"Processing text input: {user_message[:50]}...")
                
                # Initialize history if needed
                updated_history = list(chat_history) if chat_history else []
                
                # Add user message with a placeholder for the assistant's response
                updated_history.append({"role": "user", "content": user_message})
                updated_history.append({"role": "assistant", "content": None})  # Placeholder for response
                
                # Clear inputs and set loading state
                return "", None, None, updated_history, True
                
            except Exception as e:
                error_msg = f"Error in process_inputs: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return "", None, None, chat_history or [], False
            
        # Define clear_inputs function
        def clear_inputs() -> Tuple[str, None, None, List[Dict[str, str]]]:
            """Clear all input fields and reset the chat history.
            
            Returns:
                Tuple of (empty_text, None, None, empty_list) to reset the UI
            """
            logger.info("Clearing all inputs and chat history")
            return "", None, None, []  # text, image, audio, chat_history
            
        # Define process_and_respond function with new message format
        def process_and_respond(
            message: str,
            image: Optional[str],
            audio: Optional[str],
            chat_history: List[Dict[str, str]]
        ) -> Tuple[str, Optional[str], Optional[str], List[Dict[str, str]]]:
            """Process the user input and generate a response using the new message format.
            
            Args:
                message: The user's message text
                image: Path to an uploaded image file, if any
                audio: Path to an uploaded audio file, if any
                chat_history: List of message dictionaries with 'role' and 'content' keys
                
            Returns:
                Tuple of (cleared_text, None, None, updated_chat_history)
            """
            try:
                # Determine input type and prepare input data
                if image:
                    # Process image
                    input_type = "image"
                    input_data = image
                    user_message = message if message.strip() else "Tell me about this image"
                    logger.info(f"Processing image input: {image}")
                elif audio:
                    # Process audio
                    input_type = "audio"
                    input_data = audio
                    user_message = message if message.strip() else "Transcribe this audio"
                    logger.info(f"Processing audio input: {audio}")
                else:
                    # Process text
                    input_type = "text"
                    input_data = message.strip()
                    user_message = input_data
                    
                    # Don't process empty messages
                    if not user_message:
                        return "", None, None, chat_history
                
                # Add user message to chat history
                chat_history.append({"role": "user", "content": user_message})
                
                # Add a placeholder for the assistant's response
                chat_history.append({"role": "assistant", "content": None})
                
                # Generate response with error handling
                try:
                    # Convert chat history to old format for the pipeline if needed
                    old_format_history = []
                    for msg in chat_history:
                        if msg["role"] == "user":
                            old_format_history.append((msg["content"], None))
                        elif msg["role"] == "assistant" and msg["content"] is not None:
                            if old_format_history and old_format_history[-1][1] is None:
                                old_format_history[-1] = (old_format_history[-1][0], msg["content"])
                    
                    # Generate response with error handling
                    response = run_fuse_pipeline(
                        user_input=input_data,
                        input_type=input_type,
                        metadata={"chat_history": old_format_history}
                    )
                    
                    # Extract response text
                    if isinstance(response, dict) and 'error' in response:
                        response_text = f"❌ Error: {response['error']}"
                        logger.error(f"Pipeline error: {response['error']}")
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                        
                except Exception as e:
                    error_msg = str(e)
                    response_text = f"❌ Error processing your request: {error_msg}"
                    logger.error(f"Error in pipeline execution: {error_msg}", exc_info=True)
                
                # Update the chat history with the response
                if chat_history and chat_history[-1]["role"] == "assistant":
                    chat_history[-1]["content"] = response_text
                
                # Clear input fields and return
                return "", None, None, chat_history
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Unexpected error in process_and_respond: {error_msg}", exc_info=True)
                
                # Add error message to chat history if we have a message to attach it to
                if chat_history and len(chat_history) > 0:
                    # If we have a pending assistant message, update it with the error
                    if chat_history[-1]["role"] == "assistant" and chat_history[-1]["content"] is None:
                        chat_history[-1]["content"] = f"❌ An error occurred: {error_msg}"
                    else:
                        # Otherwise, add a new error message
                        chat_history.append({"role": "assistant", "content": f"❌ An error occurred: {error_msg}"})
                else:
                    # If no history, create a new chat with the error
                    chat_history = [{"role": "assistant", "content": f"❌ An error occurred: {error_msg}"}]
                
                return "", None, None, chat_history
                
        # Suggestion buttons container
        with gr.Row() as suggestion_row:
            with gr.Column(scale=1):
                with gr.Accordion("💡 Suggestions", open=False) as suggestions_accordion:
                    # Create a container for the suggestion buttons
                    with gr.Row() as suggestions_container:
                        # Define suggestion texts
                        suggestion_texts = [
                            "Tell me about yourself",
                            "What can you do?",
                            "Help me with a task"
                        ]
                        
                        # Create buttons for each suggestion
                        for text in suggestion_texts:
                            with gr.Column(scale=1, min_width=150):
                                btn = gr.Button(
                                    text,
                                    size="sm",
                                    variant="secondary",
                                    min_width=120,
                                    scale=1
                                )
                                
                                # Set up click handler with proper scoping for the new message format
                                def create_suggestion_handler(suggestion_text):
                                    def handler(chat_history):
                                        # Create a new chat history with the suggestion as a user message
                                        if chat_history is None:
                                            chat_history = []
                                        
                                        # Add the suggestion as a user message
                                        chat_history.append({"role": "user", "content": suggestion_text})
                                        
                                        # Return values in the same order as the outputs list
                                        # [text_input, image_input, audio_input, chat_history, loading_state]
                                        return (
                                            "",                  # text_input (cleared)
                                            None,                # image_input
                                            None,                # audio_input
                                            chat_history,        # updated chat history
                                            True                 # loading_state
                                        )
                                    return handler
                                
                                # Connect the click handler
                                btn.click(
                                    fn=create_suggestion_handler(text),
                                    inputs=chatbot,
                                    outputs=[text_input, image_input, audio_input, chatbot, loading_state],
                                    queue=False
                                ).then(
                                    fn=toggle_loading,
                                    inputs=loading_state,
                                    outputs=[loading, submit_btn, text_input, image_input, audio_input, clear_btn],
                                    queue=False
                                ).then(
                                    fn=process_and_respond,
                                    inputs=[text_input, image_input, audio_input, chatbot],
                                    outputs=[text_input, image_input, audio_input, chatbot],
                                    queue=True
                                ).then(
                                    fn=toggle_loading,
                                    inputs=gr.State(False),
                                    outputs=[loading, submit_btn, text_input, image_input, audio_input, clear_btn],
                                    queue=False
                                )
                
                # Action buttons
                with gr.Row():
                    submit_btn = gr.Button(
                        "Send",
                        variant="primary",
                        size="lg",
                        min_width=100
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        size="lg",
                        min_width=100
                    )
                
                # Loading indicator is now defined at the top of the function
        
        # ===== Helper Functions =====
        
        def on_suggestion_click(suggestion, history):
            """Handle suggestion button clicks."""
            if not history:
                history = []
            # Add user message with None as bot response (loading state)
            history.append((suggestion, None))
            # Return updated state and show loading
            return "", None, None, history, True  # text, image, audio, chat_history, loading
        
        # ===== Event Handlers =====
        
        def process_uploaded_file(file_path: Optional[Union[str, bytes]] = None) -> Optional[str]:
            """Process an uploaded file and return its path if valid."""
            try:
                if not file_path:
                    return None
                    
                # Handle file-like objects (Gradio's temp files)
                if hasattr(file_path, 'name') and os.path.isfile(file_path.name):
                    return os.path.abspath(file_path.name)
                    
                # Handle string paths
                if isinstance(file_path, str):
                    # Check if it's a file and not a directory
                    if os.path.isfile(file_path):
                        return os.path.abspath(file_path)
                        
                logging.warning(f"Invalid file path provided: {file_path}")
                return None
                
            except Exception as e:
                log_event("web_ui", f"Error processing uploaded file: {str(e)}", level="error")
                return None
        
        def create_suggestion_button(suggestion, is_visible=True):
            """Helper to create a suggestion button with consistent styling and behavior."""
            btn = gr.Button(
                suggestion,
                size="sm",
                variant="secondary",
                min_width=120,
                scale=1,
                visible=is_visible
            )
            
            # Set up click handler
            click_event = btn.click(
                fn=on_suggestion_click,
                inputs=[gr.Textbox(suggestion, visible=False), chatbot],
                outputs=[text_input, image_input, audio_input, chatbot, loading_state],
                queue=False
            )
            
            # Set up loading state
            click_event = click_event.then(
                fn=toggle_loading,
                inputs=loading_state,
                outputs=[
                    loading,        # Show loading indicator
                    submit_btn,     # Disable submit button
                    text_input,     # Disable text input
                    image_input,    # Disable image input
                    audio_input,    # Disable audio input
                    clear_btn,      # Disable clear button
                    btn             # Disable this button
                ],
                queue=False
            )
            
            # Process the response
            click_event = click_event.then(
                fn=process_and_respond,
                inputs=[gr.Textbox(suggestion, visible=False), None, None, chatbot],
                outputs=[text_input, image_input, audio_input, chatbot],
                queue=True
            )
            
            # Reset loading state
            click_event.then(
                fn=toggle_loading,
                inputs=gr.State(False),
                outputs=[
                    loading,        # Hide loading indicator
                    submit_btn,     # Re-enable submit button
                    text_input,     # Re-enable text input
                    image_input,    # Re-enable image input
                    audio_input,    # Re-enable audio input
                    clear_btn,      # Re-enable clear button
                    btn             # Re-enable this button
                ],
                queue=False
            )
            
            return btn
            
        def create_submission_handler(input_component):
            """Create a submission handler for the given input component.
            
            Args:
                input_component: The input component that triggered the submission
                
            Returns:
                A handler function that processes the submission
            """
            def handler(*args):
                try:
                    # Extract inputs from the event
                    text, image, audio, history = args[:4]
                    
                    # Skip if no input
                    if not text and not image and not audio:
                        logger.warning("Submission with no input")
                        return "", None, None, history or [], False
                    
                    # Create user message based on input type
                    if image:
                        user_message = text.strip() if text.strip() else "[Image]"
                        logger.info(f"Processing image submission: {user_message}")
                    elif audio:
                        user_message = text.strip() if text.strip() else "[Audio]"
                        logger.info(f"Processing audio submission: {user_message}")
                    else:
                        user_message = text.strip()
                        logger.info(f"Processing text submission: {user_message[:50]}...")
                    
                    # Initialize or copy the history
                    updated_history = list(history) if history else []
                    
                    # Add user message and a placeholder for the assistant's response
                    updated_history.append({"role": "user", "content": user_message})
                    updated_history.append({"role": "assistant", "content": None})  # Placeholder for response
                    
                    # Clear inputs and show loading
                    return "", None, None, updated_history, True
                    
                except Exception as e:
                    error_msg = f"Error in submission handler: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return "", None, None, history or [], False
            
            return handler
            
        def process_submission(text, image, audio, history):
            """Process the submission and generate a response.
            
            Args:
                text: The user's text input
                image: Path to uploaded image file, if any
                audio: Path to uploaded audio file, if any
                history: Current chat history in the new message format (list of dicts)
                
            Returns:
                Tuple of (cleared_text, None, None, updated_history, loading_state)
            """
            try:
                # Call the existing process_and_respond function
                new_text, new_image, new_audio, updated_history = process_and_respond(text, image, audio, history)
                # Return values in the correct order: text, image, audio, history, loading_state
                return new_text, new_image, new_audio, updated_history, False
                
            except Exception as e:
                error_msg = f"Error processing submission: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Update history with error message in the new format
                history = history or []
                
                try:
                    # If the last message is an assistant's placeholder, update it with the error
                    if history and history[-1]["role"] == "assistant" and history[-1]["content"] is None:
                        history[-1]["content"] = f"❌ {error_msg}"
                    else:
                        # Otherwise, add a new error message
                        history.append({"role": "assistant", "content": f"❌ {error_msg}"})
                except (IndexError, KeyError, TypeError) as history_error:
                    logger.error(f"Error updating chat history with error: {history_error}")
                    # If we can't update the history, create a new one with the error
                    history = [{"role": "assistant", "content": f"❌ {error_msg}"}]
                
                # Return values in the correct order: text, image, audio, history, loading_state
                return "", None, None, history, False
                
        def process_and_respond(
            message: str,
            image: Optional[str],
            audio: Optional[str],
            chat_history: List[Dict[str, str]]
        ) -> Tuple[str, Optional[str], Optional[str], List[Dict[str, str]]]:
            """Process the user input and generate a response."""
            try:
                # Determine input type and prepare input data
                if image:
                    # Process image
                    input_type = "image"
                    input_data = image
                    user_message = message if message.strip() else "Tell me about this image"
                    logger.info(f"Processing image input: {image}")
                elif audio:
                    # Process audio
                    input_type = "audio"
                    input_data = audio
                    user_message = message if message.strip() else "Transcribe this audio"
                    logger.info(f"Processing audio input: {audio}")
                else:
                    # Process text
                    input_type = "text"
                    input_data = message.strip()
                    user_message = input_data
                    
                    # Don't process empty messages
                    if not user_message:
                        return "", None, None, chat_history
                
                # Add user message to chat history
                chat_history = chat_history or []
                chat_history.append({"role": "user", "content": user_message})
                
                # Add placeholder for assistant's response
                chat_history.append({"role": "assistant", "content": None})
                
                # Generate response with error handling
                try:
                    # Convert chat history to old format for compatibility
                    old_format_history = []
                    for msg in chat_history:
                        if msg["role"] == "user":
                            old_format_history.append((msg["content"], None))
                        elif msg["role"] == "assistant" and msg["content"] is not None:
                            if old_format_history and old_format_history[-1][1] is None:
                                old_format_history[-1] = (old_format_history[-1][0], msg["content"])
                            else:
                                old_format_history.append((None, msg["content"]))
                    
                    # Generate response
                    response = run_fuse_pipeline(
                        user_input=input_data,
                        input_type=input_type,
                        metadata={"chat_history": old_format_history}
                    )
                    
                    # Process response to extract clean text
                    if isinstance(response, dict):
                        if 'error' in response:
                            # Handle error case
                            response_text = f"❌ Error: {response['error']}"
                            logger.error(f"Pipeline error: {response['error']}")
                        elif 'response' in response:
                            # Extract just the response text if available
                            response_text = response['response']
                        elif 'text' in response:
                            # Handle case where response is in a 'text' field
                            response_text = response['text']
                        elif 'generated_text' in response:
                            # Handle HuggingFace style response
                            response_text = response['generated_text']
                        else:
                            # Fallback to string representation if no recognized format
                            response_text = str(response)
                    elif isinstance(response, str):
                        # If it's already a string, use it as is
                        response_text = response
                    else:
                        # For any other type, convert to string
                        response_text = str(response)
                        
                    # Clean up the response text if needed
                    if response_text.startswith('```'):
                        # Remove markdown code blocks if present
                        response_text = response_text.replace('```', '').strip()
                    response_text = response_text.strip()
                        
                    # Update the assistant's message with the response
                    if chat_history and chat_history[-1]["role"] == "assistant":
                        chat_history[-1]["content"] = response_text
                        
                except Exception as e:
                    error_msg = str(e)
                    response_text = f"❌ Error processing your request: {error_msg}"
                    logger.error(f"Error in pipeline execution: {error_msg}", exc_info=True)
                    
                    # Update the assistant's message with the error
                    if chat_history and chat_history[-1]["role"] == "assistant":
                        chat_history[-1]["content"] = response_text
                
                # Clear input fields and return
                return "", None, None, chat_history
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Unexpected error in process_and_respond: {error_msg}", exc_info=True)
                
                # Add error message to chat history
                chat_history = chat_history or []
                chat_history.append({"role": "assistant", "content": f"❌ An error occurred: {error_msg}"})
                
                return "", None, None, chat_history
        
        # Function to prepare for processing
        def prepare_processing(text, image, audio, history):
            """Prepare the UI for processing.
            
            Args:
                text: User's text input
                image: Path to uploaded image file, if any
                audio: Path to uploaded audio file, if any
                history: Current chat history (list of dicts with 'role' and 'content')
                
            Returns:
                Tuple of (cleared_text, None, None, updated_history, loading_state)
            """
            try:
                # Validate that we have some input
                if not text and not image and not audio:
                    return "", None, None, history or [], False
                
                # Create user message based on input type
                if image:
                    user_message = text.strip() if text.strip() else "[Image]"
                elif audio:
                    user_message = text.strip() if text.strip() else "[Audio]"
                else:
                    user_message = text.strip()
                
                # Initialize history if needed
                updated_history = list(history) if history else []
                
                # Add user message with a placeholder for the assistant's response
                updated_history.append({"role": "user", "content": user_message})
                updated_history.append({"role": "assistant", "content": None})  # Placeholder for response
                
                # Log the processing start
                logger.info(f"Preparing to process input. Type: {'image' if image else 'audio' if audio else 'text'}")
                
                # Clear inputs and set loading state
                return "", None, None, updated_history, True
                
            except Exception as e:
                error_msg = f"Error in prepare_processing: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return "", None, None, history or [], False

        def create_submission_handler(input_component):
            """Create a submission handler for the given input component.
            
            Args:
                input_component: The input component that triggered the submission
                
            Returns:
                A handler function that processes the submission
            """
            def handler(*args):
                # Extract inputs from the event
                text, image, audio, history = args[:4]
                
                # Skip if no input
                if not text and not image and not audio:
                    logger.warning("Submission with no input")
                    return "", None, None, history or [], False
                
                # Create user message
                if image:
                    user_message = text.strip() if text.strip() else "[Image]"
                elif audio:
                    user_message = text.strip() if text.strip() else "[Audio]"
                else:
                    user_message = text.strip()
                
                # Add to history with new message format
                history = history or []
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": None})  # Placeholder for response
                
                # Clear inputs and show loading
                return "", None, None, history, True
            
            return handler
        
        # Common processing function for both submission methods
        def process_submission(text, image, audio, history):
            """Process the submission and generate a response.
            
            Args:
                text: User's text input
                image: Path to uploaded image file, if any
                audio: Path to uploaded audio file, if any
                history: Chat history in the new message format
                
            Returns:
                Tuple of (text, image, audio, history, loading_state)
            """
            try:
                # Call the existing process_and_respond function
                text, image, audio, history = process_and_respond(text, image, audio, history)
                return text, image, audio, history, False  # Set loading_state to False when done
            except Exception as e:
                error_msg = f"Error processing submission: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Update history with error message in the new format
                if history and len(history) > 0 and history[-1]["role"] == "assistant" and history[-1]["content"] is None:
                    history[-1]["content"] = f"❌ {error_msg}"
                else:
                    history = history or []
                    history.append({"role": "assistant", "content": f"❌ {error_msg}"})
                
                return "", None, None, history, False  # Set loading_state to False on error
        
        # Handle form submission via button click
        submit_btn.click(
            fn=prepare_processing,  # First prepare the UI for processing
            inputs=[text_input, image_input, audio_input, chatbot],
            outputs=[text_input, image_input, audio_input, chatbot, loading_state],
            queue=True
        ).then(
            fn=toggle_loading,  # Show loading state
            inputs=loading_state,
            outputs=[loading, submit_btn, text_input, image_input, audio_input, clear_btn],
            queue=False
        ).then(
            fn=process_submission,  # Process the actual request with proper loading state handling
            inputs=[text_input, image_input, audio_input, chatbot],
            outputs=[text_input, image_input, audio_input, chatbot, loading_state],
            queue=True
        ).then(
            fn=toggle_loading,  # Hide loading state
            inputs=gr.State(False),
            outputs=[loading, submit_btn, text_input, image_input, audio_input, clear_btn],
            queue=False
        )
        
        # Handle clear button
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[text_input, image_input, audio_input, chatbot],
            queue=False
        )
        
        # Handle tab changes
        def on_tab_change(tab_index):
            """Handle tab changes to show/hide the appropriate input fields."""
            logger.info(f"Tab changed to index: {tab_index}")
            # Show only the selected input field and hide others
            if tab_index == 0:  # Text tab
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif tab_index == 1:  # Image tab
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:  # Audio tab
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                
        # Initialize tabs if they exist in the UI
        if 'tabs' in locals():
            tabs.select(
                fn=on_tab_change,
                inputs=[tabs],
                outputs=[text_input, image_input, audio_input],
                queue=False
            )
        
        # Also submit when user presses Enter in the text input
        text_input.submit(
            fn=process_inputs,
            inputs=[text_input, image_input, audio_input, chatbot],
            outputs=[text_input, image_input, audio_input, chatbot, loading_state],
            queue=True
        ).then(
            fn=toggle_loading,
            inputs=gr.State(False),
            outputs=[
                loading,        # Hide loading indicator
                submit_btn,     # Re-enable submit button
                text_input,     # Re-enable text input
                image_input,    # Re-enable image input
                audio_input,    # Re-enable audio input
                clear_btn       # Re-enable clear button
            ],
            queue=False
        )
        
        # Clear button handler
        def clear_chat():
            """Clear all inputs and chat history.
            
            Returns:
                Tuple of (empty_text, None, None, empty_list, False) to reset the UI
            """
            logger.info("Clearing all inputs and chat history")
            return "", None, None, [], False  # text, image, audio, chat_history, loading_state
        
        # Set up clear button click handler
        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[
                text_input,     # Clear text input
                image_input,    # Clear image input
                audio_input,    # Clear audio input
                chatbot,        # Clear chat history
                loading_state   # Reset loading state
            ],
            queue=False
        ).then(
            fn=toggle_loading,
            inputs=gr.State(False),  # Ensure loading is off
            outputs=[
                loading,        # Hide loading indicator
                submit_btn,     # Ensure submit button is enabled
                text_input,     # Ensure text input is enabled
                image_input,    # Ensure image input is enabled
                audio_input,    # Ensure audio input is enabled
                clear_btn       # Ensure clear button is enabled
            ],
            queue=False
        )
        
        # Add quick suggestions
        suggestions = [
            "Tell me a joke",
            "Explain quantum computing in simple terms",
            "What's the weather like today?",
            "Can you help me with a coding problem?",
            "Tell me about artificial intelligence"
        ]
        
        # Function to handle suggestion clicks
        def on_suggestion_click(suggestion, history):
            """Handle suggestion button clicks.
            
            Args:
                suggestion: The suggested text to use as user input
                history: Current chat history in the new message format
                
            Returns:
                Tuple of ("", None, None, updated_history, True) to update the UI
            """
            # Initialize history if needed
            history = history or []
            
            # Add user message with the suggestion
            history.append({"role": "user", "content": suggestion})
            
            # Add placeholder for assistant's response
            history.append({"role": "assistant", "content": None})
            
            # Return updated state and show loading
            return "", None, None, history, True  # text, image, audio, chat_history, loading
        
        # Function to update loading state - moved to the top of the event handlers section
        def toggle_loading(loading_state: bool) -> List[Dict]:
            """Show or hide loading indicator and update UI elements.
            
            Args:
                loading_state: Whether to show the loading state
                
            Returns:
                List of component updates in the correct order for Gradio
            """
            try:
                logger.debug(f"Toggling loading state: {loading_state}")
                
                # Return updates in the exact order expected by Gradio
                return [
                    # Loading indicator visibility
                    gr.update(visible=loading_state),  # loading
                    # Submit button state
                    gr.update(
                        interactive=not loading_state,
                        variant="secondary" if loading_state else "primary"
                    ),  # submit_btn
                    # Input fields state
                    gr.update(interactive=not loading_state),  # text_input
                    gr.update(interactive=not loading_state),  # image_input
                    gr.update(interactive=not loading_state),  # audio_input
                    # Clear button state
                    gr.update(interactive=not loading_state)   # clear_btn
                ]
                
            except Exception as e:
                error_msg = f"Error in toggle_loading: {str(e)}"
                logger.error(error_msg, exc_info=True)
                # Return minimal updates on error in the correct order
                return [
                    gr.update(visible=loading_state),  # loading
                    gr.update(interactive=not loading_state),  # submit_btn
                    gr.update(interactive=not loading_state),  # text_input
                    gr.update(interactive=not loading_state),  # image_input
                    gr.update(interactive=not loading_state),  # audio_input
                    gr.update(interactive=not loading_state)   # clear_btn
                ]

        # Create suggestion buttons with proper loading states
        def create_suggestion_button(suggestion, is_visible=True):
            """Helper to create a suggestion button with consistent styling and behavior."""
            btn = gr.Button(
                suggestion,
                size="sm",
                variant="secondary",
                min_width=120,
                scale=1,
                visible=is_visible
            )
            
            # Set up click handler
            click_event = btn.click(
                fn=on_suggestion_click,
                inputs=[gr.Textbox(suggestion, visible=False), chatbot],
                outputs=[text_input, image_input, audio_input, chatbot, loading_state],
                queue=False
            )
            
            # Set up loading state
            click_event = click_event.then(
                fn=toggle_loading,
                inputs=loading_state,
                outputs=[
                    loading,        # Show loading indicator
                    submit_btn,     # Disable submit button
                    text_input,     # Disable text input
                    image_input,    # Disable image input
                    audio_input,    # Disable audio input
                    clear_btn,      # Disable clear button
                    btn             # Disable this button
                ],
                queue=False
            )
            
            # Process the response
            click_event = click_event.then(
                fn=process_and_respond,
                inputs=[gr.Textbox(suggestion, visible=False), None, None, chatbot],
                outputs=[text_input, image_input, audio_input, chatbot],
                queue=True
            )
            
            # Reset loading state
            click_event.then(
                fn=toggle_loading,
                inputs=gr.State(False),
                outputs=[
                    loading,        # Hide loading indicator
                    submit_btn,     # Re-enable submit button
                    text_input,     # Re-enable text input
                    image_input,    # Re-enable image input
                    audio_input,    # Re-enable audio input
                    clear_btn,      # Re-enable clear button
                    btn             # Re-enable this button
                ],
                queue=False
            )
            
            return btn
        
        # Suggestion buttons will be initialized after components are created
    
    return demo


def create_footer() -> gr.HTML:
    """Create a footer for the web interface."""
    return """
    <div class="footer">
        <p>FuseLLM - A simple yet powerful AI assistant</p>
        <div style="margin-top: 10px;">
            <a href="#" style="margin: 0 10px; color: #666; text-decoration: none;">API</a>
            <span>•</span>
            <a href="https://gradio.app" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">Made with Gradio</a>
            <span>•</span>
            <a href="#" style="margin: 0 10px; color: #666; text-decoration: none;">Settings</a>
        </div>
    </div>
    """

def launch_web_interface(server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch the Gradio web interface."""
    # Create the web interface
    demo = create_web_interface()
    
    # Add footer to the interface
    with demo:
        gr.HTML(create_footer())
    
    # Configure and launch the interface with minimal required parameters
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        debug=True  # Enable debug for development
    )


if __name__ == "__main__":
    launch_web_interface()
