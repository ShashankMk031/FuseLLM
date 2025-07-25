# # main.py

# import argparse
# from core.orchestrator import run_fuse_pipeline

# def main():
#     parser = argparse.ArgumentParser(description="FuseLLM Assistant CLI")
#     parser.add_argument("--input", type=str, required=True, help="Path to input file or a query string.")
#     args = parser.parse_args()

#     try:
#         result = run_fuse_pipeline(args.input)
#         print("\n=== Final Output ===")
#         print(result)
#     except Exception as e:
#         print("[Error] Failed to process input:", e)

# if __name__ == "__main__":
#     main()
 
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from src.core.orchestrator import run_fuse_pipeline
from src.utils.log_utils import log_event


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FuseLLM - A versatile AI pipeline system")
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to process",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to process",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file to process",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the web interface",
    )
    return parser.parse_args()


def run_cli():
    """Run the command-line interface."""
    print("Welcome to FuseLLM CLI!")
    print("Type 'exit' or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ('exit', 'quit'):
                print("👋 Exiting FuseLLM. Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\n🤖 Processing...")
            response = run_fuse_pipeline(user_input)
            print(f"\n🤖 FuseLLM: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting FuseLLM. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def run_web_interface():
    """Run the web interface."""
    from src.fuse_app import launch_web_interface
    launch_web_interface()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    try:
        if args.web:
            run_web_interface()
        elif args.text:
            response = run_fuse_pipeline(args.text)
            print(response)
        elif args.image:
            # Handle image processing
            response = run_fuse_pipeline(args.image, input_type="image")
            print(response)
        elif args.audio:
            # Handle audio processing
            response = run_fuse_pipeline(args.audio, input_type="audio")
            print(response)
        else:
            # Interactive CLI mode
            run_cli()
            
    except Exception as e:
        log_event("ERROR", f"Application error: {str(e)}", level="error")
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

    if mode == "cli":
        from core.orchestrator import orchestrate
        ...
    elif mode == "web":
        os.system("python fuse_app.py")
