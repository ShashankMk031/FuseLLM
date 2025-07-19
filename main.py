# main.py

import argparse
from core.orchestrator import run_fuse_pipeline

def main():
    parser = argparse.ArgumentParser(description="FuseLLM Assistant CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input file or a query string.")
    args = parser.parse_args()

    try:
        result = run_fuse_pipeline(args.input)
        print("\n=== Final Output ===")
        print(result)
    except Exception as e:
        print("[Error] Failed to process input:", e)

if __name__ == "__main__":
    main()
