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
 
from core.orchestrator import orchestrate

def main():
    print("Welcome to FuseLLM CLI!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input(">> ")

        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting FuseLLM.")
            break

        try:
            response = orchestrate(user_input)
            print("\n📤 Response:")
            print(response)
            print("\n" + "-" * 40 + "\n")
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
