import os
from dotenv import load_dotenv

print("Starting test...")

load_dotenv("backend/.env")
key = os.getenv("OPENROUTER_API_KEY")

if key:
    print(f"SUCCESS: {key[:20]}...")
else:
    print("FAILED: No key found")

