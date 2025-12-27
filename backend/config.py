"""
Configuration management for LLM Council
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("backend/.env")
load_dotenv()  # Also try root

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment. Check your .env file")

# Model configuration - FREE MODELS for testing
GENERATOR_MODELS = [
    os.getenv("GENERATOR_MODEL_1", "meta-llama/llama-3.2-3b-instruct:free"),
    os.getenv("GENERATOR_MODEL_2", "google/gemini-2.0-flash-exp:free"),
    os.getenv("GENERATOR_MODEL_3", "mistralai/mistral-7b-instruct:free"),
]

JUDGE_MODELS = [
    os.getenv("JUDGE_MODEL_1", "meta-llama/llama-3.2-3b-instruct:free"),
    os.getenv("JUDGE_MODEL_2", "google/gemini-2.0-flash-exp:free"),
]

CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "google/gemini-2.0-flash-exp:free")

# Safety configuration
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE", "0.5"))

# Audit configuration
AUDIT_DB_PATH = os.getenv("AUDIT_DB_PATH", "audit.db")
