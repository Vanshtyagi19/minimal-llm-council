# Minimal LLM Council

A production-ready multi-agent LLM system with safety gating and quality control.

## Architecture

- **3 Specialist Agents**: Generate diverse responses using different models
- **2 Judge Agents**: Score and evaluate responses
- **Safety Gate**: Validates output quality before delivery

## Features

- Multi-model support (OpenAI, Anthropic, Gemini)
- Async processing with retry logic
- Configurable thresholds
- Production logging

## Usage

from llm_council import run_council

result = await run_council("Your prompt here")
## Setup

pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python main.py