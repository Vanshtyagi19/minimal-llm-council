"""OpenRouter API client for making LLM requests."""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API with retry logic.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                data = response.json()
                message = data['choices'][0]['message']

                return {
                    'content': message.get('content'),
                    'reasoning_details': message.get('reasoning_details')
                }

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            # Handle rate limiting (429)
            if status_code == 429 and attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[RETRY] Rate limited on {model}, waiting {delay}s before retry {attempt + 2}/{max_retries}")
                await asyncio.sleep(delay)
                continue
            
            # Handle payment required (402)
            elif status_code == 402:
                print(f"Error querying model {model}: Payment required - add credits at https://openrouter.ai/credits")
                return None
            
            # Handle not found (404)
            elif status_code == 404:
                print(f"Error querying model {model}: Model not found - check model name")
                return None
            
            # Other HTTP errors
            else:
                print(f"Error querying model {model}: HTTP {status_code} - {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None

        except httpx.TimeoutException:
            print(f"Error querying model {model}: Request timeout")
            if attempt < max_retries - 1:
                print(f"[RETRY] Retrying {attempt + 2}/{max_retries}")
                await asyncio.sleep(1)
                continue
            return None

        except Exception as e:
            print(f"Error querying model {model}: {e}")
            if attempt < max_retries - 1:
                print(f"[RETRY] Retrying {attempt + 2}/{max_retries}")
                await asyncio.sleep(1)
                continue
            return None

    print(f"[FAILED] Model {model} failed after {max_retries} attempts")
    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
 
    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


# Add this to the END of existing openrouter.py file

async def query_model_json(
    model: str, 
    prompt: str, 
    system_prompt: str = None,
    max_retries: int = 3
) -> dict:
    """
    Query a single model with JSON response format.
    Used for structured outputs from judges and synthesizer.

    Args:
        model: OpenRouter model identifier
        prompt: User prompt
        system_prompt: Optional system prompt
        max_retries: Maximum retry attempts

    Returns:
        Parsed JSON dict, or fallback dict on error
    """
    import json
    import re
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Call query_model with retry logic
    response = await query_model(
        model=model,
        messages=messages,
        max_retries=max_retries
    )
    
    # Handle None response (API error) - RETURN FALLBACK, DON'T RAISE
     # Handle None response (API error) - RETURN FALLBACK, DON'T RAISE
    if response is None:
        print(f"[WARNING] Got None response from {model}, returning fallback")
        return {
            # For judges
            "rankings": ["agent_0", "agent_1", "agent_2"],
            "scores": {},
            "rationale": "API error - using default ranking",
            # For synthesizer
            "final_answer": "Unable to generate synthesis due to API rate limits",
            "confidence": 0.3,
            "risks": [],
            "citations": [],
            # Flag
            "fallback": True
        }
    
    # Extract content from response dict
    if isinstance(response, dict) and 'content' in response:
        content = response['content']
    else:
        content = response
    
    # Handle None content - RETURN FALLBACK
    if content is None:
        print(f"[WARNING] Got None content from {model}, returning fallback")
        return {
            "rankings": ["agent_0", "agent_1", "agent_2"],
            "scores": {},
            "rationale": "Empty response - using default ranking",
            "final_answer": "Unable to generate synthesis due to API rate limits",
            "confidence": 0.3,
            "risks": [],
            "citations": [],
            "fallback": True
        }
    
    # If content is already a dict, return it
    if isinstance(content, dict):
        return content
    
    # If it's a string, try to parse as JSON
    if isinstance(content, str):
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        json_match = re.search(r'``````', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding any JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Final fallback: return default structure
        print(f"[WARNING] Could not parse JSON from {model}, using fallback")
        return {
            "rankings": ["agent_0", "agent_1", "agent_2"],
            "scores": {},
            "rationale": f"Failed to parse JSON - text response: {content[:200]}",
            "fallback": True
        }
    
    # Unexpected type - RETURN FALLBACK, DON'T RAISE
    print(f"[WARNING] Unexpected response type from {model}: {type(content)}")
    return {
        "rankings": ["agent_0", "agent_1", "agent_2"],
        "scores": {},
        "rationale": f"Unexpected type {type(content)} - using default",
        "fallback": True
    }
