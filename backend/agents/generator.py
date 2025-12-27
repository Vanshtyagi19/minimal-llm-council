"""
Stage 1: Three independent agents generate answers
"""
from typing import List
from datetime import datetime
import asyncio
from backend.openrouter import query_models_parallel
from backend.schemas.decision import AgentResponse
from backend.config import GENERATOR_MODELS
# Fixed 3-agent configuration


async def stage1_generate_answers(question: str) -> List[AgentResponse]:
    """
    Stage 1: Three agents independently generate answers to the question
    
    Args:
        question: User's question
        
    Returns:
        List of 3 AgentResponse objects
    """
    print(f"[Stage 1] Querying {len(GENERATOR_MODELS)} generator agents...")
    
    # System prompt for generators
    system_prompt = """You are an expert analyst providing a comprehensive answer.
    Be accurate, cite reasoning, and acknowledge uncertainties."""
    
    # Query all models in parallel
    start_time = asyncio.get_event_loop().time()
    
    responses = await query_models_parallel(
        models=GENERATOR_MODELS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    
    # Structure responses
    agent_responses = []
    for i, (model, response) in enumerate(zip(GENERATOR_MODELS, responses)):
        agent_responses.append(AgentResponse(
            agent_id=f"agent_{i}",
            model=model,
            response=response,
            timestamp=datetime.utcnow()
        ))
    
    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"[Stage 1] Completed in {elapsed:.2f}s")
    
    return agent_responses
