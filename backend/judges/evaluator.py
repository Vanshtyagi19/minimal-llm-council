"""
Stage 2: Two judges evaluate agent responses using a structured rubric
Judges do NOT generate answers - they only compare and rank
"""
from typing import List
import json
from datetime import datetime
from backend.openrouter import query_model_json
from backend.schemas.decision import AgentResponse, JudgeEvaluation
from backend.config import JUDGE_MODELS

# Fixed 2-judge configuration


EVALUATION_RUBRIC = """You are an expert judge evaluating AI-generated responses.

Rate each response on a scale of 1-10 for these criteria:
1. **Factual Accuracy**: Correctness and precision of information
2. **Completeness**: Covers all aspects of the question thoroughly
3. **Clarity**: Well-structured, easy to understand
4. **Safety**: No harmful, biased, or misleading content
5. **Citation Quality**: Proper reasoning and evidence

You must return ONLY a valid JSON object with this exact structure:
{
    "rankings": ["agent_0", "agent_2", "agent_1"],
    "scores": {
        "agent_0": {
            "factual_accuracy": 9,
            "completeness": 8,
            "clarity": 9,
            "safety": 10,
            "citation_quality": 7
        },
        "agent_1": {...},
        "agent_2": {...}
    },
    "rationale": "Brief explanation of your rankings (2-3 sentences)"
}
"""

async def stage2_judge_evaluation(
    question: str,
    agent_responses: List[AgentResponse]
) -> List[JudgeEvaluation]:
    """
    Stage 2: Judges evaluate and rank agent responses
    
    Args:
        question: Original question for context
        agent_responses: List of 3 agent responses
        
    Returns:
        List of 2 JudgeEvaluation objects
    """
    print(f"[Stage 2] Sending to {len(JUDGE_MODELS)} judges for evaluation...")
    
    # Anonymize responses (hide which model generated what)
    anonymized_text = f"Question: {question}\n\n"
    for i, resp in enumerate(agent_responses):
        anonymized_text += f"Response {chr(65+i)} (agent_{i}):\n{resp.response}\n\n"
    
    judge_prompt = f"{EVALUATION_RUBRIC}\n\n{anonymized_text}"
    
    # Query judges in parallel
    import asyncio
    tasks = [
        query_model_json(
            model=model,
            prompt=judge_prompt,
            system_prompt="You are an impartial judge. Return only valid JSON."
        )
        for model in JUDGE_MODELS
    ]
    
    judge_outputs = await asyncio.gather(*tasks)
    
    # Structure evaluations
    evaluations = []
    for i, (model, output) in enumerate(zip(JUDGE_MODELS, judge_outputs)):
        try:
            # Check if output has required fields
            if isinstance(output, dict) and "rankings" in output:
                evaluations.append(JudgeEvaluation(
                    judge_id=f"judge_{i}",
                    model=model,
                    rankings=output["rankings"],
                    scores=output.get("scores", {}),
                    rationale=output.get("rationale", "No rationale provided")
                ))
            else:
                # Fallback: create default ranking if JSON parsing failed
                print(f"[WARNING] Judge {i} returned invalid format: {output}")
                print(f"[WARNING] Using fallback ranking for judge {i}")
                
                # Default ranking: alphabetical order
                default_rankings = [f"agent_{j}" for j in range(len(agent_responses))]
                
                evaluations.append(JudgeEvaluation(
                    judge_id=f"judge_{i}",
                    model=model,
                    rankings=default_rankings,
                    scores={f"agent_{j}": {
                        "factual_accuracy": 5,
                        "completeness": 5,
                        "clarity": 5,
                        "safety": 5,
                        "citation_quality": 5
                    } for j in range(len(agent_responses))},
                    rationale=f"Fallback ranking due to invalid JSON response: {str(output)[:100]}"
                ))
        except Exception as e:
            print(f"[ERROR] Failed to process judge {i}: {e}")
            # Create minimal fallback
            default_rankings = [f"agent_{j}" for j in range(len(agent_responses))]
            evaluations.append(JudgeEvaluation(
                judge_id=f"judge_{i}",
                model=model,
                rankings=default_rankings,
                scores={},
                rationale=f"Error processing judge response: {str(e)}"
            ))
    
    print(f"[Stage 2] Completed")
    return evaluations