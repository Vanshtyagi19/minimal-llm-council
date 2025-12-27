"""
Stage 3: Synthesize final decision from agent responses and judge evaluations
"""
import json
import uuid
from datetime import datetime
from typing import List
from backend.openrouter import query_model_json
from backend.schemas.decision import (
    AgentResponse, JudgeEvaluation, DecisionObject, Risk, Citation
)
from backend.config import CHAIRMAN_MODEL


async def stage3_synthesize_decision(
    question: str,
    agent_responses: List[AgentResponse],
    judge_evaluations: List[JudgeEvaluation],
    start_time: float
) -> DecisionObject:
    """
    Stage 3: Chairman synthesizes final decision with confidence and risk analysis
    
    Args:
        question: Original question
        agent_responses: 3 agent responses
        judge_evaluations: 2 judge evaluations
        start_time: Timestamp when processing started
        
    Returns:
        Complete DecisionObject
    """
    print(f"[Stage 3] Chairman synthesizing final decision...")
    
    # Prepare context for chairman
    context = {
        "question": question,
        "agent_responses": [
            {"agent_id": r.agent_id, "response": r.response}
            for r in agent_responses
        ],
        "judge_evaluations": [
            {
                "judge_id": e.judge_id,
                "rankings": e.rankings,
                "scores": e.scores,
                "rationale": e.rationale
            }
            for e in judge_evaluations
        ]
    }
    
    synthesis_prompt = f"""Given the question, agent responses, and judge evaluations, synthesize a final answer.

Context:
{json.dumps(context, indent=2)}

Requirements:
1. Create a comprehensive final answer incorporating the best insights
2. Calculate confidence score (0.0-1.0) based on judge agreement and response quality
3. Identify any risks (factual accuracy, bias, safety concerns)
4. Provide citations showing which agents contributed key points

Return a JSON object with this structure:
{{
    "final_answer": "Your synthesized answer here (2-3 paragraphs)",
    "confidence": 0.85,
    "risks": [
        {{
            "category": "factual_accuracy",
            "severity": "low",
            "description": "Minor discrepancy in dates between agents"
        }}
    ],
    "citations": [
        {{
            "source": "agent_0",
            "excerpt": "Key point from agent 0",
            "confidence": 0.9
        }}
    ]
}}
"""
    
    # Query chairman
    synthesis_data = await query_model_json(
        model=CHAIRMAN_MODEL,
        prompt=synthesis_prompt,
        system_prompt="You are a chairman synthesizing expert opinions. Return only valid JSON."
    )
    
    # Calculate processing time
    import asyncio
    processing_time = asyncio.get_event_loop().time() - start_time
    
    # Build DecisionObject
    if synthesis_data.get("fallback"):
        print(f"[WARNING] Using fallback synthesis due to API error")
        
        # Create simple fallback answer
        fallback_answer = f"Based on the responses: {agent_responses[0].response[:200]}..."

        
        decision = DecisionObject(
            question=question,
            final_answer=fallback_answer,
            confidence=0.5,  # Low confidence for fallback
            agent_responses=agent_responses,
            judge_evaluations=judge_evaluations,
            risks=[Risk(
                category="api_error",
                severity="medium",
                description="Chairman synthesis failed, using fallback"
            )],
            citations=[],
            timestamp=datetime.utcnow(),
            audit_id=str(uuid.uuid4()),
            processing_time_seconds=round(processing_time, 2)
        )
    else:
        # Normal path: use synthesis_data
        decision = DecisionObject(
            question=question,
            final_answer=synthesis_data.get("final_answer", "No answer generated"),
            confidence=synthesis_data.get("confidence", 0.5),
            agent_responses=agent_responses,
            judge_evaluations=judge_evaluations,
            risks=[Risk(**r) for r in synthesis_data.get("risks", [])],
            citations=[Citation(**c) for c in synthesis_data.get("citations", [])],
            timestamp=datetime.utcnow(),
            audit_id=str(uuid.uuid4()),
            processing_time_seconds=round(processing_time, 2)
        )
    
    print(f"[Stage 3] Completed - Confidence: {decision.confidence:.2%}")
    return decision