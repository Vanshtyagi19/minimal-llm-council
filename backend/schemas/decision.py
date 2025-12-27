"""
Structured schema for council decisions with confidence, risks, and citations
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class Citation(BaseModel):
    """Reference to source agent with confidence score"""
    source: str  # e.g., "agent_0", "agent_1", "agent_2"
    excerpt: str  # The specific text being cited
    confidence: float = Field(ge=0.0, le=1.0)

class Risk(BaseModel):
    """Identified risk in the decision"""
    category: str  # e.g., "factual_accuracy", "bias", "safety", "hallucination"
    severity: str  # "low", "medium", "high"
    description: str  # Detailed explanation

class AgentResponse(BaseModel):
    """Individual agent's response"""
    agent_id: str  # e.g., "agent_0"
    model: str  # e.g., "anthropic/claude-3.5-sonnet"
    response: str  # Full text response
    timestamp: datetime

class JudgeEvaluation(BaseModel):
    """Judge's evaluation with rubric scores"""
    judge_id: str  # e.g., "judge_0"
    model: str  # e.g., "openai/gpt-4o"
    rankings: List[str]  # Ordered list: ["agent_0", "agent_2", "agent_1"]
    scores: Dict[str, Dict[str, float]]  # {"agent_0": {"accuracy": 9, ...}}
    rationale: str  # Judge's reasoning

class DecisionObject(BaseModel):
    """Final structured decision with all metadata"""
    question: str
    final_answer: str
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in decision")
    
    # Raw data
    agent_responses: List[AgentResponse]
    judge_evaluations: List[JudgeEvaluation]
    
    # Analysis
    risks: List[Risk]
    citations: List[Citation]
    
    # Metadata
    timestamp: datetime
    audit_id: str  # UUID linking to audit log
    processing_time_seconds: Optional[float] = None
