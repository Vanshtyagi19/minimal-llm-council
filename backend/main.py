"""
Main FastAPI application orchestrating the LLM Council
Modified from Karpathy's original to implement:
- 3 fixed generator agents
- 2 judge evaluators (no generation)
- Structured Decision Object output
- Safety gating
- Persistent audit logging
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
from datetime import datetime
from typing import Optional

# Import our new components
from backend.agents.generator import stage1_generate_answers
from backend.judges.evaluator import stage2_judge_evaluation
from backend.agents.synthesizer import stage3_synthesize_decision
from backend.safety.gate import safety_check
from backend.audit.logger import AuditLogger

# Initialize FastAPI app
app = FastAPI(
    title="LLM Council API",
    version="2.0",
    description="Minimal LLM Council with 3 agents, 2 judges, safety gating, and audit logging"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize audit logger (singleton)
audit_logger = AuditLogger()

# Request/Response models
class QuestionRequest(BaseModel):
    """Request schema for deliberation"""
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main benefits of regular exercise?"
            }
        }

class CouncilResponse(BaseModel):
    """Response schema for deliberation"""
    status: str  # "approved" or "rejected"
    decision: Optional[dict] = None
    audit_id: str
    reason: Optional[str] = None
    processing_time_seconds: Optional[float] = None

class AuditQueryResponse(BaseModel):
    """Response schema for audit retrieval"""
    found: bool
    data: Optional[dict] = None

# ============================================================================
# Main Endpoints
# ============================================================================

@app.post("/api/council/deliberate", response_model=CouncilResponse)
async def deliberate(request: QuestionRequest):
    """
    Main endpoint: Run full 3-stage council deliberation
    
    Flow:
    1. Stage 1: 3 agents generate independent answers
    2. Stage 2: 2 judges evaluate responses using rubric (no generation)
    3. Stage 3: Chairman synthesizes final decision with confidence & risks
    4. Safety gating checks
    5. Persistent audit logging
    
    Returns:
        CouncilResponse with status, decision object, and audit ID
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        print(f"\n{'='*70}")
        print(f"[Council] New deliberation request")
        print(f"[Council] Question: {request.question}")
        print(f"{'='*70}\n")
        
        # Validate question
        if not request.question or len(request.question.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Question must be at least 10 characters long"
            )
        
        # Stage 1: Generator agents create independent responses
        print("[Council] Starting Stage 1: Agent Generation")
        agent_responses = await stage1_generate_answers(request.question)
        print(f"[Council] ✓ Stage 1 complete: {len(agent_responses)} responses")
        
        # Stage 2: Judge evaluation with rubric
        print("[Council] Starting Stage 2: Judge Evaluation")
        judge_evaluations = await stage2_judge_evaluation(
            question=request.question,
            agent_responses=agent_responses
        )
        print(f"[Council] ✓ Stage 2 complete: {len(judge_evaluations)} evaluations")
        
        # Stage 3: Chairman synthesis
        print("[Council] Starting Stage 3: Decision Synthesis")
        decision = await stage3_synthesize_decision(
            question=request.question,
            agent_responses=agent_responses,
            judge_evaluations=judge_evaluations,
            start_time=start_time
        )
        print(f"[Council] ✓ Stage 3 complete: Confidence {decision.confidence:.2%}")
        
        # Safety gating
        print("[Council] Running safety gate checks")
        safety_result = await safety_check(decision)
        
        # Audit logging (log regardless of safety result)
        print("[Council] Logging to audit trail")
        await audit_logger.log_decision(decision, safety_result)
        
        # Calculate total processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Check if decision passed safety
        if not safety_result["passed"]:
            print(f"[Council] ❌ Decision REJECTED due to safety violations")
            print(f"[Council] Violations: {safety_result['violations']}")
            
            return CouncilResponse(
                status="rejected",
                audit_id=decision.audit_id,
                reason=f"Safety gate failed: {len(safety_result['violations'])} violation(s) detected",
                processing_time_seconds=round(processing_time, 2)
            )
        
        # Success - return approved decision
        print(f"[Council] ✅ Decision APPROVED")
        print(f"[Council] Audit ID: {decision.audit_id}")
        print(f"[Council] Total processing time: {processing_time:.2f}s\n")
        
        # Add warnings to decision if any
        decision_dict = decision.dict()
        if safety_result.get("warnings"):
            decision_dict["safety_warnings"] = safety_result["warnings"]
        
        return CouncilResponse(
            status="approved",
            decision=decision_dict,
            audit_id=decision.audit_id,
            processing_time_seconds=round(processing_time, 2)
        )
    
    except ValueError as e:
        # Handle expected errors (e.g., JSON parsing failures)
        print(f"[Council] ❌ ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Handle unexpected errors
        print(f"[Council] ❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ============================================================================
# Audit Endpoints
# ============================================================================

@app.get("/api/audit/{audit_id}", response_model=AuditQueryResponse)
async def get_audit(audit_id: str):
    """
    Retrieve a logged decision by audit ID
    
    Args:
        audit_id: UUID of the decision to retrieve
        
    Returns:
        Complete decision object with all metadata
    """
    try:
        decision = audit_logger.get_decision(audit_id)
        
        if not decision:
            return AuditQueryResponse(
                found=False,
                data=None
            )
        
        # Parse JSON fields
        decision_parsed = {
            **decision,
            "agent_responses": json.loads(decision["agent_responses"]),
            "judge_evaluations": json.loads(decision["judge_evaluations"]),
            "risks": json.loads(decision["risks"]),
            "citations": json.loads(decision["citations"]),
            "safety_violations": json.loads(decision["safety_violations"]) if decision["safety_violations"] else [],
            "safety_warnings": json.loads(decision["safety_warnings"]) if decision["safety_warnings"] else []
        }
        
        return AuditQueryResponse(
            found=True,
            data=decision_parsed
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audit/recent/{limit}")
async def get_recent_decisions(limit: int = 10):
    """
    Get most recent decisions from audit log
    
    Args:
        limit: Number of recent decisions to retrieve (default: 10, max: 100)
        
    Returns:
        List of recent decisions with summary info
    """
    if limit > 100:
        limit = 100
    
    try:
        recent = audit_logger.get_recent_decisions(limit=limit)
        
        return {
            "count": len(recent),
            "decisions": [
                {
                    "audit_id": row[0],
                    "timestamp": row[1],
                    "question": row[2],
                    "confidence": row[3],
                    "safety_passed": bool(row[4])
                }
                for row in recent
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audit/stats")
async def get_audit_stats():
    """
    Get overall statistics from audit log
    
    Returns:
        Summary statistics including total decisions, pass rate, etc.
    """
    try:
        stats = audit_logger.get_statistics()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0"
    }

@app.get("/api/info")
async def get_info():
    """
    Get information about council configuration
    """
    from backend.agents.generator import GENERATOR_MODELS
    from backend.judges.evaluator import JUDGE_MODELS
    from backend.agents.synthesizer import CHAIRMAN_MODEL
    
    return {
        "council_config": {
            "generator_agents": {
                "count": len(GENERATOR_MODELS),
                "models": GENERATOR_MODELS
            },
            "judge_models": {
                "count": len(JUDGE_MODELS),
                "models": JUDGE_MODELS
            },
            "chairman_model": CHAIRMAN_MODEL
        },
        "features": {
            "safety_gating": True,
            "audit_logging": True,
            "confidence_scoring": True,
            "risk_analysis": True,
            "citations": True
        }
    }

@app.get("/")
async def root():
    """
    Root endpoint with API documentation link
    """
    return {
        "message": "LLM Council API v2.0",
        "documentation": "/docs",
        "health": "/health",
        "main_endpoint": "/api/council/deliberate"
    }

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on startup
    """
    print("\n" + "="*70)
    print("🏛️  LLM COUNCIL API v2.0")
    print("="*70)
    print("✓ Audit logger initialized")
    print("✓ 3 generator agents configured")
    print("✓ 2 judge evaluators configured")
    print("✓ Safety gating enabled")
    print("✓ API ready for requests")
    print("="*70 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources on shutdown
    """
    print("\n[Shutdown] Closing audit logger connection...")
    audit_logger.close()
    print("[Shutdown] Complete\n")

# ============================================================================
# Run Server (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
