"""
Safety gating system to filter unsafe or low-quality decisions
"""
import re
from typing import Dict, List
from backend.schemas.decision import DecisionObject

# Pattern-based safety checks
UNSAFE_PATTERNS = [
    (r'\b(kill|murder|suicide|harm yourself)\b', "violence"),
    (r'\b(illegal|hack|crack|pirate)\b', "illegal_activity"),
    (r'\b(racist|sexist|discriminat)\b', "discrimination"),
]

# Minimum confidence threshold
MIN_CONFIDENCE = 0.5

async def safety_check(decision: DecisionObject) -> Dict:
    """
    Run safety checks on the final decision
    
    Args:
        decision: DecisionObject to evaluate
        
    Returns:
        Dict with "passed" bool, "violations" list, "warnings" list
    """
    print("[Safety] Running safety gate checks...")
    
    violations = []
    warnings = []
    
    # Check 1: Pattern matching for unsafe content
    for pattern, violation_type in UNSAFE_PATTERNS:
        if re.search(pattern, decision.final_answer, re.IGNORECASE):
            violations.append({
                "type": violation_type,
                "reason": f"Detected unsafe pattern: {violation_type}"
            })
    
    # Check 2: Confidence threshold
    if decision.confidence < MIN_CONFIDENCE:
        warnings.append({
            "type": "low_confidence",
            "value": decision.confidence,
            "threshold": MIN_CONFIDENCE,
            "reason": f"Confidence {decision.confidence:.2%} below threshold {MIN_CONFIDENCE:.2%}"
        })
    
    # Check 3: High-severity risks
    high_risks = [r for r in decision.risks if r.severity == "high"]
    if high_risks:
        violations.append({
            "type": "high_risk_detected",
            "count": len(high_risks),
            "risks": [r.description for r in high_risks]
        })
    
    # Check 4: Judge disagreement
    if len(decision.judge_evaluations) >= 2:
        rank1 = decision.judge_evaluations[0].rankings
        rank2 = decision.judge_evaluations[1].rankings
        
        # If top-ranked agents are different
        if rank1[0] != rank2[0]:
            warnings.append({
                "type": "judge_disagreement",
                "reason": "Judges disagreed on best response",
                "judge_0_top": rank1[0],
                "judge_1_top": rank2[0]
            })
    
    passed = len(violations) == 0
    
    print(f"[Safety] {'PASSED' if passed else 'FAILED'} - {len(violations)} violations, {len(warnings)} warnings")
    
    return {
        "passed": passed,
        "violations": violations,
        "warnings": warnings,
        "confidence": decision.confidence
    }
