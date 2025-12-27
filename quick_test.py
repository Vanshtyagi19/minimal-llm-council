"""
Quick test to verify all components work
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv("backend/.env")

async def test_full_flow():
    """Test complete council flow"""
    
    print("\n" + "="*70)
    print("🧪 TESTING LLM COUNCIL COMPONENTS")
    print("="*70 + "\n")
    
    # Test 1: Check API key
    print("[Test 1] Checking OpenRouter API key...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ API key not found! Check your .env file")
        return
    
    # Test 2: Import check
    print("\n[Test 2] Importing modules...")
    try:
        from backend.schemas.decision import DecisionObject, AgentResponse
        from backend.agents.generator import stage1_generate_answers
        from backend.judges.evaluator import stage2_judge_evaluation
        from backend.agents.synthesizer import stage3_synthesize_decision
        from backend.safety.gate import safety_check
        from backend.audit.logger import AuditLogger
        print("✓ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test 3: Simple question
    question = "What is 2+2?"
    print(f"\n[Test 3] Testing with simple question: '{question}'")
    print("This will make real API calls to OpenRouter...\n")
    
    try:
        # Stage 1
        print("Stage 1: Generating agent responses...")
        start = asyncio.get_event_loop().time()
        agents = await stage1_generate_answers(question)
        print(f"✓ Got {len(agents)} agent responses")
        for agent in agents:
            print(f"  - {agent.agent_id} ({agent.model}): {agent.response[:80]}...")
        
        # WAIT for rate limits
        print("\n⏳ Waiting 30 seconds to avoid rate limits...")
        await asyncio.sleep(30)
        
        # Stage 2
        print("\nStage 2: Judge evaluation...")
        judges = await stage2_judge_evaluation(question, agents)
        print(f"✓ Got {len(judges)} judge evaluations")
        for judge in judges:
            print(f"  - {judge.judge_id}: Rankings = {judge.rankings}")
        
        # WAIT again
        print("\n⏳ Waiting 30 seconds...")
        await asyncio.sleep(30)
        
        # Stage 3
        print("\nStage 3: Synthesizing decision...")
        decision = await stage3_synthesize_decision(question, agents, judges, start)
        print(f"✓ Decision created")
        print(f"  - Confidence: {decision.confidence:.1%}")
        print(f"  - Final answer: {decision.final_answer[:100]}...")
        
        # Safety
        print("\nRunning safety checks...")
        safety = await safety_check(decision)
        print(f"✓ Safety: {'PASSED ✅' if safety['passed'] else 'FAILED ❌'}")
        if safety['violations']:
            print(f"  - Violations: {safety['violations']}")
        if safety['warnings']:
            print(f"  - Warnings: {len(safety['warnings'])} warning(s)")
        
        # Audit
        print("\nLogging to audit...")
        logger = AuditLogger()
        audit_id = await logger.log_decision(decision, safety)
        print(f"✓ Logged with audit ID: {audit_id}")
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print(f"\nFinal Answer: {decision.final_answer}")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Processing time: {decision.processing_time_seconds:.2f}s")
        print(f"Audit ID: {audit_id}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_flow())
