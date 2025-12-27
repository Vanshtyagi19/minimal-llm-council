"""
Gradio interface for LLM Council - Hugging Face Spaces deployment
All-in-one file that runs the entire backend
"""
import gradio as gr
import asyncio
import json
from datetime import datetime

# Import backend components
from backend.agents.generator import stage1_generate_answers
from backend.judges.evaluator import stage2_judge_evaluation
from backend.agents.synthesizer import stage3_synthesize_decision
from backend.safety.gate import safety_check
from backend.audit.logger import AuditLogger

# Initialize audit logger
audit_logger = AuditLogger()

async def process_question(question: str, progress=gr.Progress()) -> str:
    """
    Main processing function called when user submits a question
    """
    if not question.strip():
        return "❌ Please enter a question"
    
    try:
        progress(0, desc="Starting...")
        start_time = asyncio.get_event_loop().time()
        
        # Stage 1: Agents generate
        progress(0.2, desc="Stage 1: Querying 3 AI agents...")
        agent_responses = await stage1_generate_answers(question)
        
        # Wait between stages for free tier rate limits
        await asyncio.sleep(5)
        
        # Stage 2: Judges evaluate
        progress(0.5, desc="Stage 2: Judges evaluating responses...")
        judge_evaluations = await stage2_judge_evaluation(question, agent_responses)
        
        await asyncio.sleep(5)
        
        # Stage 3: Synthesize
        progress(0.7, desc="Stage 3: Synthesizing final decision...")
        decision = await stage3_synthesize_decision(
            question, agent_responses, judge_evaluations, start_time
        )
        
        # Safety check
        progress(0.9, desc="Running safety checks...")
        safety_result = await safety_check(decision)
        
        # Audit log
        await audit_logger.log_decision(decision, safety_result)
        
        progress(1.0, desc="Complete!")
        
        # Format output
        if not safety_result["passed"]:
            return f"""
# ❌ Decision Rejected

**Audit ID:** `{decision.audit_id}`

**Reason:** Safety violations detected

**Violations:**
{json.dumps(safety_result['violations'], indent=2)}

"""
        
        # Build success output
        output = f"""
# ✅ Council Decision

## Final Answer

{decision.final_answer}

---

### 📊 Metadata

| Metric | Value |
|--------|-------|
| **Confidence** | {decision.confidence:.1%} |
| **Processing Time** | {decision.processing_time_seconds:.2f}s |
| **Audit ID** | `{decision.audit_id}` |

---

### 🤖 Agent Responses

"""
        for resp in decision.agent_responses:
            output += f"""
**{resp.agent_id}** ({resp.model}):
> {resp.response[:300]}{'...' if len(resp.response) > 300 else ''}

"""
        
        output += """
---

### ⚖️ Judge Evaluations

"""
        for eval in decision.judge_evaluations:
            output += f"""
**{eval.judge_id}** ({eval.model}):
- **Rankings:** {' > '.join(eval.rankings)}
- **Rationale:** {eval.rationale}

"""
        
        if decision.risks:
            output += """
---

### ⚠️ Identified Risks

"""
            for risk in decision.risks:
                emoji = "🔴" if risk.severity == "high" else "🟡" if risk.severity == "medium" else "🟢"
                output += f"""
{emoji} **{risk.category.replace('_', ' ').title()}** ({risk.severity})
> {risk.description}

"""
        
        if decision.citations:
            output += """
---

### 📚 Citations

"""
            for citation in decision.citations:
                output += f"""
- **{citation.source}** (confidence: {citation.confidence:.0%}): {citation.excerpt}
"""
        
        if safety_result.get("warnings"):
            output += f"""
---

### ⚠️ Warnings

{len(safety_result['warnings'])} warning(s) detected
"""
        
        return output
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
# ❌ Error

An error occurred while processing your request:


**Details:**
{error_details}

Please try again or contact support if the issue persists.
"""

def run_council(question):
    """Wrapper to run async function in Gradio"""
    return asyncio.run(process_question(question))

with gr.Blocks(theme=gr.themes.Soft(), title="🏛️ LLM Council") as demo:
    gr.Markdown("""
    # 🏛️ LLM Council
    
    **Multi-agent deliberation system with judges and safety gating**
    
    - 🤖 **3 independent AI agents** generate answers
    - ⚖️ **2 judge models** evaluate using a structured rubric
    - 🎯 **Chairman** synthesizes final decision with confidence scoring
    - 🛡️ **Safety gating** prevents harmful outputs
    - 📝 **Persistent audit logging** for all decisions
    
    *Powered by OpenRouter (Claude, GPT-4, Gemini)*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a Question",
                placeholder="Enter your question here...",
                lines=4
            )
            submit_btn = gr.Button("🚀 Deliberate", variant="primary", size="lg")
    
    output_display = gr.Markdown(label="Council Decision")
    
    gr.Examples(
        examples=[
            ["What are the main causes of climate change?"],
            ["Explain quantum computing in simple terms"],
            ["What are the pros and cons of remote work?"],
            ["How can I improve my productivity?"],
            ["What is the future of artificial intelligence?"],
            ["What are three benefits of regular exercise?"],
            ["Explain the concept of blockchain technology"],
            ["What are the ethical considerations of AI development?"]
        ],
        inputs=question_input
    )
    
    submit_btn.click(
        fn=run_council,
        inputs=question_input,
        outputs=output_display
    )
    
    gr.Markdown("""
    ---
    
    ### 🏗️ Architecture
    
    This system implements a minimal LLM Council architecture:
    
    1. **Stage 1 - Generation:** 3 agents generate independent answers
    2. **Stage 2 - Evaluation:** 2 judges evaluate responses using rubric criteria (accuracy, completeness, clarity, safety, citations)
    3. **Stage 3 - Synthesis:** Chairman synthesizes final decision with confidence score and risk analysis
    4. **Safety Layer:** Filters unsafe/low-quality decisions based on patterns and confidence thresholds
    5. **Audit Trail:** SQLite database logs all decisions for transparency
    
    ### 📋 Features
    
    - ✅ Structured Decision Object with confidence scoring
    - ✅ Risk identification and severity classification
    - ✅ Citation tracking from agent sources
    - ✅ Fallback handling for API failures
    - ✅ Rate limit management with retry logic
    - ✅ Complete audit trail
    
    ### 🔗 Links
    
    [GitHub Repository](#) | [Documentation](#) | [API Reference](#)
    
    ---
    
    *Note: Due to free tier rate limits, each deliberation may take 30-60 seconds. For production use, consider adding credits to OpenRouter.*
    """)

if __name__ == "__main__":
    demo.launch()
