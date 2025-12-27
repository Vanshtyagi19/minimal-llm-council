"""
Persistent audit logging system for all council decisions
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional
from backend.schemas.decision import DecisionObject

class AuditLogger:
    """SQLite-based audit logger for decision tracking"""
    
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Initialize database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                confidence REAL NOT NULL,
                
                -- Serialized data
                agent_responses TEXT NOT NULL,
                judge_evaluations TEXT NOT NULL,
                risks TEXT NOT NULL,
                citations TEXT NOT NULL,
                
                -- Safety results
                safety_passed BOOLEAN NOT NULL,
                safety_violations TEXT,
                safety_warnings TEXT,
                
                -- Metadata
                processing_time_seconds REAL,
                user_feedback TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON audit_log(timestamp)
        """)
        
        self.conn.commit()
    
    async def log_decision(
        self,
        decision: DecisionObject,
        safety_result: Dict
    ) -> str:
        """
        Log a decision to the audit trail
        
        Args:
            decision: DecisionObject to log
            safety_result: Result from safety_check()
            
        Returns:
            audit_id of logged entry
        """
        # Use Pydantic's model_dump with mode='json' to handle datetime
        try:
            decision_dict = decision.model_dump(mode='json')
        except AttributeError:
            # Fallback for older Pydantic versions
            decision_dict = json.loads(decision.json())
        
        self.conn.execute("""
            INSERT INTO audit_log (
                audit_id, timestamp, question, final_answer, confidence,
                agent_responses, judge_evaluations, risks, citations,
                safety_passed, safety_violations, safety_warnings,
                processing_time_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.audit_id,
            decision_dict["timestamp"],
            decision.question,
            decision.final_answer,
            decision.confidence,
            json.dumps(decision_dict["agent_responses"]),
            json.dumps(decision_dict["judge_evaluations"]),
            json.dumps(decision_dict["risks"]),
            json.dumps(decision_dict["citations"]),
            safety_result["passed"],
            json.dumps(safety_result["violations"]),
            json.dumps(safety_result["warnings"]),
            decision.processing_time_seconds
        ))
        
        self.conn.commit()
        print(f"[Audit] Logged decision {decision.audit_id}")
        return decision.audit_id
    
    def get_decision(self, audit_id: str) -> Optional[Dict]:
        """Retrieve a logged decision by ID"""
        cursor = self.conn.execute(
            "SELECT * FROM audit_log WHERE audit_id = ?",
            (audit_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    
    def get_recent_decisions(self, limit: int = 10) -> list:
        """Get most recent decisions"""
        cursor = self.conn.execute(
            "SELECT audit_id, timestamp, question, confidence, safety_passed "
            "FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()
    
    def get_statistics(self) -> dict:
        """Get overall audit statistics"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_decisions,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN safety_passed = 1 THEN 1 ELSE 0 END) as passed_count,
                SUM(CASE WHEN safety_passed = 0 THEN 1 ELSE 0 END) as rejected_count,
                AVG(processing_time_seconds) as avg_processing_time
            FROM audit_log
        """)
        
        row = cursor.fetchone()
        
        if not row or row[0] == 0:
            return {
                "total_decisions": 0,
                "avg_confidence": 0.0,
                "pass_rate": 0.0,
                "avg_processing_time": 0.0
            }
        
        total, avg_conf, passed, rejected, avg_time = row
        
        return {
            "total_decisions": total,
            "passed_decisions": passed,
            "rejected_decisions": rejected,
            "pass_rate": (passed / total) if total > 0 else 0.0,
            "avg_confidence": round(avg_conf, 3) if avg_conf else 0.0,
            "avg_processing_time_seconds": round(avg_time, 2) if avg_time else 0.0
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
