#!/usr/bin/env python3
"""
Demo Script: Session Intelligence - Relatedness, Forgetting, Context Routing

Demonstrates both follow-up and new-topic flows with expected logs.

Usage:
    python scripts/demo_session_intelligence.py
    
Or with curl:
    # See curl examples at bottom of file
"""
import os
import sys
import json
import time
import requests
from datetime import datetime

BASE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_response(resp: dict, label: str = "Response"):
    print(f"\n{label}:")
    print(json.dumps(resp, indent=2, default=str))


def create_session(user_id: str = "demo_user") -> str:
    """Create a new session and return session_id"""
    resp = requests.post(
        f"{BASE_URL}/api/session/create",
        json={"user_id": user_id}
    )
    data = resp.json()
    if data.get("success"):
        return data["session"]["session_id"]
    raise Exception(f"Failed to create session: {data}")


def query_session(
    session_id: str, 
    question: str, 
    force_session_prioritize: bool = False
) -> dict:
    """Query with session context"""
    resp = requests.post(
        f"{BASE_URL}/api/session/{session_id}/query",
        json={
            "user_id": "demo_user",
            "question": question,
            "force_session_prioritize": force_session_prioritize
        }
    )
    return resp.json()


def get_status(session_id: str) -> dict:
    """Get session status"""
    resp = requests.get(f"{BASE_URL}/api/session/{session_id}/status")
    return resp.json()


def reset_session(session_id: str) -> dict:
    """Reset session context"""
    resp = requests.post(f"{BASE_URL}/api/session/{session_id}/reset")
    return resp.json()


def main():
    print_header("Session Intelligence Demo")
    print(f"API URL: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Create session
    print_header("Step 1: Create Session")
    session_id = create_session()
    print(f"✅ Session created: {session_id}")
    
    # ==========================================================================
    # Scenario 1: Follow-up Related
    # ==========================================================================
    print_header("Scenario 1: Follow-up Related")
    
    print("Q1: Explain the Pythagoras theorem")
    r1 = query_session(session_id, "Explain the Pythagoras theorem")
    print(f"   Decision: {r1.get('session_context_decision')}")
    print(f"   Similarity: {r1.get('max_similarity', 0):.2f}")
    print(f"   Request ID: {r1.get('request_id')}")
    time.sleep(0.5)
    
    print("\nQ2: How is this theorem used in real life?")
    r2 = query_session(session_id, "How is this theorem used in real life?")
    print(f"   Decision: {r2.get('session_context_decision')}")
    print(f"   Similarity: {r2.get('max_similarity', 0):.2f}")
    print(f"   Most similar turn: {r2.get('most_similar_turn_index')}")
    
    if r2.get("session_context_decision") == "related":
        print("\n✅ PASS: Follow-up correctly detected as RELATED")
    else:
        print(f"\n⚠️  INFO: Follow-up detected as {r2.get('session_context_decision')}")
    
    # ==========================================================================
    # Scenario 2: New Unrelated Question
    # ==========================================================================
    print_header("Scenario 2: New Unrelated Question")
    
    print("Q3: Explain photosynthesis in plants")
    r3 = query_session(session_id, "Explain photosynthesis in plants")
    print(f"   Decision: {r3.get('session_context_decision')}")
    print(f"   Similarity: {r3.get('max_similarity', 0):.2f}")
    
    # Note: With real embeddings, this would likely be "new_topic"
    # Placeholder responses use same embedding, so may show as related
    print(f"   Retrieval order: {r3.get('data', {}).get('metadata', {}).get('retrieval_order', [])}")
    
    # ==========================================================================
    # Scenario 3: Session Status
    # ==========================================================================
    print_header("Scenario 3: Check Session Status")
    
    status = get_status(session_id)
    print(f"Turn count: {status.get('status', {}).get('turn_count')}")
    print(f"Last decision: {status.get('status', {}).get('last_decision')}")
    print(f"Has topic vector: {status.get('status', {}).get('has_topic_vector')}")
    print("\nRecent turns:")
    for turn in status.get("recent_turns", []):
        print(f"  {turn['turn_number']}: {turn['question']}")
    
    # ==========================================================================
    # Scenario 4: Session Reset
    # ==========================================================================
    print_header("Scenario 4: Session Reset")
    
    print("Resetting session context...")
    reset_resp = reset_session(session_id)
    print(f"Reset status: {reset_resp.get('status')}")
    
    print("\nQ4 (after reset): What is the Pythagoras theorem again?")
    r4 = query_session(session_id, "What is the Pythagoras theorem again?")
    print(f"   Decision: {r4.get('session_context_decision')}")
    print(f"   Similarity: {r4.get('max_similarity', 0):.2f}")
    
    # After reset, should be new_topic (no previous turns to compare)
    if r4.get("session_context_decision") == "new_topic":
        print("\n✅ PASS: Post-reset query correctly treated as NEW_TOPIC")
    else:
        print(f"\n⚠️  INFO: Post-reset detected as {r4.get('session_context_decision')}")
    
    # ==========================================================================
    # Scenario 5: Force Session Prioritize
    # ==========================================================================
    print_header("Scenario 5: Force Session Prioritize")
    
    print("Q5 (with force_session_prioritize=true): Newton's laws")
    r5 = query_session(session_id, "Newton's laws", force_session_prioritize=True)
    retrieval_order = r5.get('data', {}).get('metadata', {}).get('retrieval_order', [])
    print(f"   Retrieval order: {retrieval_order}")
    
    if "session" in retrieval_order:
        print("\n✅ PASS: Force flag correctly includes session in retrieval")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_header("Demo Complete")
    print(f"Session ID: {session_id}")
    print(f"Total queries: 5")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# Curl Examples
# ============================================================================
"""
# Create session
curl -X POST http://localhost:8001/api/session/create \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user"}'

# Query 1 (Pythagoras)
curl -X POST http://localhost:8001/api/session/{session_id}/query \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user", "question": "Explain the Pythagoras theorem"}'

# Query 2 (Follow-up)
curl -X POST http://localhost:8001/api/session/{session_id}/query \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user", "question": "How is this used in real life?"}'

# Check status
curl http://localhost:8001/api/session/{session_id}/status

# Reset context
curl -X POST http://localhost:8001/api/session/{session_id}/reset

# Query with force
curl -X POST http://localhost:8001/api/session/{session_id}/query \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user", "question": "Different topic", "force_session_prioritize": true}'

# Expected log output:
# [SESSION] session_id=abc123... request_id=xyz789 turn_index=2 query="How is this..."
# [EMB] emb_hash=sha256:a1b2c3... emb_dim=384
# [SIM] sims=[0.78] max_sim=0.78 most_similar_turn=1 centroid_sim=none
# [DECISION] decision=related threshold_related=0.65 threshold_forget=0.45 last_decision=new_topic hysteresis_active=false
# [RETR] session_hits=0 classroom_hits=0 web_hits=0
# [MCP] context_order=['session', 'classroom', 'global', 'web']
"""
