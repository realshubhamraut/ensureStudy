#!/usr/bin/env python3
"""
Demo Script: Session Chaining and LaTeX Rendering

Demonstrates the full session-based resource chaining workflow:
1. Create a new session
2. Ask initial question (Q1)
3. Ask related follow-up (Q2) - should detect as related
4. Ask unrelated question (Q3) - should detect as unrelated
5. Export session JSON with resource list

Usage:
    python scripts/demo_session_chaining.py
    
Requirements:
    - AI Tutor service running on localhost:8001
    - Qdrant running (for retrieval, optional)
"""
import os
import sys
import json
import time
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_json(data: dict, title: str = ""):
    """Print formatted JSON"""
    if title:
        print(f"\n{title}:")
    print(json.dumps(data, indent=2))


def create_session(user_id: str, classroom_id: str = None) -> dict:
    """Create a new tutoring session"""
    print_header("Step 1: Create Session")
    
    response = requests.post(
        f"{BASE_URL}/api/session/create",
        json={
            "user_id": user_id,
            "classroom_id": classroom_id
        }
    )
    
    data = response.json()
    if data.get("success"):
        print(f"✅ Session created!")
        print(f"   Session ID: {data['session']['session_id']}")
        print(f"   User ID: {data['session']['user_id']}")
        return data["session"]
    else:
        print(f"❌ Failed: {data.get('error')}")
        return None


def query_session(session_id: str, question: str, find_resources: bool = True) -> dict:
    """Query with session context"""
    print(f"\n📝 Query: \"{question}\"")
    print(f"   find_resources: {find_resources}")
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/session/{session_id}/query",
        json={
            "user_id": "demo_user",
            "question": question,
            "find_resources": find_resources
        }
    )
    elapsed = int((time.time() - start) * 1000)
    
    data = response.json()
    if data.get("success"):
        turn = data.get("turn", {})
        print(f"✅ Response received in {elapsed}ms")
        print(f"   Turn: {turn.get('turn_number')}")
        print(f"   Related: {turn.get('related')} (score: {turn.get('relatedness_score', 0):.2f})")
        print(f"   Resources appended: {data.get('resources_appended', 0)}")
        
        # Show LaTeX blocks if present
        if data.get("data", {}).get("latex_blocks"):
            print(f"   LaTeX blocks: {len(data['data']['latex_blocks'])}")
            for block in data["data"]["latex_blocks"][:2]:  # Show first 2
                print(f"     - {block['id']}: {block['latex'][:50]}...")
        
        return data
    else:
        print(f"❌ Failed: {data.get('error')}")
        return None


def get_session_resources(session_id: str) -> list:
    """Get session resource list"""
    response = requests.get(f"{BASE_URL}/api/session/{session_id}/resources")
    data = response.json()
    return data.get("resources", [])


def export_session(session_id: str) -> dict:
    """Export full session JSON"""
    response = requests.get(f"{BASE_URL}/api/session/{session_id}/export")
    data = response.json()
    return data.get("session", {})


def main():
    """Run the demo"""
    print_header("Session Chaining Demo")
    print(f"API URL: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Step 1: Create session
    session = create_session(
        user_id="demo_user_123",
        classroom_id=None  # No classroom for demo
    )
    
    if not session:
        print("\n❌ Demo failed - could not create session")
        return 1
    
    session_id = session["session_id"]
    
    # Step 2: Initial question
    print_header("Step 2: Initial Question (Q1)")
    q1 = query_session(
        session_id=session_id,
        question="Explain the quadratic formula"
    )
    time.sleep(1)
    
    # Step 3: Related follow-up
    print_header("Step 3: Related Follow-up (Q2)")
    q2 = query_session(
        session_id=session_id,
        question="How do I solve x^2 - 5x + 6 = 0 using that formula?"
    )
    
    if q2 and q2.get("turn", {}).get("related"):
        print("\n✅ PASS: Q2 correctly detected as RELATED to Q1")
    else:
        print("\n⚠️  WARN: Q2 was not detected as related")
    
    time.sleep(1)
    
    # Step 4: Unrelated question
    print_header("Step 4: Unrelated Question (Q3)")
    q3 = query_session(
        session_id=session_id,
        question="What is photosynthesis?"
    )
    
    if q3 and not q3.get("turn", {}).get("related"):
        print("\n✅ PASS: Q3 correctly detected as UNRELATED")
    else:
        print("\n⚠️  WARN: Q3 was incorrectly marked as related")
    
    # Step 5: Export session
    print_header("Step 5: Export Session JSON")
    session_export = export_session(session_id)
    
    if session_export:
        print(f"Session ID: {session_export.get('session_id')}")
        print(f"Query chain: {len(session_export.get('query_chain', []))} turns")
        print(f"Resource list: {len(session_export.get('resource_list', []))} resources")
        
        # Show query chain
        print("\nQuery Chain:")
        for turn in session_export.get("query_chain", []):
            status = "🔗" if turn.get("related") else "🆕"
            print(f"  {status} Turn {turn['turn']}: {turn['question'][:40]}...")
        
        # Show resources
        resources = session_export.get("resource_list", [])
        if resources:
            print("\nResource List:")
            for res in resources[:5]:  # Show first 5
                print(f"  📄 {res.get('title', 'Untitled')} [{res.get('source')}]")
        
        # Save export
        export_file = f"/tmp/session_export_{session_id[:8]}.json"
        with open(export_file, "w") as f:
            json.dump(session_export, f, indent=2)
        print(f"\n💾 Session exported to: {export_file}")
    
    print_header("Demo Complete")
    print("Summary:")
    print(f"  - Created session: {session_id}")
    print(f"  - Executed 3 queries")
    print(f"  - Q1 → Q2: Relatedness detection {'✅' if q2 and q2.get('turn', {}).get('related') else '❌'}")
    print(f"  - Q2 → Q3: Unrelated detection {'✅' if q3 and not q3.get('turn', {}).get('related') else '❌'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
