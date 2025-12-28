"""
ensureStudy Proctoring Module

Enforces exam integrity during assessments by detecting:
- Head pose anomalies
- Face absence
- Gaze diversion
- Multiple-person presence
- Prohibited objects
- Hand presence anomalies

Produces an Integrity Score (0-100) for each session.
"""

from .api import router

__all__ = ["router"]
