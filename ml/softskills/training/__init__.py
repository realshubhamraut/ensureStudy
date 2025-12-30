"""
Soft Skills Training Utilities

This package contains utility modules for soft skills analysis,
generated from the notebooks in ml/softskills/notebooks/.

Modules:
- fluency_utils: Speech fluency analysis (WPM, filler detection)
- gaze_utils: Eye contact and gaze direction detection
- gesture_utils: Hand gesture analysis and movement tracking
- posture_utils: Body posture and stability analysis
"""

from . import fluency_utils
from . import gaze_utils
from . import gesture_utils
from . import posture_utils

__all__ = [
    'fluency_utils',
    'gaze_utils',
    'gesture_utils',
    'posture_utils',
]
