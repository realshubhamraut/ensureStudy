# Soft Skills Communication Module - Database & Dataset Requirements

## Overview

Real-time evaluation of communication skills for the `/softskills/communication` module.

**Skills to Evaluate:**
| Skill | Description | Weight |
|-------|-------------|--------|
| **Fluency** | Speech rate, pauses, and flow | 30% |
| **Grammar** | Sentence structure and correctness | 20% |
| **Eye Contact** | Looking at camera/interviewer | 15% |
| **Hand Gestures** | Natural and stable movements | 10% |
| **Posture** | Body stability and presence | 10% |
| **Expressions** | Facial engagement | 10% |

---

## 1. Existing Project Assets

### Available Data Files

| File | Type | Purpose |
|------|------|---------|
| `new2_csv/train/Train_Video1_processed.csv` | CSV | Proctoring data (gaze, object detection) |
| `new2_csv/train/Train_Video2_processed.csv` | CSV | Proctoring data |
| `new2_csv/test/Test_Video1_processed.csv` | CSV | Proctoring test data |
| `new2_csv/test/Test_Video2_processed.csv` | CSV | Proctoring test data |

**Note:** The existing CSV data is for **cheating/proctoring detection**, NOT soft skills. It contains:
- `iris_pos`, `iris_ratio`, `gaze_direction`, `gaze_zone` (eye tracking)
- `x_rotation`, `y_rotation`, `z_rotation` (head pose)
- `watch`, `headphone`, `cell phone`, `earpiece` (object detection)
- `is_cheating` (label)

### Existing Models

| Path | Description |
|------|-------------|
| `ml/models/engagement_model.pth` | PyTorch engagement model |
| `ml/models/proctoring/` | 6 proctoring-related models |

### Existing Backend Code

| File | Purpose |
|------|---------|
| `backend/ai-service/app/api/routes/softskills.py` | API endpoints (partially implemented) |
| `frontend/app/(dashboard)/softskills/communication/session/page.tsx` | Frontend UI (simulated scores) |

---

## 2. Database Schema Requirements

### PostgreSQL Tables (New)

```sql
-- Soft Skills Sessions
CREATE TABLE IF NOT EXISTS softskills_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module_type VARCHAR(30) NOT NULL,  -- 'communication', 'mock_interview'
    avatar_id VARCHAR(20),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    prompt_count INTEGER DEFAULT 0,
    is_complete BOOLEAN DEFAULT FALSE
);

-- Session Responses (per prompt)
CREATE TABLE IF NOT EXISTS softskills_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES softskills_sessions(id) ON DELETE CASCADE,
    prompt_index INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    transcript TEXT,
    audio_duration_seconds FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, prompt_index)
);

-- Fluency Metrics
CREATE TABLE IF NOT EXISTS fluency_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    words_per_minute FLOAT,
    pause_count INTEGER DEFAULT 0,
    pause_ratio FLOAT,  -- pause_time / total_time
    filler_word_count INTEGER DEFAULT 0,
    filler_words_detected TEXT[],  -- Array of detected fillers
    score FLOAT,  -- 0-100
    created_at TIMESTAMP DEFAULT NOW()
);

-- Grammar Metrics
CREATE TABLE IF NOT EXISTS grammar_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    error_count INTEGER DEFAULT 0,
    sentence_count INTEGER DEFAULT 0,
    avg_sentence_length FLOAT,
    vocabulary_richness FLOAT,  -- unique_words / total_words
    score FLOAT,  -- 0-100
    created_at TIMESTAMP DEFAULT NOW()
);

-- Visual Metrics (aggregated per response)
CREATE TABLE IF NOT EXISTS visual_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    frames_analyzed INTEGER DEFAULT 0,
    
    -- Eye Contact
    eye_contact_score FLOAT,
    gaze_center_ratio FLOAT,  -- % frames looking at camera
    gaze_left_ratio FLOAT,
    gaze_right_ratio FLOAT,
    
    -- Head Pose
    head_forward_ratio FLOAT,
    avg_head_pitch FLOAT,
    avg_head_yaw FLOAT,
    head_stability_score FLOAT,  -- lower movement = higher stability
    
    -- Hands
    hand_gesture_score FLOAT,
    hands_visible_ratio FLOAT,
    hand_movement_frequency FLOAT,  -- movements per minute
    
    -- Posture
    posture_score FLOAT,
    upper_body_stability FLOAT,
    shoulder_alignment_score FLOAT,
    
    -- Expressions
    expression_score FLOAT,
    smile_ratio FLOAT,
    neutral_ratio FLOAT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Frame-level Analysis (for detailed debugging/replay)
CREATE TABLE IF NOT EXISTS frame_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER,
    
    -- Detections
    face_detected BOOLEAN DEFAULT FALSE,
    num_faces INTEGER DEFAULT 0,
    gaze_direction VARCHAR(20),  -- 'center', 'left', 'right', 'up', 'down'
    head_pose_pitch FLOAT,
    head_pose_yaw FLOAT,
    head_pose_roll FLOAT,
    
    hands_visible BOOLEAN DEFAULT FALSE,
    num_hands INTEGER DEFAULT 0,
    hand_positions JSONB,  -- [{x, y, z}, ...]
    
    shoulder_keypoints JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_frame_analysis_response ON frame_analysis(response_id);
CREATE INDEX idx_frame_analysis_timestamp ON frame_analysis(response_id, timestamp_ms);

-- Final Scores
CREATE TABLE IF NOT EXISTS softskills_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES softskills_sessions(id) ON DELETE CASCADE,
    
    -- Individual Scores (0-100)
    fluency_score FLOAT,
    grammar_score FLOAT,
    eye_contact_score FLOAT,
    hand_gesture_score FLOAT,
    posture_score FLOAT,
    expression_score FLOAT,
    
    -- Weighted Overall
    overall_score FLOAT,
    
    -- Feedback
    strengths TEXT[],
    areas_for_improvement TEXT[],
    detailed_feedback TEXT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_softskills_scores_session ON softskills_scores(session_id);
CREATE INDEX idx_softskills_sessions_user ON softskills_sessions(user_id);
```

### Redis Keys (Caching)

```
# Real-time frame analysis (expires after session)
softskills:frame:{session_id}:{frame_num} -> JSON frame data
TTL: 1 hour

# Running aggregates
softskills:agg:{session_id} -> JSON running scores
TTL: 24 hours

# Session state
softskills:state:{session_id} -> JSON current state
TTL: 2 hours
```

---

## 3. External Datasets

### Speech Fluency Datasets

| Dataset | Source | Description | Use Case |
|---------|--------|-------------|----------|
| **DisfluencySpeech** | [HuggingFace](https://huggingface.co/datasets/disfluency-speech) | 10h labeled disfluency audio | Filler words, pauses |
| **PodcastFillers** | [Adobe Research](https://github.com/adobe-research/podcast-fillers) | 35K annotated filler words | Filler detection training |
| **SEP-28k** | [Kaggle](https://www.kaggle.com/datasets/sep-28k) | Stuttering events in podcasts | Pause analysis |
| **FluencyBank** | [ASHA](https://www.asha.org) | Word timings, disfluencies | WPM calculation |
| **Avalinguo** | [GitHub](https://github.com/agrija9/Avalinguo-Dataset) | Fluency level classification | Fluency scoring |
| **Kaggle Speech Fluency** | [Kaggle](https://www.kaggle.com/datasets/speech-fluency-pronunciation) | Content, fluency, pronunciation scores | Multi-metric training |

### Eye Contact / Gaze Tracking

| Dataset | Source | Description | Use Case |
|---------|--------|-------------|----------|
| **Gaze Capture** | MIT CSAIL | 1474 subjects, mobile data | Gaze estimation |
| **EyeT4Empathy** | [Kaggle](https://www.kaggle.com/datasets/eyet4empathy) | Eye-tracking + empathy | Engagement analysis |
| **VEDB** | NIH | 240h egocentric video + gaze | Video gaze tracking |
| **MPIIGaze** | MPI | Face images with gaze labels | Gaze direction model |
| **GazeCapture** | [MIT](http://gazecapture.csail.mit.edu/) | Large-scale mobile gaze | Cross-device gaze |

### Hand Gesture Recognition

| Dataset | Source | Description | Use Case |
|---------|--------|-------------|----------|
| **IPN Hand** | [GitHub](https://gibranbenitez.github.io/IPN_Hand) | 800K frames, 50 subjects | Dynamic gestures |
| **HaGRID** | [Kaggle/GitHub](https://github.com/hukenovs/hagrid) | 552K images, 18 gestures | Static gestures |
| **NVIDIA DHG** | NVIDIA | 20K clips, diverse lighting | Real-world conditions |
| **DHG-14/28** | [CV Foundation](https://www-sop.inria.fr/reves/Basilic/ResearchTopics/Gestures/DHG-dataset.html) | Skeleton + depth | Hand pose |

### Posture / Body Language

| Dataset | Source | Description | Use Case |
|---------|--------|-------------|----------|
| **MPII Human Pose** | [MPI](http://human-pose.mpi-inf.mpg.de/) | 25K images, 15 body joints | Pose estimation |
| **COCO Keypoints** | [COCO](https://cocodataset.org/) | 250K keypoint annotations | Multi-person pose |
| **Human3.6M** | [CVLab](http://vision.imar.ro/human3.6m/) | 3.6M 3D poses | 3D pose estimation |
| **BoLD Challenge** | [PSU](https://github.com/bohaohuang/BoLD) | Body language + emotion | Emotion from pose |
| **Kaggle Postures** | [Kaggle](https://www.kaggle.com/datasets/postures) | 4.8K silhouettes | Basic posture classification |

### Interview-Specific Datasets

| Dataset | Source | Description | Use Case |
|---------|--------|-------------|----------|
| **MIT Interview Dataset** | [MIT Media Lab](https://www.media.mit.edu/) | Video interviews with labels | End-to-end interview scoring |
| **Interview Candidate Dataset** | [Kaggle](https://www.kaggle.com/datasets/interview-candidate) | Confident/not confident images | Confidence detection |
| **HiRe Dataset** | [GitHub](https://github.com/PALASH-BAJPAI/HiRe_Automated_Interviewing_tool) | HR interview analysis | Lexical/prosodic features |
| **Multimodal Behavioral Analytics** | [GitHub](https://github.com/sunitharavi9/Multimodal-Behavioral-Analytics) | Engagement, speaking rate, eye contact | Multi-modal fusion |

---

## 4. Technology Stack

### Backend Requirements

```python
# requirements.txt additions
mediapipe>=0.10.0           # Face mesh, hands, pose
opencv-python>=4.8.0        # Video processing
dlib>=19.24.0               # Face detection (existing)
language_tool_python>=2.7   # Grammar checking
nltk>=3.8                   # Text processing
librosa>=0.10               # Audio analysis (pauses)
webrtcvad>=2.0.10           # Voice activity detection
torch>=2.0                  # ML models
transformers>=4.30          # NLP models
```

### Frontend Requirements

```bash
npm install @mediapipe/face_mesh @mediapipe/hands @mediapipe/pose
```

### Real-time Architecture

```
┌─────────────┐    WebSocket    ┌─────────────────┐
│   Browser   │ ──────────────► │  FastAPI WS     │
│  (MediaPipe │                 │  Endpoint       │
│   client)   │                 └────────┬────────┘
└─────────────┘                          │
                                         ▼
                               ┌─────────────────┐
                               │  Frame Analyzer │
                               │  (MediaPipe+CV) │
                               └────────┬────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
              ┌────────────┐    ┌────────────┐    ┌────────────┐
              │ Gaze Score │    │ Hand Score │    │ Pose Score │
              └────────────┘    └────────────┘    └────────────┘
                     │                  │                  │
                     └──────────────────┼──────────────────┘
                                        ▼
                               ┌─────────────────┐
                               │  Redis (agg)    │
                               └─────────────────┘
```

---

## 5. Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create PostgreSQL tables (migration script)
- [ ] Set up WebSocket endpoint for real-time frames
- [ ] Integrate MediaPipe Face Mesh + Hands + Pose

### Phase 2: Audio/Speech Analysis (Week 2)
- [ ] Implement fluency analyzer (WPM, pauses, fillers)
- [ ] Integrate grammar checker (LanguageTool)
- [ ] Download/prepare DisfluencySpeech dataset

### Phase 3: Visual Analysis (Week 3)
- [ ] Eye contact detection from gaze direction
- [ ] Hand gesture scoring (movement frequency, visibility)
- [ ] Posture stability from shoulder keypoints

### Phase 4: Model Training (Week 4)
- [ ] Train fluency classifier on DisfluencySpeech
- [ ] Fine-tune pose model for interview context
- [ ] Create scoring calibration dataset

---

## 6. Environment Variables

```env
# Soft Skills Feature Flags
FEATURE_SOFTSKILLS=true
SOFTSKILLS_FRAME_RATE=5              # Frames per second to analyze
SOFTSKILLS_MIN_FACE_CONFIDENCE=0.7
SOFTSKILLS_GAZE_CENTER_THRESHOLD=0.3  # Iris deviation threshold

# Fluency Thresholds
FLUENCY_OPTIMAL_WPM_MIN=120
FLUENCY_OPTIMAL_WPM_MAX=160
FLUENCY_FILLER_PENALTY=0.05          # Per filler word

# Grammar
GRAMMAR_SPACY_MODEL=en_core_web_sm
GRAMMAR_LANGUAGETOOL_URL=http://localhost:8081

# Scoring Weights
SCORE_WEIGHT_FLUENCY=0.30
SCORE_WEIGHT_GRAMMAR=0.20
SCORE_WEIGHT_EYE_CONTACT=0.15
SCORE_WEIGHT_GESTURES=0.10
SCORE_WEIGHT_POSTURE=0.10
SCORE_WEIGHT_EXPRESSIONS=0.10
```

---

## 7. Quick Dataset Download Commands

```bash
# DisfluencySpeech (HuggingFace)
huggingface-cli download disfluency-speech --local-dir ./datasets/disfluency-speech

# PodcastFillers (Adobe)
git clone https://github.com/adobe-research/podcast-fillers.git ./datasets/podcast-fillers

# HaGRID Hand Gestures
kaggle datasets download -d hagrid-hand-gestures -p ./datasets/hagrid

# IPN Hand
git clone https://github.com/GibranBenitez/IPN_Hand.git ./datasets/ipn-hand

# Interview Candidate (Kaggle)
kaggle datasets download -d interview-candidate-dataset -p ./datasets/interview-candidate
```

---

## Summary

| Category | Status | Action Required |
|----------|--------|-----------------|
| **Existing Data** | ⚠️ Not suitable | Proctoring data, not soft skills |
| **Database Schema** | ❌ Missing | Create migration for 8 new tables |
| **External Datasets** | ❌ Not downloaded | Download 5-10 key datasets |
| **Backend Code** | ⚠️ Partial | Add real CV analysis (currently simulated) |
| **Frontend** | ⚠️ Partial | Connect to real WebSocket analysis |
