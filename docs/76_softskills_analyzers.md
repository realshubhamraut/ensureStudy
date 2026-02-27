# Page 76: Soft Skills Analyzers — Implementation Deep Dive

> Supplements Page 17 (Soft Skills Evaluation) with implementation-level detail from `softskills.md`.

---

## 76.1 Analyzer Inventory

| Analyzer | Library | Input | Output |
|----------|---------|-------|--------|
| `FluencyAnalyzer` | Custom | transcript + duration | WPM, filler rate, score |
| `GrammarAnalyzer` | `language_tool_python` | transcript text | error count, corrections |
| `VocabularyAnalyzer` | `nltk` + `wordnet` | transcript text | TTR, advanced words |
| `EyeContactAnalyzer` | `MediaPipe FaceMesh` | video frames | contact %, deviation |
| `ExpressionAnalyzer` | `FER` (MTCNN) | video frames | emotion distribution |

---

## 76.2 FluencyAnalyzer

### Source: `softskills_pipeline.py`

```python
class FluencyAnalyzer:
    def __init__(self):
        self.filler_words = [
            'um', 'uh', 'like', 'you know', 'basically',
            'actually', 'literally', 'so', 'right', 'okay'
        ]
    
    def analyze(self, transcript: str, audio_duration: float) -> dict:
        words = transcript.lower().split()
        word_count = len(words)
        
        # Words per minute
        wpm = (word_count / audio_duration) * 60 if audio_duration > 0 else 0
        
        # Filler word count
        filler_count = sum(
            transcript.lower().count(f) for f in self.filler_words
        )
        filler_rate = filler_count / word_count if word_count > 0 else 0
        
        # Score: optimal WPM is 120-150
        wpm_score = self._score_wpm(wpm)
        filler_score = max(0, 100 - filler_rate * 500)
        
        return {
            'words_per_minute': round(wpm, 1),
            'filler_count': filler_count,
            'filler_rate': round(filler_rate, 3),
            'fluency_score': round((wpm_score + filler_score) / 2, 1),
            'fillers_detected': self._find_fillers(transcript)
        }
    
    def _score_wpm(self, wpm: float) -> float:
        if 120 <= wpm <= 150:
            return 100
        elif wpm < 120:
            return max(0, 100 - (120 - wpm) * 2)
        else:
            return max(0, 100 - (wpm - 150) * 1.5)
```

### Scoring Curve

| WPM Range | Score | Assessment |
|-----------|-------|------------|
| 120-150 | 100 | Optimal pace |
| 100-119 | 60-99 | Slightly slow |
| 151-170 | 70-99 | Slightly fast |
| <100 | 0-59 | Too slow |
| >170 | 0-69 | Too fast |

---

## 76.3 GrammarAnalyzer

```python
import language_tool_python

class GrammarAnalyzer:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')
    
    def analyze(self, text: str) -> dict:
        matches = self.tool.check(text)
        
        errors_by_type = {}
        for match in matches:
            category = match.category
            errors_by_type[category] = errors_by_type.get(category, 0) + 1
        
        word_count = len(text.split())
        error_rate = len(matches) / word_count if word_count > 0 else 0
        score = max(0, 100 - error_rate * 200)
        
        return {
            'error_count': len(matches),
            'errors_by_type': errors_by_type,
            'grammar_score': round(score, 1),
            'corrections': [
                {
                    'original': text[m.offset:m.offset + m.errorLength],
                    'suggestion': m.replacements[0] if m.replacements else None,
                    'message': m.message,
                    'category': m.category
                }
                for m in matches[:10]
            ]
        }
```

---

## 76.4 VocabularyAnalyzer

```python
from collections import Counter
import nltk
from nltk.corpus import wordnet

class VocabularyAnalyzer:
    def __init__(self):
        self.common_words = set(nltk.corpus.words.words()[:3000])
    
    def analyze(self, text: str) -> dict:
        words = nltk.word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0  # Type-token ratio
        
        advanced_words = [
            w for w in unique_words
            if w not in self.common_words and len(w) > 5
        ]
        
        diversity_score = min(100, ttr * 200)
        advanced_score = min(100, len(advanced_words) * 5)
        
        return {
            'total_words': len(words),
            'unique_words': len(unique_words),
            'type_token_ratio': round(ttr, 3),
            'advanced_words': advanced_words[:20],
            'vocabulary_score': round((diversity_score + advanced_score) / 2, 1),
            'top_words': Counter(words).most_common(10)
        }
```

---

## 76.5 EyeContactAnalyzer

```python
class EyeContactAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.total_frames = 0
        self.contact_frames = 0
    
    def process_frame(self, frame: np.ndarray) -> dict:
        self.total_frames += 1
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return {'eye_contact': False, 'reason': 'no_face'}
        
        landmarks = results.multi_face_landmarks[0]
        
        # Iris landmarks (468=left, 473=right)
        left_iris = landmarks.landmark[468]
        right_iris = landmarks.landmark[473]
        
        # Eye corner landmarks for reference
        left_center = (landmarks.landmark[133].x + landmarks.landmark[33].x) / 2
        right_center = (landmarks.landmark[362].x + landmarks.landmark[263].x) / 2
        
        left_deviation = abs(left_iris.x - left_center)
        right_deviation = abs(right_iris.x - right_center)
        avg_deviation = (left_deviation + right_deviation) / 2
        
        is_contact = avg_deviation < 0.25  # Threshold
        if is_contact:
            self.contact_frames += 1
        
        return {
            'eye_contact': is_contact,
            'deviation': round(avg_deviation, 3),
            'contact_rate': round(self.contact_frames / self.total_frames, 3)
        }
```

---

## 76.6 ExpressionAnalyzer

```python
from fer import FER

class ExpressionAnalyzer:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.expression_counts = {}
        self.total_frames = 0
    
    def process_frame(self, frame: np.ndarray) -> dict:
        self.total_frames += 1
        result = self.detector.detect_emotions(frame)
        
        if not result:
            return {'expression': 'no_face', 'confidence': 0}
        
        emotions = result[0]['emotions']
        dominant = max(emotions, key=emotions.get)
        self.expression_counts[dominant] = self.expression_counts.get(dominant, 0) + 1
        
        return {
            'expression': dominant,
            'confidence': round(emotions[dominant], 2),
            'all_emotions': {k: round(v, 2) for k, v in emotions.items()}
        }
    
    def get_summary(self) -> dict:
        positive_rate = sum(
            self.expression_counts.get(e, 0) for e in ['happy', 'neutral']
        ) / self.total_frames if self.total_frames > 0 else 0
        
        return {
            'expression_distribution': {
                k: round(v / self.total_frames * 100, 1) 
                for k, v in self.expression_counts.items()
            },
            'expression_score': round(positive_rate * 100, 1)
        }
```

---

## 76.7 WebSocket Streaming Protocol

```python
@router.websocket("/evaluate/{session_id}/stream")
async def stream_evaluation(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    eye_analyzer = EyeContactAnalyzer()
    expression_analyzer = ExpressionAnalyzer()
    
    try:
        while True:
            data = await websocket.receive_json()
            results = {}
            
            if 'video_frame' in data:
                frame = decode_frame(data['video_frame'])  # Base64 → numpy
                results['eye_contact'] = eye_analyzer.process_frame(frame)
                results['expression'] = expression_analyzer.process_frame(frame)
            
            await websocket.send_json(results)
    except WebSocketDisconnect:
        pass
```

### Client Integration (TypeScript)

```typescript
class SoftSkillsClient {
    private ws: WebSocket;
    private video: HTMLVideoElement;
    private canvas: HTMLCanvasElement;
    
    async start(sessionId: string) {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true, audio: true
        });
        this.video.srcObject = stream;
        
        this.ws = new WebSocket(
            `wss://api.example.com/api/softskills/evaluate/${sessionId}/stream`
        );
        
        // Send frames at 5 FPS
        setInterval(() => this.sendFrame(), 200);
    }
    
    private sendFrame() {
        const ctx = this.canvas.getContext('2d')!;
        ctx.drawImage(this.video, 0, 0);
        const frameData = this.canvas.toDataURL('image/jpeg', 0.7);
        
        this.ws.send(JSON.stringify({
            video_frame: frameData.split(',')[1]
        }));
    }
}
```

---

## 76.8 Combined Scoring Formula

```python
combined_score = (
    fluency['fluency_score']        * 0.25 +  # Speech rate + fillers
    grammar['grammar_score']        * 0.20 +  # LanguageTool errors
    vocabulary['vocabulary_score']  * 0.15 +  # TTR + advanced words
    eye_contact['eye_contact_score'] * 0.15 + # MediaPipe iris tracking
    expression['expression_score']  * 0.10 +  # FER emotion detection
    posture_score                   * 0.10 +  # Body position
    confidence_score                * 0.05    # Composite delivery
)
```

| Score | Level | Interpretation |
|-------|-------|---------------|
| 90-100 | Excellent | Ready for professional settings |
| 75-89 | Good | Minor improvements needed |
| 60-74 | Moderate | Practice recommended |
| 40-59 | Developing | Significant practice needed |
| 0-39 | Beginning | Focus on fundamentals |

### Evaluation Modes

| Mode | Duration | Focus |
|------|----------|-------|
| Interview | 10-30 min | Q&A responses, confidence |
| Presentation | 5-15 min | Structured delivery, engagement |
| Speech | 3-10 min | Fluency, expressiveness |
| Quick Check | 1-3 min | Basic metrics snapshot |
