# Page 79: Proctoring System — Implementation Deep Dive

> Supplements Page 14 (Proctoring Engine) with full `StaticProctor` class, gaze estimation math, head pose PnP, `IntegrityScorer`, browser event monitoring, and TypeScript client integration from `proctoring.md`.

---

## 79.1 StaticProctor Class

### Source: `backend/ai-service/app/proctor/`

```python
class StaticProctor:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')
        self.yolo_classes = self.yolo.names
        
        self.face_landmarker = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.violations = []
        self.frame_count = 0
        self.face_absent_frames = 0
    
    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_count += 1
        results = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'detections': [],
            'violations': []
        }
        
        # YOLO: multiple people + mobile phone
        yolo_results = self.yolo(frame, verbose=False)[0]
        
        people_count = sum(
            1 for box in yolo_results.boxes
            if self.yolo_classes[int(box.cls)] == 'person'
        )
        if people_count > 1:
            results['violations'].append({
                'type': 'multiple_faces', 'count': people_count
            })
        
        for box in yolo_results.boxes:
            if self.yolo_classes[int(box.cls)] == 'cell phone' and box.conf > 0.5:
                results['violations'].append({
                    'type': 'mobile_phone',
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        # MediaPipe: face presence + gaze
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_landmarker.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            self.face_absent_frames += 1
            if self.face_absent_frames > 90:  # ~3s at 30fps
                results['violations'].append({
                    'type': 'face_absent',
                    'duration_frames': self.face_absent_frames
                })
        else:
            self.face_absent_frames = 0
            gaze = self._calculate_gaze(face_results.multi_face_landmarks[0])
            if abs(gaze['horizontal']) > 30 or abs(gaze['vertical']) > 20:
                results['violations'].append({
                    'type': 'gaze_deviation',
                    'horizontal': gaze['horizontal'],
                    'vertical': gaze['vertical']
                })
        
        return results
```

---

## 79.2 Gaze Estimation Algorithm

Uses iris landmarks (468/473) relative to eye corner landmarks to compute angular deviation:

```python
def _calculate_gaze(self, landmarks) -> dict:
    # Eye corner landmark indices
    left_eye = [landmarks.landmark[i] for i in [33, 133, 160, 144, 145, 153]]
    right_eye = [landmarks.landmark[i] for i in [362, 263, 387, 373, 380, 374]]
    
    # Iris landmarks (refined)
    left_iris = landmarks.landmark[468]
    right_iris = landmarks.landmark[473]
    
    # Compute center of each eye
    left_center = np.mean([[p.x, p.y] for p in left_eye], axis=0)
    right_center = np.mean([[p.x, p.y] for p in right_eye], axis=0)
    
    # Deviation from center, normalized
    left_deviation = (left_iris.x - left_center[0]) / 0.02
    right_deviation = (right_iris.x - right_center[0]) / 0.02
    horizontal = (left_deviation + right_deviation) / 2 * 45  # degrees
    
    left_vert = (left_iris.y - left_center[1]) / 0.015
    right_vert = (right_iris.y - right_center[1]) / 0.015
    vertical = (left_vert + right_vert) / 2 * 30  # degrees
    
    return {'horizontal': horizontal, 'vertical': vertical}
```

### Thresholds

| Axis | Normal Range | Violation Threshold |
|------|-------------|---------------------|
| Horizontal | ±30° | >30° off-center |
| Vertical | ±20° | >20° up/down |

---

## 79.3 Head Pose via PnP

Uses `cv2.solvePnP` with 6 canonical 3D face model points:

```python
def _calculate_head_pose(self, landmarks, frame_shape) -> dict:
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)
    
    h, w = frame_shape[:2]
    indices = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
        for i in indices
    ], dtype=np.float64)
    
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    _, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, None
    )
    
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles = cv2.decomposeProjectionMatrix(
        np.hstack((rotation_mat, translation_vec.reshape(3, 1)))
    )[6]
    
    return {
        'yaw': angles[1][0],    # Left-right
        'pitch': angles[0][0],  # Up-down
        'roll': angles[2][0]    # Tilt
    }
```

---

## 79.4 IntegrityScorer

```python
class IntegrityScorer:
    weights = {
        'face_absent': 0.3,
        'multiple_faces': 0.4,
        'gaze_deviation': 0.1,
        'mobile_phone': 0.5,
        'head_rotation': 0.15,
        'tab_switch': 0.2
    }
    
    def calculate_score(self, session_violations: list) -> dict:
        violation_counts = {}
        for v in session_violations:
            v_type = v['type']
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
        
        # Logarithmic diminishing returns for repeated violations
        penalty = sum(
            self.weights.get(v_type, 0.1) * np.log1p(count)
            for v_type, count in violation_counts.items()
        )
        
        raw_score = max(0, 100 - penalty * 10)
        
        return {
            'score': round(raw_score, 1),
            'violation_summary': violation_counts,
            'risk_level': (
                'low' if raw_score >= 90 else
                'medium' if raw_score >= 70 else
                'high' if raw_score >= 50 else
                'critical'
            )
        }
```

---

## 79.5 Browser Event Monitoring

Client-side JavaScript monitors tab switching, clipboard, and context menu:

```typescript
// Tab visibility
document.addEventListener('visibilitychange', () => {
    if (document.hidden) sendViolation({ type: 'tab_switch' });
});

// Window blur
window.addEventListener('blur', () => {
    sendViolation({ type: 'window_blur' });
});

// Clipboard block
document.addEventListener('copy', (e) => {
    e.preventDefault();
    sendViolation({ type: 'copy_attempt' });
});

// Right-click block
document.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    sendViolation({ type: 'context_menu' });
});
```

---

## 79.6 ProctoringClient (TypeScript)

```typescript
class ProctoringClient {
    private websocket: WebSocket | null = null;
    private video: HTMLVideoElement;
    private canvas: HTMLCanvasElement;
    
    async start(sessionId: string) {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        this.video.srcObject = stream;
        
        this.websocket = new WebSocket(
            `wss://api.example.com/api/proctor/sessions/${sessionId}/stream`
        );
        
        this.websocket.onmessage = (event) => {
            const results = JSON.parse(event.data);
            this.handleResults(results);
        };
        
        // 10 FPS capture
        setInterval(() => this.captureAndSend(), 100);
    }
    
    private captureAndSend() {
        const ctx = this.canvas.getContext('2d')!;
        ctx.drawImage(this.video, 0, 0, 640, 480);
        
        this.canvas.toBlob((blob) => {
            if (blob && this.websocket?.readyState === WebSocket.OPEN) {
                this.websocket.send(blob);  // Binary JPEG
            }
        }, 'image/jpeg', 0.8);
    }
}
```

### WebSocket Protocol

| Direction | Format | Content |
|-----------|--------|---------|
| Client → Server | Binary (JPEG blob) | Compressed frame at 0.8 quality |
| Server → Client | JSON | `{ frame_number, violations[], timestamp }` |

---

## 79.7 Session Lifecycle

```
POST /proctor/sessions/start
    → Redis: proctor:session:{id} (TTL 2h)
    → Return session_id + WebSocket URL

WS /proctor/sessions/{id}/stream
    → Frame loop: capture → decode → process → respond
    → Violations appended to Redis session

POST /proctor/sessions/{id}/end
    → IntegrityScorer.calculate_score()
    → Store report to PostgreSQL/MongoDB
    → Delete Redis session
    → Return full report
```

### Report Schema

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | UUID | Session identifier |
| `integrity_score` | float | 0-100 score |
| `risk_level` | string | low/medium/high/critical |
| `violation_summary` | object | Count per violation type |
| `detailed_violations` | array | Full violation records with timestamps |
| `frame_snapshots` | array | Saved evidence frames |
