# Page 32: Core Service API — Complete Endpoint Reference

---

## 32.1 Overview

The Core Service (Flask) exposes **29 blueprint modules** with an estimated **120+ REST endpoints**. This page provides a complete endpoint reference organized by blueprint.

### Base URL: `http://localhost:8000`

---

## 32.2 Endpoint Reference by Blueprint

### Authentication (`routes/auth.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/auth/register` | — | Create new user account |
| POST | `/api/auth/login` | — | Authenticate and receive JWT |
| POST | `/api/auth/refresh` | Token | Refresh expired JWT |
| GET | `/api/auth/me` | Token | Get current user profile |
| PUT | `/api/auth/me` | Token | Update user profile |
| POST | `/api/auth/change-password` | Token | Change password |
| POST | `/api/auth/forgot-password` | — | Initiate password reset |

### Users (`routes/users.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/users` | Admin | List all users |
| GET | `/api/users/<id>` | Token | Get user by ID |
| PUT | `/api/users/<id>` | Token | Update user |
| DELETE | `/api/users/<id>` | Admin | Delete user |

### Classrooms (`routes/classroom.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/classrooms` | Teacher | Create classroom |
| GET | `/api/classrooms` | Token | List user's classrooms |
| GET | `/api/classrooms/<id>` | Token | Get classroom details |
| PUT | `/api/classrooms/<id>` | Teacher | Update classroom |
| DELETE | `/api/classrooms/<id>` | Teacher | Delete classroom |
| POST | `/api/classrooms/<id>/join` | Student | Join via code |
| GET | `/api/classrooms/<id>/students` | Teacher | List students |
| POST | `/api/classrooms/<id>/materials` | Teacher | Upload material |
| GET | `/api/classrooms/<id>/materials` | Token | List materials |
| POST | `/api/classrooms/<id>/syllabus` | Teacher | Upload syllabus |

### Assessments (`routes/assessments.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/assessments` | Teacher | Create assessment |
| GET | `/api/assessments` | Token | List assessments |
| GET | `/api/assessments/<id>` | Token | Get assessment structure |
| POST | `/api/assessments/<id>/submit` | Student | Submit answers |
| GET | `/api/assessments/<id>/results` | Token | Get results |
| GET | `/api/assessments/<id>/results/<user_id>` | Teacher | Get student result |

### Chat (`routes/chat.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/chat/sessions` | Token | Create chat session |
| GET | `/api/chat/sessions` | Token | List sessions |
| GET | `/api/chat/sessions/<id>` | Token | Get session messages |
| POST | `/api/chat/sessions/<id>/messages` | Token | Send message |
| DELETE | `/api/chat/sessions/<id>` | Token | Delete session |

### Curriculum (`routes/curriculum.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/curriculum/subjects` | Token | List subjects |
| GET | `/api/curriculum/topics/<subject_id>` | Token | List topics for subject |
| POST | `/api/curriculum/topics` | Teacher | Create topic |
| PUT | `/api/curriculum/topics/<id>` | Teacher | Update topic |
| GET | `/api/curriculum/questions/<topic_id>` | Token | Get questions |

### Progress (`routes/progress.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/progress` | Token | Get user progress |
| GET | `/api/progress/subject/<subject>` | Token | Progress by subject |
| PUT | `/api/progress/<id>` | Token | Update progress |
| GET | `/api/progress/weak-topics` | Token | Get weak topics |
| GET | `/api/progress/analytics` | Token | Analytics data |

### Leaderboard (`routes/leaderboard.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/leaderboard` | Token | Global leaderboard |
| GET | `/api/leaderboard/classroom/<id>` | Token | Classroom leaderboard |
| GET | `/api/leaderboard/me` | Token | User's rank |

### Meetings (`routes/meetings.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/meetings` | Teacher | Create meeting |
| GET | `/api/meetings/<id>` | Token | Get meeting details |
| POST | `/api/meetings/<id>/start` | Teacher | Start meeting |
| POST | `/api/meetings/<id>/end` | Teacher | End meeting |
| POST | `/api/meetings/<id>/join` | Token | Join meeting |

### Recordings (`routes/recordings.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/recordings/<meeting_id>` | Token | List recordings |
| POST | `/api/recordings` | Teacher | Save recording |
| GET | `/api/recordings/<id>/stream` | Token | Stream recording |

### Notes (`routes/notes.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/notes` | Token | Create note |
| GET | `/api/notes` | Token | List user notes |
| PUT | `/api/notes/<id>` | Token | Update note |
| DELETE | `/api/notes/<id>` | Token | Delete note |

### Documents (`routes/documents.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/documents/upload` | Token | Upload document |
| GET | `/api/documents/<id>` | Token | Get document metadata |
| POST | `/api/documents/<id>/index` | Token | Trigger indexing |

### Files (`routes/files.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/files/upload` | Token | General file upload |
| GET | `/api/files/<id>` | Token | Download file |

### Notifications (`routes/notifications.py`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/notifications` | Token | List notifications |
| PUT | `/api/notifications/<id>/read` | Token | Mark as read |
| DELETE | `/api/notifications/<id>` | Token | Delete notification |

### Additional Blueprints

| Blueprint | Key Endpoints |
|-----------|---------------|
| `admin.py` | Platform administration, user management |
| `feedback.py` | Submit/retrieve feedback on AI interactions |
| `grading_callback.py` | Callback endpoint for async grading results |
| `interact.py` | Interactive learning session management |
| `interview_questions.py` | Interview question CRUD |
| `revision.py` | Spaced revision scheduling |
| `teacher_assistant.py` | AI-powered teacher assistant |
| `teacher.py` | Teacher-specific operations |
| `students.py` | Student management |
| `topics.py` | Topic CRUD operations |
| `web_resources.py` | Web resource bookmarking |
| `evaluation.py` | Answer evaluation callbacks |
| `assignment.py` | Assignment management |
| `question_progress.py` | Per-question progress tracking |

---

## 32.3 Health Endpoint

```python
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'core-api'})
```
