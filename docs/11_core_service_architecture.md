# Page 11: Core Service — Flask Architecture & Data Models

---

## 11.1 Overview

The Core Service is the **primary backend API** for ensureStudy, built with Flask and SQLAlchemy. It manages all CRUD operations, user authentication, file uploads, classroom management, and serves as the persistence layer for the entire platform.

### Source: `backend/core-service/`

| Metric | Value |
|--------|-------|
| Framework | Flask 3.x with Application Factory |
| ORM | SQLAlchemy (Flask-SQLAlchemy) |
| Database | PostgreSQL |
| Migrations | Flask-Migrate (Alembic) |
| Auth | JWT (PyJWT) |
| CORS | Flask-CORS (all origins for `api/*`) |
| Max Upload | 500 MB |
| Blueprints | 29 registered |
| Model Files | 20 |

---

## 11.2 Application Factory

### Source: `backend/core-service/app/__init__.py` (125 lines)

```python
def create_app(config_name=None):
    app = Flask(__name__)
    
    # PostgreSQL only — no SQLite fallback
    database_url = os.getenv('DATABASE_URL', 
        'postgresql://ensure_study_user:secure_password_123@localhost:5432/ensure_study')
    
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,     # Verify connections before use
        'pool_recycle': 300,       # Recycle connections every 5 minutes
    }
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
    
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register 29 blueprints...
    # Import all models for table creation...
    
    with app.app_context():
        db.create_all()
    
    return app
```

---

## 11.3 Complete Data Model Reference

### Model File: `user.py` (398 lines — 9 models)

| Model | Table | Key Fields | Purpose |
|-------|-------|------------|---------|
| **User** | `users` | id, username, email, password_hash, role, first_name, last_name, avatar_url | Multi-role user (student, teacher, parent, admin) |
| **Progress** | `progress` | user_id, subject, topic, score, total_questions | Per-topic progress tracking |
| **Assessment** | `assessments` | topic, subject, num_questions, questions (JSON), difficulty, assessment_type | Quiz/assessment definitions |
| **AssessmentResult** | `assessment_results` | user_id, assessment_id, score, total, answers (JSON), feedback (JSON) | Student submission results |
| **ChatSession** | `chat_sessions` | user_id, subject, topic, is_active | AI tutor chat sessions |
| **ModerationLog** | `moderation_logs` | user_id, content, action, confidence, was_blocked, reason | Content moderation audit |
| **Leaderboard** | `leaderboard` | user_id, classroom_id, total_score, streak, level, xp | Gamification leaderboard |
| **StudyNote** | `study_notes` | user_id, title, content, subject, topic, note_type, is_public | AI-generated or user notes |
| **AssessmentChallenge** | `assessment_challenges` | sender_id, recipient_id, assessment_id, status, scores | Peer challenge tracking |

### Model File: `classroom.py` (193 lines — 3 models)

| Model | Table | Key Fields | Purpose |
|-------|-------|------------|---------|
| **Classroom** | `classrooms` | name, grade, section, subject, join_code, teacher_id, organization_id, syllabus_url | Google Classroom-style with 6-char join codes |
| **StudentClassroom** | `student_classrooms` | student_id, classroom_id, joined_at, is_active | Many-to-many join table |
| **ClassroomMaterial** | `classroom_materials` | classroom_id, name, file_url, file_type, source, indexing_status, chunk_count | Uploaded materials with RAG indexing status |

### Model File: `curriculum.py` (996 lines — 16 models)

| Model | Table | Key Fields | Purpose |
|-------|-------|------------|---------|
| **Subject** | `subjects` | name, description, grade_level, classroom_id | Subject definitions |
| **Topic** | `topics` | name, description, subject_id, order, estimated_hours | Topics within subjects |
| **Subtopic** | `subtopics` | name, description, topic_id, difficulty, order | Subtopics within topics |
| **SubtopicAssessment** | `subtopic_assessments` | subtopic_id, questions (JSON), num_questions, difficulty | MCQ assessments per subtopic |
| **StudentSubtopicProgress** | `student_subtopic_progress` | user_id, subtopic_id, score, attempts, mastery_level | Per-subtopic mastery tracking |
| **Syllabus** | `syllabi` | classroom_id, subject_id, file_url, extraction_status, extracted_topics (JSON) | Syllabus documents |
| **QuestionBank** | `question_banks` | classroom_id, subject_id, name, total_questions | Question collections |
| **Question** | `questions` | question_bank_id, text, options (JSON), correct_answer, difficulty, analytics | Individual questions with analytics |
| **Chapter** | `chapters` | classroom_id, name, description, order, color | Chapter/lesson groupings |
| **ClassroomTopic** | `classroom_topics` | chapter_id, classroom_id, name, description, difficulty, total_questions | Shared classroom topics |
| **TopicQuestion** | `topic_questions` | topic_id, classroom_id, question_text, question_type, options (JSON), analytics | MCQ + descriptive questions |
| **StudentTopicScore** | `student_topic_scores` | user_id, topic_id, mcq_score, descriptive_score, mastery_percentage | Cumulative mastery tracking |
| **StudentQuestionResponse** | `student_question_responses` | user_id, question_id, selected_answer, is_correct, time_taken, source | Individual answer records |
| **StudyScheduleEntry** | `study_schedule_entries` | user_id, classroom_topic_id, scheduled_date, duration_minutes, status | Drag-and-drop study calendar |
| **QuestionEffectiveness** | `question_effectiveness` | question_id, times_shown, times_correct, discrimination_index | Type 5 agent quality metrics |
| **LearningAgentMemory** | `learning_agent_memory` | topic_id, classroom_id, memory_data (JSON), generation_count | Persistent agent learning state |

### Other Model Files

| File | Models | Lines | Purpose |
|------|--------|-------|---------|
| `organization.py` | Organization, LicensePurchase | ~130 | Multi-tenant organization management |
| `student_profile.py` | StudentProfile, ParentStudentLink, TeacherClassAssignment | ~200 | Extended profiles, parent-student linking |
| `notes.py` | NoteProcessingJob, DigitizedNotePage, NoteEmbedding, NoteSearchHistory | ~250 | Note digitization pipeline tracking |
| `assignment.py` | Assignment, AssignmentAttachment, Submission, SubmissionFile | ~200 | Teacher assignments and submissions |
| `exam_evaluation.py` | ExamSession, StudentEvaluation | ~180 | Exam evaluation sessions |
| `notification.py` | Notification | ~80 | Push/in-app notifications |
| `meeting.py` | Meeting, MeetingParticipant, MeetingRecording | 241 | Video conferencing with recordings |
| `chat.py` | ChatConversation, ChatMessage, ChatSource | ~200 | Rich chat with source citations |
| `feedback.py` | AgentInteraction, InteractionFeedback, LearningExample, AgentPerformanceMetrics | ~250 | Agent feedback and performance |
| `interact.py` | InteractionSession data models | ~150 | Interactive study sessions |
| `interview_questions.py` | InterviewQuestion, InterviewSession, InterviewResponse | ~200 | Interview preparation tracking |
| `document.py` | Document processing models | ~150 | Document ingestion tracking |
| `document_intelligence.py` | DocumentIntelligence models | ~100 | AI document analysis |
| `announcement.py` | Announcement model | ~80 | Classroom announcements |
| `progress.py` | Additional progress tracking | ~100 | Extended progress models |

---

## 11.4 Entity Relationship Overview

```mermaid
erDiagram
    User ||--o{ StudentClassroom : enrolls
    StudentClassroom }o--|| Classroom : belongs_to
    Classroom }o--|| Organization : part_of
    Classroom ||--o{ Chapter : contains
    Chapter ||--o{ ClassroomTopic : groups
    ClassroomTopic ||--o{ TopicQuestion : has
    ClassroomTopic ||--o{ StudentTopicScore : tracks
    ClassroomTopic ||--o{ StudentQuestionResponse : records
    Classroom ||--o{ ClassroomMaterial : stores
    Classroom ||--o{ Syllabus : references
    Classroom ||--o{ Meeting : hosts
    Meeting ||--o{ MeetingRecording : captures
    Classroom ||--o{ Assignment : assigns
    Assignment ||--o{ Submission : receives
    User ||--o{ Progress : tracks
    User ||--o{ Assessment : takes
    Assessment ||--o{ AssessmentResult : produces
    User ||--o{ ChatSession : starts
    User ||--o{ Leaderboard : ranks
    User ||--o{ StudyNote : creates
    User ||--o{ Notification : receives
    User ||--|| StudentProfile : has
    StudentProfile ||--o{ ParentStudentLink : links
```

---

## 11.5 Database Indexes

Key performance indexes across models:

| Index | Table | Columns | Purpose |
|-------|-------|---------|---------|
| `idx_progress_user_subject` | progress | user_id, subject | Fast progress lookups |
| `idx_result_user_assessment` | assessment_results | user_id, assessment_id | Assessment result queries |
| `idx_challenge_sender` | assessment_challenges | sender_id | Sent challenges lookup |
| `idx_challenge_recipient` | assessment_challenges | recipient_id | Received challenges lookup |
| `idx_chapter_classroom` | chapters | classroom_id | Chapter listing |
| `idx_classroom_topic_chapter` | classroom_topics | chapter_id | Topic hierarchy |
| `idx_classroom_topic_classroom` | classroom_topics | classroom_id | All topics in classroom |
| `idx_response_user` | student_question_responses | user_id | Student answer history |
| `idx_response_question` | student_question_responses | question_id | Question analytics |
| `idx_schedule_user_date` | study_schedule_entries | user_id, scheduled_date | Daily schedule lookup |
| `idx_learning_memory_topic` | learning_agent_memory | topic_id | Agent memory retrieval |
| `unique_student_classroom` | student_classrooms | student_id, classroom_id | Prevents duplicate enrollment |
| `unique_user_subtopic` | student_subtopic_progress | user_id, subtopic_id | One progress per subtopic |
