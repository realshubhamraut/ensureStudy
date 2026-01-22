## API Reference

Complete API documentation for all services including endpoints, request/response formats, and authentication.

### Authentication

All API requests require authentication via JWT tokens unless marked as public.

**Headers**

```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Token Refresh**

```
POST /api/auth/refresh
Authorization: Bearer <refresh_token>

Response:
{
    "access_token": "new_token",
    "expires_in": 900
}
```

### Core Service API

Base URL: `https://api.ensurestudy.com/api`

#### Authentication Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/auth/register` | Create new account | No |
| POST | `/auth/login` | Login and get tokens | No |
| POST | `/auth/refresh` | Refresh access token | Refresh |
| POST | `/auth/logout` | Invalidate tokens | Yes |
| POST | `/auth/forgot-password` | Request password reset | No |
| POST | `/auth/reset-password` | Set new password | No |

**POST /auth/register**

```json
// Request
{
    "email": "student@example.com",
    "password": "securePassword123",
    "name": "John Doe",
    "role": "student"
}

// Response 201
{
    "data": {
        "id": "uuid",
        "email": "student@example.com",
        "name": "John Doe",
        "role": "student",
        "created_at": "2024-01-15T10:30:00Z"
    },
    "message": "Account created successfully"
}
```

**POST /auth/login**

```json
// Request
{
    "email": "student@example.com",
    "password": "securePassword123"
}

// Response 200
{
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "Bearer",
    "expires_in": 900,
    "user": {
        "id": "uuid",
        "email": "student@example.com",
        "name": "John Doe",
        "role": "student"
    }
}
```

#### User Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/users/me` | Get current user profile |
| PUT | `/users/me` | Update profile |
| GET | `/users/me/preferences` | Get preferences |
| PUT | `/users/me/preferences` | Update preferences |

**GET /users/me**

```json
// Response 200
{
    "data": {
        "id": "uuid",
        "email": "student@example.com",
        "name": "John Doe",
        "role": "student",
        "avatar_url": "https://...",
        "created_at": "2024-01-15T10:30:00Z",
        "classrooms_count": 5
    }
}
```

#### Classroom Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/classrooms` | List user's classrooms |
| POST | `/classrooms` | Create classroom (teacher) |
| GET | `/classrooms/{id}` | Get classroom details |
| PUT | `/classrooms/{id}` | Update classroom |
| DELETE | `/classrooms/{id}` | Delete classroom |
| POST | `/classrooms/join` | Join with code |
| GET | `/classrooms/{id}/members` | List members |
| POST | `/classrooms/{id}/materials` | Upload material |
| GET | `/classrooms/{id}/materials` | List materials |

**POST /classrooms**

```json
// Request
{
    "name": "Calculus 101",
    "description": "Introduction to calculus",
    "subject": "Mathematics"
}

// Response 201
{
    "data": {
        "id": "uuid",
        "name": "Calculus 101",
        "description": "Introduction to calculus",
        "subject": "Mathematics",
        "join_code": "ABC12345",
        "teacher": {
            "id": "uuid",
            "name": "Prof. Smith"
        },
        "members_count": 1,
        "created_at": "2024-01-15T10:30:00Z"
    }
}
```

**POST /classrooms/join**

```json
// Request
{
    "join_code": "ABC12345"
}

// Response 200
{
    "data": {
        "classroom_id": "uuid",
        "classroom_name": "Calculus 101",
        "role": "student"
    },
    "message": "Joined classroom successfully"
}
```

#### Meeting Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/classrooms/{id}/meetings` | List meetings |
| POST | `/classrooms/{id}/meetings` | Schedule meeting |
| GET | `/meetings/{id}` | Get meeting details |
| PUT | `/meetings/{id}` | Update meeting |
| DELETE | `/meetings/{id}` | Cancel meeting |
| POST | `/meetings/{id}/start` | Start meeting |
| POST | `/meetings/{id}/end` | End meeting |

**POST /classrooms/{id}/meetings**

```json
// Request
{
    "title": "Weekly Lecture",
    "description": "Chapter 5: Integration",
    "scheduled_at": "2024-01-20T14:00:00Z",
    "duration_minutes": 60
}

// Response 201
{
    "data": {
        "id": "uuid",
        "title": "Weekly Lecture",
        "scheduled_at": "2024-01-20T14:00:00Z",
        "duration_minutes": 60,
        "status": "scheduled",
        "meeting_url": "https://meet.ensurestudy.com/abc123"
    }
}
```

#### Assessment Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/classrooms/{id}/assessments` | List assessments |
| POST | `/classrooms/{id}/assessments` | Create assessment |
| GET | `/assessments/{id}` | Get assessment |
| PUT | `/assessments/{id}` | Update assessment |
| DELETE | `/assessments/{id}` | Delete assessment |
| POST | `/assessments/{id}/submit` | Submit answers |
| GET | `/assessments/{id}/submissions` | List submissions (teacher) |

**POST /classrooms/{id}/assessments**

```json
// Request
{
    "title": "Midterm Exam",
    "type": "exam",
    "total_points": 100,
    "due_date": "2024-01-25T23:59:00Z",
    "time_limit_minutes": 90,
    "is_proctored": true,
    "questions": [
        {
            "question_text": "What is the derivative of x^2?",
            "question_type": "short_answer",
            "points": 10,
            "correct_answer": "2x"
        },
        {
            "question_text": "Which is a valid integral technique?",
            "question_type": "multiple_choice",
            "points": 5,
            "options": ["Guessing", "Substitution", "Hoping", "Wishing"],
            "correct_answer": "Substitution"
        }
    ]
}

// Response 201
{
    "data": {
        "id": "uuid",
        "title": "Midterm Exam",
        "type": "exam",
        "total_points": 100,
        "question_count": 2,
        "created_at": "2024-01-15T10:30:00Z"
    }
}
```

### AI Service API

Base URL: `https://api.ensurestudy.com/api/ai`

#### Tutor Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tutor/chat` | Send message to tutor |
| GET | `/tutor/sessions/{id}` | Get session history |
| DELETE | `/tutor/sessions/{id}` | Clear session |

**POST /tutor/chat**

```json
// Request
{
    "session_id": "uuid",
    "message": "Can you explain integration by parts?",
    "classroom_id": "uuid"
}

// Response 200
{
    "data": {
        "message": "Integration by parts is a technique...",
        "sources": [
            {
                "filename": "calculus_ch5.pdf",
                "page": 42,
                "relevance": 0.89
            }
        ],
        "topic": "Calculus > Integration > Integration by Parts",
        "follow_up_suggestions": [
            "Can you show me an example?",
            "When should I use this technique?"
        ]
    }
}
```

#### Indexing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/indexing/upload` | Upload and index document |
| GET | `/indexing/status/{id}` | Check indexing status |
| DELETE | `/indexing/documents/{id}` | Delete indexed document |

**POST /indexing/upload**

```
// Request (multipart/form-data)
file: <binary>
classroom_id: uuid
document_type: material

// Response 202
{
    "data": {
        "document_id": "uuid",
        "status": "processing",
        "filename": "chapter5.pdf"
    },
    "message": "Document queued for indexing"
}
```

#### Grading Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/grading/grade` | Auto-grade submission |
| POST | `/grading/feedback` | Generate feedback |

**POST /grading/grade**

```json
// Request
{
    "submission_id": "uuid",
    "rubric": "Clear explanation of concept (10pts), Correct formula (10pts), Correct answer (5pts)",
    "submission_text": "The derivative of x^2 is 2x because...",
    "max_score": 25
}

// Response 200
{
    "data": {
        "score": 22,
        "feedback": [
            {
                "criterion": "Clear explanation of concept",
                "score": 9,
                "max": 10,
                "comment": "Good explanation, could include more detail on the power rule"
            },
            {
                "criterion": "Correct formula",
                "score": 10,
                "max": 10,
                "comment": "Correct application of the power rule"
            },
            {
                "criterion": "Correct answer",
                "score": 3,
                "max": 5,
                "comment": "Answer is correct but could simplify notation"
            }
        ],
        "suggestions": [
            "Include the general power rule formula",
            "Show step-by-step work"
        ]
    }
}
```

#### Proctoring Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/proctor/sessions/start` | Start proctoring |
| WS | `/proctor/sessions/{id}/stream` | Frame stream |
| POST | `/proctor/sessions/{id}/end` | End and get report |
| GET | `/proctor/reports/{id}` | Get detailed report |

**POST /proctor/sessions/start**

```json
// Request
{
    "user_id": "uuid",
    "assessment_id": "uuid"
}

// Response 200
{
    "data": {
        "session_id": "uuid",
        "websocket_url": "wss://api.ensurestudy.com/api/ai/proctor/sessions/uuid/stream"
    }
}
```

**POST /proctor/sessions/{id}/end**

```json
// Response 200
{
    "data": {
        "session_id": "uuid",
        "integrity_score": 92.5,
        "risk_level": "low",
        "violation_summary": {
            "gaze_deviation": 3,
            "face_absent": 1
        },
        "duration_minutes": 45
    }
}
```

#### Soft Skills Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/softskills/evaluate/start` | Start evaluation |
| WS | `/softskills/evaluate/{id}/stream` | Real-time stream |
| POST | `/softskills/evaluate/{id}/end` | End and get results |

**POST /softskills/evaluate/{id}/end**

```json
// Request
{
    "full_transcript": "Hello, my name is...",
    "audio_duration": 120.5
}

// Response 200
{
    "data": {
        "session_id": "uuid",
        "overall_score": 78.5,
        "metrics": {
            "fluency": {
                "score": 82,
                "words_per_minute": 135,
                "filler_count": 4
            },
            "grammar": {
                "score": 88,
                "error_count": 2
            },
            "vocabulary": {
                "score": 75,
                "unique_words": 89
            },
            "eye_contact": {
                "score": 72,
                "percentage": 72
            },
            "expression": {
                "score": 68,
                "dominant": "neutral"
            }
        },
        "feedback": {
            "strengths": ["Good speaking pace", "Strong grammar"],
            "improvements": ["Reduce filler words", "More eye contact"],
            "tips": ["Practice pausing instead of saying 'um'"]
        }
    }
}
```

### Error Responses

All endpoints return errors in a consistent format:

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message",
        "details": {}
    },
    "status": "error"
}
```

| HTTP Code | Error Code | Description |
|-----------|------------|-------------|
| 400 | VALIDATION_ERROR | Invalid request data |
| 401 | UNAUTHORIZED | Missing or invalid token |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Resource already exists |
| 429 | RATE_LIMITED | Too many requests |
| 500 | INTERNAL_ERROR | Server error |

### Rate Limits

| Endpoint Group | Limit | Window |
|----------------|-------|--------|
| Authentication | 10 | 1 minute |
| Tutor Chat | 30 | 1 minute |
| File Upload | 5 | 1 minute |
| General API | 100 | 1 minute |

Rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705320000
```

### Pagination

List endpoints support pagination:

```
GET /api/classrooms?page=1&per_page=20
```

Response includes pagination metadata:

```json
{
    "data": [...],
    "pagination": {
        "page": 1,
        "per_page": 20,
        "total": 150,
        "pages": 8,
        "has_next": true,
        "has_prev": false
    }
}
```

### WebSocket Protocols

**Proctoring Stream**

```
Client -> Server: Binary frame (JPEG image)
Server -> Client: JSON message
{
    "frame_number": 1,
    "violations": [],
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**Soft Skills Stream**

```
Client -> Server: JSON
{
    "video_frame": "base64...",
    "audio_chunk": "base64..."  // optional
}

Server -> Client: JSON
{
    "eye_contact": true,
    "expression": "neutral",
    "confidence": 0.85
}
```
