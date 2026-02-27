# Page 63: Teacher Dashboard & Analytics

---

## 63.1 Overview

The teacher dashboard provides **real-time classroom analytics**, student progress monitoring, assessment management, material uploads, and AI-powered insights. It is the primary interface for educators.

---

## 63.2 Dashboard Routes

| Route | Purpose |
|-------|---------|
| `/teacher/dashboard` | Main dashboard with overview widgets |
| `/teacher/classrooms` | Classroom management |
| `/teacher/classrooms/[id]` | Individual classroom view |
| `/teacher/assessments` | Assessment creation and management |
| `/teacher/assessments/[id]` | Assessment details and results |
| `/teacher/students` | Student list and progress |
| `/teacher/analytics` | Detailed analytics and reports |
| `/teacher/materials` | Material management |
| `/teacher/meetings` | Meeting scheduling |

---

## 63.3 Dashboard Widgets

```mermaid
flowchart TB
    subgraph DASH["📊 TEACHER DASHBOARD"]
        direction TB
        subgraph STATS["Key Metrics"]
            direction LR
            S1["Students<br/>42"]
            S2["Avg Score<br/>76.3%"]
            S3["Pending<br/>5 tasks"]
        end

        subgraph CHARTS["Analytics"]
            direction LR
            C1["📈 Class Performance<br/>Score Trend + Completion Rate"]
            C2["⚠️ Attention Needed<br/>Alice: Trig 32%<br/>Bob: Calc 41%<br/>Eve: Alg 45%"]
        end

        subgraph RECENT["Details"]
            direction LR
            R1["📋 Recent Assessments<br/>Physics: 76% avg<br/>Math: 82% avg<br/>Chem: 68% avg"]
            R2["📉 Score Distribution<br/>Histogram"]
        end
    end

    style STATS fill:#3b82f6,color:#fff
    style CHARTS fill:#10b981,color:#fff
    style RECENT fill:#f59e0b,color:#000
```

---

## 63.4 Analytics API

### Class-Level Analytics

```json
GET /api/analytics/classroom/{id}

{
    "total_students": 42,
    "active_students": 38,
    "average_score": 76.3,
    "completion_rate": 0.85,
    "average_streak": 5.2,
    "topics_covered": 18,
    "total_topics": 27,
    "weak_topics": [
        {"topic": "Trigonometry", "students_weak": 8, "avg_score": 42},
        {"topic": "Calculus", "students_weak": 5, "avg_score": 51}
    ],
    "top_performers": [
        {"name": "Alice", "score": 95.2, "streak": 23},
        {"name": "Bob", "score": 91.7, "streak": 15}
    ],
    "score_trend": [
        {"date": "2025-01-01", "average": 68},
        {"date": "2025-02-01", "average": 76}
    ]
}
```

### Student-Level Analytics

```json
GET /api/analytics/student/{id}

{
    "student": {"name": "Alice", "email": "alice@..."},
    "overall_score": 85.2,
    "topics_mastered": 14,
    "topics_total": 27,
    "weak_topics": ["Trigonometry", "Integration"],
    "study_streak": 23,
    "total_study_hours": 45.5,
    "assessment_history": [
        {"title": "Physics Quiz", "score": 92, "date": "2025-02-15"},
        {"title": "Math Test", "score": 78, "date": "2025-02-20"}
    ],
    "progress_over_time": [...]
}
```

---

## 63.5 Assessment Management

### Assessment Creation UI

```mermaid
flowchart TB
    subgraph CREATE["CREATE ASSESSMENT"]
        direction TB
        FORM["Title: Physics Midterm<br/>Subject: Physics<br/>Duration: 60 min<br/>Marks: 100<br/>Proctored: Yes"]

        subgraph Q["Question Sources"]
            direction LR
            MANUAL["Manual<br/>+ MCQ<br/>+ Descriptive<br/>+ True/False"]
            AIGEN["AI Generated<br/>Topic: Thermodynamics<br/>Count: 10, Medium"]
        end

        FORM --> Q
        Q --> SAVE["Save Draft / Publish"]
    end

    style MANUAL fill:#3b82f6,color:#fff
    style AIGEN fill:#8b5cf6,color:#fff
```

---

## 63.6 Results Review

```mermaid
flowchart TB
    subgraph RESULTS["Physics Quiz Results"]
        direction TB
        HEADER["Submissions: 38/42 • Average: 76.3%"]

        subgraph DIST["Score Distribution"]
            D1["90-100: 8 students"]
            D2["80-89: 12 students"]
            D3["70-79: 6 students"]
            D4["60-69: 8 students"]
            D5["<60: 4 students"]
        end

        subgraph QA["Question Analysis"]
            Q1["Q1: 92% correct ✅"]
            Q2["Q2: 78% correct"]
            Q3["Q3: 45% correct ⚠️"]
            Q4["Q4: 88% correct ✅"]
        end
    end

    style DIST fill:#3b82f6,color:#fff
    style QA fill:#f59e0b,color:#000
```

---

## 63.7 Meeting Scheduling

| Feature | Implementation |
|---------|---------------|
| Schedule meeting | Form with date/time picker |
| Notify students | Automatic notification on creation |
| Start meeting | Generate LiveKit room + tokens |
| Record meeting | Optional; triggers transcription |
| Share summary | AI-generated meeting summary + notes |
