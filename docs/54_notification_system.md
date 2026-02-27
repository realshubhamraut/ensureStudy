# Page 54: Notification System — In-App, Email & Real-Time

---

## 54.1 Overview

ensureStudy has a **multi-channel notification system** that delivers in-app notifications, real-time updates, and event-driven alerts. Notifications are triggered by assessments, classrooms, meetings, grading, and system events.

---

## 54.2 Notification Model

### Source: `backend/core-service/app/models/notification.py`

```python
class Notification(db.Model):
    __tablename__ = "notifications"
    
    id          = Column(String(36), primary_key=True, default=uuid4)
    user_id     = Column(String(36), ForeignKey("users.id"), nullable=False)
    title       = Column(String(200), nullable=False)
    message     = Column(Text, nullable=False)
    type        = Column(String(50))     # assessment, classroom, meeting, system
    priority    = Column(String(20), default="normal")  # low, normal, high, urgent
    is_read     = Column(Boolean, default=False)
    action_url  = Column(String(500))    # Deep link URL
    metadata    = Column(JSON)           # Additional data
    created_at  = Column(DateTime, default=datetime.utcnow)
    read_at     = Column(DateTime)
```

---

## 54.3 Notification Types

| Type | Trigger | Priority | Example |
|------|---------|----------|---------|
| `assessment_available` | Teacher creates assessment | High | "New Physics Assessment available" |
| `assessment_graded` | AI grading complete | High | "Your Chemistry quiz has been graded: 85%" |
| `classroom_joined` | Student joins classroom | Normal | "You've joined 'Advanced Math'" |
| `material_uploaded` | Teacher uploads material | Normal | "New study material: Chapter 5.pdf" |
| `meeting_scheduled` | Teacher schedules meeting | High | "Meeting scheduled: Tomorrow 3 PM" |
| `meeting_starting` | Meeting about to start | Urgent | "Live class starting in 5 minutes!" |
| `streak_milestone` | Study streak reached | Normal | "🔥 7-day streak! +100 XP" |
| `weak_topic_alert` | Progress drops below threshold | High | "⚠️ Review needed: Trigonometry" |
| `notes_shared` | Notes shared by classmate | Low | "Alice shared notes on Algebra" |
| `system_announcement` | Platform announcement | Normal | "Scheduled maintenance tonight" |

---

## 54.4 Notification Routes

### Source: `backend/core-service/app/routes/notifications.py`

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/notifications` | List notifications (paginated) |
| GET | `/api/notifications/unread-count` | Get unread count |
| PUT | `/api/notifications/<id>/read` | Mark as read |
| PUT | `/api/notifications/read-all` | Mark all as read |
| DELETE | `/api/notifications/<id>` | Delete notification |

### Response Format

```json
{
    "notifications": [
        {
            "id": "notif_123",
            "title": "Assessment Graded",
            "message": "Your Physics quiz scored 85%. Well done!",
            "type": "assessment_graded",
            "priority": "high",
            "is_read": false,
            "action_url": "/assessments/take/assess_456",
            "created_at": "2025-02-27T14:30:00Z"
        }
    ],
    "unread_count": 3,
    "total": 15,
    "page": 1
}
```

---

## 54.5 Notification Triggers

### Classroom Events

```python
# Core Service: routes/classroom.py
def handle_material_upload(classroom_id, material):
    students = get_classroom_students(classroom_id)
    
    for student in students:
        create_notification(
            user_id=student.id,
            title="New Study Material",
            message=f"New material uploaded: {material.name}",
            type="material_uploaded",
            action_url=f"/classrooms/{classroom_id}",
            metadata={"classroom_id": classroom_id, "material_id": material.id}
        )
```

### Grading Callbacks

```python
# Core Service: routes/grading_callback.py
def handle_grading_complete(assessment_id, user_id, score):
    create_notification(
        user_id=user_id,
        title="Assessment Graded",
        message=f"Your assessment scored {score}%.",
        type="assessment_graded",
        priority="high",
        action_url=f"/assessments/take/{assessment_id}",
        metadata={"assessment_id": assessment_id, "score": score}
    )
```

### Meeting Events

```python
# Core Service: routes/meetings.py
def handle_meeting_created(meeting):
    students = get_classroom_students(meeting.classroom_id)
    
    for student in students:
        create_notification(
            user_id=student.id,
            title="Meeting Scheduled",
            message=f"'{meeting.title}' on {meeting.scheduled_time}",
            type="meeting_scheduled",
            priority="high",
            action_url=f"/meet/{meeting.id}"
        )
```

---

## 54.6 Frontend Notification Components

### NotificationBell

```typescript
// components/NotificationBell.tsx
function NotificationBell() {
    const { unreadCount } = useNotifications();
    
    return (
        <button className="relative">
            <BellIcon />
            {unreadCount > 0 && (
                <span className="badge">{unreadCount}</span>
            )}
        </button>
    );
}
```

### NotificationProvider

```typescript
// components/NotificationProvider.tsx
function NotificationProvider({ children }) {
    // Poll for new notifications every 30 seconds
    useEffect(() => {
        const interval = setInterval(async () => {
            const { unread_count } = await fetchUnreadCount();
            setUnreadCount(unread_count);
        }, 30000);
        
        return () => clearInterval(interval);
    }, []);
    
    return <NotificationContext.Provider value={...}>
        {children}
    </NotificationContext.Provider>;
}
```

---

## 54.7 Notification Pages

| Route | Role | Purpose |
|-------|------|---------|
| `/notifications` | Student | Student notification center |
| `/parent/notifications` | Parent | Parent notification center |
| `/teacher/dashboard` | Teacher | Includes notification section |
| `/admin/dashboard` | Admin | System-wide notifications |
