# Page 39: Frontend Components — UI Building Blocks

---

## 39.1 Overview

The Next.js 14 frontend has **53+ reusable components** across shared components, feature-specific components, and page-level compositions. This page documents the component library, state management, and key UI patterns.

---

## 39.2 Shared Components (11 files)

### Source: `frontend/components/`

| Component | Purpose |
|-----------|---------|
| `Providers.tsx` | Root provider wrapper (NextAuth, Zustand, Theme) |
| `NotificationBell.tsx` | Real-time notification indicator |
| `NotificationProvider.tsx` | Notification context and polling |
| `LatexRenderer.tsx` | Render LaTeX math expressions (KaTeX) |
| `PDFViewer.tsx` | In-browser PDF viewer |
| `PDFViewerWithHighlight.tsx` | PDF viewer with text highlighting |
| `PptxToPdfViewer.tsx` | PowerPoint → PDF conversion viewer |
| `ImageViewer.tsx` | Image viewer with zoom/pan |
| `DocumentSidebar.tsx` | Document navigation sidebar |
| `DocumentContextPanel.tsx` | Document context and metadata panel |
| `SessionDecisionBadge.tsx` | Session intelligence decision indicator |

---

## 39.3 Feature Components by Domain

### Chat & Tutor

| Component | Purpose |
|-----------|---------|
| `ChatInterface` | Main chat UI with message list and input |
| `ChatInput` | Message input with attachments |
| `ChatMessage` | Individual message bubble (supports Markdown, LaTeX) |
| `StreamingResponse` | Real-time SSE rendering |
| `ContextPanel` | RAG context display |
| `SessionSelector` | Chat session picker |

### Classroom

| Component | Purpose |
|-----------|---------|
| `ClassroomCard` | Classroom preview card |
| `ClassroomList` | Grid/list of classrooms |
| `MaterialUploader` | Drag-and-drop file upload |
| `SyllabusViewer` | Syllabus display with topics |
| `JoinClassroomForm` | Join via code form |
| `MemberList` | Classroom members |

### Assessment

| Component | Purpose |
|-----------|---------|
| `AssessmentCard` | Assessment preview |
| `QuestionRenderer` | Render MCQ / descriptive questions |
| `AnswerInput` | Answer input (radio, text, code) |
| `ResultsSummary` | Assessment results dashboard |
| `ProctoringOverlay` | Webcam feed + integrity indicator |

### Progress & Analytics

| Component | Purpose |
|-----------|---------|
| `ProgressChart` | Subject progress (Recharts) |
| `WeakTopicsList` | Topics needing improvement |
| `StudyStreak` | Daily study streak display |
| `LeaderboardTable` | Ranked student table |
| `PerformanceRadar` | Multi-subject radar chart |

### Proctoring

| Component | Purpose |
|-----------|---------|
| `WebcamCapture` | Webcam video feed |
| `IntegrityMeter` | Real-time integrity score |
| `FlagAlert` | Flag notification popup |
| `ProctoringReport` | Post-exam integrity report |

### Soft Skills

| Component | Purpose |
|-----------|---------|
| `VideoRecorder` | Record video for analysis |
| `GazeIndicator` | Eye contact metric display |
| `PostureFeedback` | Real-time posture feedback |
| `FluencyScore` | Speech fluency visualization |
| `GestureOverlay` | Hand gesture detection overlay |

### Meeting

| Component | Purpose |
|-----------|---------|
| `VideoRoom` | LiveKit video conference |
| `ParticipantGrid` | Video grid layout |
| `ChatSidebar` | In-meeting chat |
| `TranscriptView` | Post-meeting transcript |
| `MeetingSummary` | AI-generated summary display |

### Navigation & Layout

| Component | Purpose |
|-----------|---------|
| `Sidebar` | Role-based navigation sidebar |
| `TopNav` | Top navigation bar |
| `BreadcrumbNav` | Breadcrumb navigation |
| `DashboardLayout` | Dashboard page layout |
| `LoadingSpinner` | Loading state indicator |
| `EmptyState` | Empty data placeholder |

---

## 39.4 State Management

### Zustand Stores

```typescript
// Example: Chat store
import { create } from 'zustand'

interface ChatStore {
    sessions: ChatSession[]
    activeSession: string | null
    messages: Message[]
    isStreaming: boolean
    
    setActiveSession: (id: string) => void
    addMessage: (msg: Message) => void
    appendToLastMessage: (chunk: string) => void
    clearMessages: () => void
}

const useChatStore = create<ChatStore>((set) => ({
    sessions: [],
    activeSession: null,
    messages: [],
    isStreaming: false,
    
    setActiveSession: (id) => set({ activeSession: id }),
    addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
    appendToLastMessage: (chunk) => set((s) => ({
        messages: s.messages.map((m, i) => 
            i === s.messages.length - 1 
                ? { ...m, content: m.content + chunk } 
                : m
        )
    })),
}))
```

### Key Stores

| Store | State Managed |
|-------|--------------|
| `useChatStore` | Active session, messages, streaming state |
| `useClassroomStore` | Current classroom, materials, members |
| `useAuthStore` | User profile, role, JWT token |
| `useProgressStore` | Progress data, weak topics |
| `useProctoringStore` | Webcam state, integrity score, flags |

---

## 39.5 SSE Streaming Implementation

```typescript
async function streamChat(message: string, sessionId: string) {
    const response = await fetch('/api/tutor/chat', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({ message, session_id: sessionId })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                
                const parsed = JSON.parse(data);
                useChatStore.getState().appendToLastMessage(parsed.content);
            }
        }
    }
}
```

---

## 39.6 Three.js 3D Elements

```typescript
// Landing page 3D visualization
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Sphere } from '@react-three/drei'

function Hero3D() {
    return (
        <Canvas>
            <ambientLight />
            <pointLight position={[10, 10, 10]} />
            <AnimatedSphere />
            <OrbitControls enableZoom={false} />
        </Canvas>
    )
}
```

Used on the landing page for premium visual differentiation with animated 3D elements.
