# Page 66: Zustand State Management — 5 Stores

---

## 66.1 Overview

The Next.js frontend uses **Zustand** for client-side state management instead of Redux. Zustand provides lightweight, hook-based stores without boilerplate. The application has **5 distinct stores** managing chat, user, classroom, notification, and UI state.

---

## 66.2 Store Inventory

| Store | File | State Size | Purpose |
|-------|------|-----------|---------|
| `useChatStore` | `stores/chatStore.ts` | ~15 fields | Chat sessions, messages, streaming |
| `useUserStore` | `stores/userStore.ts` | ~10 fields | User profile, role, preferences |
| `useClassroomStore` | `stores/classroomStore.ts` | ~8 fields | Active classroom, materials, subjects |
| `useNotificationStore` | `stores/notificationStore.ts` | ~5 fields | Notifications, unread count |
| `useUIStore` | `stores/uiStore.ts` | ~6 fields | Sidebar, modals, theme, loading |

---

## 66.3 Chat Store (Primary Store)

```typescript
import { create } from 'zustand';

interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
}

interface ChatState {
    messages: Message[];
    sessionId: string | null;
    isStreaming: boolean;
    currentSubject: string | null;
    classroomId: string | null;
    talLevel: number;
    
    // Actions
    addMessage: (msg: Message) => void;
    appendToLastMessage: (chunk: string) => void;
    setStreaming: (streaming: boolean) => void;
    clearMessages: () => void;
    setSession: (id: string) => void;
    setContext: (classroomId: string, subject: string) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
    messages: [],
    sessionId: null,
    isStreaming: false,
    currentSubject: null,
    classroomId: null,
    talLevel: 1,
    
    addMessage: (msg) => set((state) => ({
        messages: [...state.messages, msg]
    })),
    
    appendToLastMessage: (chunk) => set((state) => {
        const messages = [...state.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === 'assistant') {
            last.content += chunk;
        }
        return { messages };
    }),
    
    setStreaming: (streaming) => set({ isStreaming: streaming }),
    
    clearMessages: () => set({ messages: [], sessionId: null }),
    
    setSession: (id) => set({ sessionId: id }),
    
    setContext: (classroomId, subject) => set({ 
        classroomId, currentSubject: subject 
    })
}));
```

---

## 66.4 User Store

```typescript
interface UserState {
    user: User | null;
    role: 'student' | 'teacher' | 'parent' | 'admin' | null;
    accessToken: string | null;
    preferences: UserPreferences;
    
    setUser: (user: User) => void;
    logout: () => void;
    updatePreferences: (prefs: Partial<UserPreferences>) => void;
}

export const useUserStore = create<UserState>((set) => ({
    user: null,
    role: null,
    accessToken: null,
    preferences: { theme: 'dark', language: 'en' },
    
    setUser: (user) => set({ 
        user, role: user.role, accessToken: user.accessToken 
    }),
    logout: () => set({ user: null, role: null, accessToken: null }),
    updatePreferences: (prefs) => set((state) => ({
        preferences: { ...state.preferences, ...prefs }
    }))
}));
```

---

## 66.5 Classroom Store

```typescript
interface ClassroomState {
    activeClassroom: Classroom | null;
    classrooms: Classroom[];
    materials: Material[];
    subjects: Subject[];
    
    setActiveClassroom: (classroom: Classroom) => void;
    setClassrooms: (list: Classroom[]) => void;
    addMaterial: (material: Material) => void;
}
```

---

## 66.6 UI Store

```typescript
interface UIState {
    sidebarOpen: boolean;
    activeModal: string | null;
    theme: 'light' | 'dark';
    isLoading: boolean;
    toasts: Toast[];
    
    toggleSidebar: () => void;
    openModal: (name: string) => void;
    closeModal: () => void;
    addToast: (toast: Toast) => void;
    removeToast: (id: string) => void;
}
```

---

## 66.7 SSE Streaming Integration

```typescript
// Chat page uses store + SSE together
function ChatPage() {
    const { messages, addMessage, appendToLastMessage, setStreaming } = useChatStore();
    
    const sendMessage = async (text: string) => {
        addMessage({ role: 'user', content: text });
        addMessage({ role: 'assistant', content: '', isStreaming: true });
        setStreaming(true);
        
        const eventSource = new EventSource(`/api/tutor/chat?message=${text}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'done') {
                setStreaming(false);
                eventSource.close();
            } else {
                appendToLastMessage(data.content);
            }
        };
    };
}
```

---

## 66.8 Why Zustand over Redux

| Feature | Zustand | Redux |
|---------|---------|-------|
| Boilerplate | Minimal | Heavy |
| Bundle size | ~1 KB | ~7 KB |
| Provider needed | No | Yes |
| DevTools | Optional plugin | Built-in |
| Async actions | Native | Thunk/Saga |
| Learning curve | Low | High |
| TypeScript | First-class | Good |
