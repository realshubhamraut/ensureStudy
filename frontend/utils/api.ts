/**
 * API Client for ensureStudy services
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

interface RequestOptions {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
    body?: any
    token?: string
}

async function request<T>(url: string, options: RequestOptions = {}): Promise<T> {
    const { method = 'GET', body, token } = options

    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
    }

    if (token) {
        headers['Authorization'] = `Bearer ${token}`
    }

    const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Request failed' }))
        throw new Error(error.error || `Request failed with status ${response.status}`)
    }

    return response.json()
}

// Auth API
export const authApi = {
    login: (email: string, password: string) =>
        request(`${API_URL}/api/auth/login`, {
            method: 'POST',
            body: { email, password },
        }),

    register: (data: {
        email: string
        username: string
        password: string
        first_name?: string
        last_name?: string
        role?: string
    }) =>
        request(`${API_URL}/api/auth/register`, {
            method: 'POST',
            body: data,
        }),

    me: (token: string) =>
        request(`${API_URL}/api/auth/me`, { token }),
}

// Progress API
export const progressApi = {
    getProgress: (token: string, subject?: string) => {
        const url = subject
            ? `${API_URL}/api/progress?subject=${subject}`
            : `${API_URL}/api/progress`
        return request(url, { token })
    },

    getWeakTopics: (token: string) =>
        request(`${API_URL}/api/progress/weak-topics`, { token }),

    getSummary: (token: string) =>
        request(`${API_URL}/api/progress/summary`, { token }),
}

// AI Service API
export const aiApi = {
    chat: (message: string, token: string, sessionId?: string) =>
        request(`${AI_SERVICE_URL}/api/tutor/chat`, {
            method: 'POST',
            body: { message, session_id: sessionId },
            token,
        }),

    generateNotes: (topic: string, subject: string, token: string) =>
        request(`${AI_SERVICE_URL}/api/notes/generate`, {
            method: 'POST',
            body: { topic, subject },
            token,
        }),
}

export { API_URL, AI_SERVICE_URL }
