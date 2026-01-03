/**
 * API Client for ensureStudy services
 * Automatically detects hostname for LAN access support
 */

// Dynamic URL getters that work with LAN access
function getApiBaseUrl(): string {
    if (typeof window === 'undefined') {
        return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    }
    const hostname = window.location.hostname
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:8000'
    }
    return `http://${hostname}:8000`
}

function getAiServiceUrl(): string {
    if (typeof window === 'undefined') {
        return process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'
    }
    const hostname = window.location.hostname
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:8001'
    }
    return `http://${hostname}:8001`
}

// Export both static (for backwards compatibility) and dynamic getters
const API_URL = typeof window !== 'undefined' ? getApiBaseUrl() : 'http://localhost:8000'
const AI_SERVICE_URL = typeof window !== 'undefined' ? getAiServiceUrl() : 'http://localhost:8001'

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

// Auth API - uses dynamic URL for LAN access
export const authApi = {
    login: (email: string, password: string) =>
        request(`${getApiBaseUrl()}/api/auth/login`, {
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
        request(`${getApiBaseUrl()}/api/auth/register`, {
            method: 'POST',
            body: data,
        }),

    me: (token: string) =>
        request(`${getApiBaseUrl()}/api/auth/me`, { token }),
}

// Progress API - uses dynamic URL for LAN access
export const progressApi = {
    getProgress: (token: string, subject?: string) => {
        const baseUrl = getApiBaseUrl()
        const url = subject
            ? `${baseUrl}/api/progress?subject=${subject}`
            : `${baseUrl}/api/progress`
        return request(url, { token })
    },

    getWeakTopics: (token: string) =>
        request(`${getApiBaseUrl()}/api/progress/weak-topics`, { token }),

    getSummary: (token: string) =>
        request(`${getApiBaseUrl()}/api/progress/summary`, { token }),
}

// AI Service API - uses dynamic URL for LAN access
export const aiApi = {
    chat: (message: string, token: string, sessionId?: string) =>
        request(`${getAiServiceUrl()}/api/tutor/chat`, {
            method: 'POST',
            body: { message, session_id: sessionId },
            token,
        }),

    generateNotes: (topic: string, subject: string, token: string) =>
        request(`${getAiServiceUrl()}/api/notes/generate`, {
            method: 'POST',
            body: { topic, subject },
            token,
        }),
}

export { API_URL, AI_SERVICE_URL, getApiBaseUrl, getAiServiceUrl }
