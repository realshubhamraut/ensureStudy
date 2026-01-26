/**
 * API Client for ensureStudy services
 * 
 * PORT CONFIGURATION:
 * - run-local.sh: Frontend 3000 → API 8000, AI 8001 (localhost only)
 * - run-lan.sh:   Frontend 4000 → API 9000, AI 9001 (network IP only)
 * 
 * Both can run SIMULTANEOUSLY in different terminals!
 */

// Dynamic URL getters - detect from window.location
export function getApiBaseUrl(): string {
    // Server-side rendering
    if (typeof window === 'undefined') {
        return process.env.NEXT_PUBLIC_API_URL || 'https://localhost:8000'
    }

    // Client-side: determine port based on frontend port
    const { hostname, port, protocol } = window.location

    // Port 3000 = run-local.sh → API on port 8000
    if (port === '3000') {
        return `${protocol}//${hostname}:8000`
    }

    // Port 4000 = run-lan.sh → API on port 9000
    if (port === '4000') {
        return `${protocol}//${hostname}:9000`
    }

    // Default: same hostname, port 8000
    return `${protocol}//${hostname}:8000`
}

export function getAiServiceUrl(): string {
    // Server-side rendering
    if (typeof window === 'undefined') {
        return process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'https://localhost:8001'
    }

    // Client-side: determine port based on frontend port
    const { hostname, port, protocol } = window.location

    // Port 3000 = run-local.sh → AI on port 8001
    if (port === '3000') {
        return `${protocol}//${hostname}:8001`
    }

    // Port 4000 = run-lan.sh → AI on port 9001
    if (port === '4000') {
        return `${protocol}//${hostname}:9001`
    }

    // Default: same hostname, port 8001
    return `${protocol}//${hostname}:8001`
}

/**
 * Fix file URLs that may have been stored with wrong port numbers.
 * This rewrites URLs like "http://localhost:9000/api/files/..." to use
 * the current API base URL.
 * 
 * @param url - The file URL to fix (may be absolute or relative)
 * @returns The fixed URL using the current API base URL
 */
export function fixFileUrl(url: string | undefined | null): string {
    if (!url) return ''

    // If it's already a relative path, make it absolute
    if (url.startsWith('/api/files/')) {
        return `${getApiBaseUrl()}${url}`
    }

    // Pattern to match common API file URLs with any port
    const filePathMatch = url.match(/https?:\/\/[^/]+\/api\/files\/(.+)$/)
    if (filePathMatch) {
        const filename = filePathMatch[1]
        return `${getApiBaseUrl()}/api/files/${filename}`
    }

    // For other absolute URLs with localhost:9000 or localhost:8000, fix them
    const localhostMatch = url.match(/https?:\/\/localhost:\d+(.*)/)
    if (localhostMatch) {
        return `${getApiBaseUrl()}${localhostMatch[1]}`
    }

    // Return as-is if no pattern matched
    return url
}

// Backward-compatible exports
export const API_URL = typeof window !== 'undefined' ? getApiBaseUrl() : 'https://localhost:8000'
export const AI_SERVICE_URL = typeof window !== 'undefined' ? getAiServiceUrl() : 'https://localhost:8001'

// Request helper
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

// Progress API
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

// AI Service API
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
