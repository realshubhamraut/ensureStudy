/**
 * Frontend Logging Utilities
 * 
 * Logs user interactions to the backend for debugging.
 * Also logs to browser console for development.
 */

// Action types
export type ActionType = 'click' | 'scroll' | 'navigate' | 'input' | 'hover' | 'error' | 'api_call' | 'api_response';

// Log action to backend
async function logToBackend(action: ActionType, target: string, details?: string) {
    try {
        await fetch('http://localhost:8001/api/log/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, target, details })
        });
    } catch {
        // Silently fail if backend isn't running
    }
}

// Console styling
const styles: Record<ActionType, string> = {
    click: 'color: #10b981; font-weight: bold;',
    scroll: 'color: #6366f1; font-weight: bold;',
    navigate: 'color: #f59e0b; font-weight: bold;',
    input: 'color: #8b5cf6; font-weight: bold;',
    hover: 'color: #06b6d4; font-weight: bold;',
    error: 'color: #ef4444; font-weight: bold;',
    api_call: 'color: #3b82f6; font-weight: bold;',
    api_response: 'color: #22c55e; font-weight: bold;',
};

// Icons
const icons: Record<ActionType, string> = {
    click: '👆',
    scroll: '📜',
    navigate: '🧭',
    input: '⌨️',
    hover: '👀',
    error: '❌',
    api_call: '📡',
    api_response: '📥',
};

/**
 * Log a user action
 */
export function logAction(action: ActionType, target: string, details?: string) {
    const icon = icons[action] || '📌';
    const style = styles[action] || '';

    // Console log
    console.log(
        `%c${icon} [${action.toUpperCase()}]`,
        style,
        target,
        details ? `- ${details}` : ''
    );

    // Backend log
    logToBackend(action, target, details);
}

/**
 * Log a click event
 */
export function logClick(target: string, details?: string) {
    logAction('click', target, details);
}

/**
 * Log a scroll event (debounced)
 */
let scrollTimeout: ReturnType<typeof setTimeout> | null = null;
export function logScroll(target: string, details?: string) {
    if (scrollTimeout) clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
        logAction('scroll', target, details);
    }, 300);
}

/**
 * Log navigation
 */
export function logNavigate(path: string, details?: string) {
    logAction('navigate', path, details);
}

/**
 * Log input change
 */
export function logInput(target: string, details?: string) {
    logAction('input', target, details);
}

/**
 * Log API call
 */
export function logApiCall(endpoint: string, method: string, body?: unknown) {
    const details = body ? `${method} with body: ${JSON.stringify(body).slice(0, 100)}...` : method;
    logAction('api_call', endpoint, details);
}

/**
 * Log API response
 */
export function logApiResponse(endpoint: string, status: number, duration?: number) {
    const details = duration ? `${status} in ${duration}ms` : String(status);
    logAction('api_response', endpoint, details);
}

/**
 * Log error
 */
export function logError(context: string, error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    logAction('error', context, message);
}

/**
 * Create a wrapped fetch that logs calls
 */
export async function loggedFetch(url: string, options?: RequestInit): Promise<Response> {
    const start = Date.now();
    const method = options?.method || 'GET';

    logApiCall(url, method, options?.body ? JSON.parse(options.body as string) : undefined);

    try {
        const response = await fetch(url, options);
        const duration = Date.now() - start;
        logApiResponse(url, response.status, duration);
        return response;
    } catch (error) {
        logError(`fetch:${url}`, error);
        throw error;
    }
}
