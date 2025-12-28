'use client'

export default function GlobalError({
    error,
    reset,
}: {
    error: Error & { digest?: string }
    reset: () => void
}) {
    return (
        <html>
            <body>
                <div style={{
                    minHeight: '100vh',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: '#f9fafb',
                    fontFamily: 'system-ui, -apple-system, sans-serif'
                }}>
                    <div style={{ textAlign: 'center', padding: '32px', maxWidth: '400px' }}>
                        <div style={{
                            width: '80px',
                            height: '80px',
                            margin: '0 auto 24px',
                            borderRadius: '50%',
                            background: '#fee2e2',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <svg width="40" height="40" fill="none" stroke="#ef4444" strokeWidth="2" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                        </div>
                        <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', marginBottom: '8px' }}>
                            Something went wrong!
                        </h2>
                        <p style={{ color: '#6b7280', marginBottom: '24px' }}>
                            A critical error occurred. Please refresh the page.
                        </p>
                        <button
                            onClick={reset}
                            style={{
                                padding: '12px 24px',
                                background: '#6366f1',
                                color: 'white',
                                border: 'none',
                                borderRadius: '8px',
                                fontSize: '16px',
                                cursor: 'pointer'
                            }}
                        >
                            Try again
                        </button>
                    </div>
                </div>
            </body>
        </html>
    )
}
