/**
 * useProctoring Hook - Real-time exam proctoring with AI analysis
 * 
 * Provides:
 * - Session management (start/stop)
 * - Frame streaming to backend
 * - Real-time integrity score updates
 * - Tab switch detection
 * - Temporal and static analysis results
 */

import { useState, useCallback, useRef, useEffect } from 'react'

const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

interface ProctorMetrics {
    integrityScore: number
    activeFlags: string[]
    cheatScore: number
    framesProcessed: number
    tabSwitchCount: number
    temporalReady: boolean
    staticAnalysis: {
        probability: number
        isCheat: boolean
        confidence: number
    } | null
    temporalAnalysis: {
        probability: number
        confidence: number
    } | null
}

interface UseProctoring {
    // State
    sessionId: string | null
    isActive: boolean
    isConnected: boolean
    metrics: ProctorMetrics
    error: string | null

    // Actions
    startSession: (assessmentId: string, studentId: string) => Promise<boolean>
    stopSession: () => Promise<void>
    sendFrame: (canvas: HTMLCanvasElement, timestamp: number) => Promise<void>
    recordTabSwitch: () => Promise<void>
    getBehaviorReport: () => Promise<any>
}

export function useProctoring(): UseProctoring {
    const [sessionId, setSessionId] = useState<string | null>(null)
    const [isActive, setIsActive] = useState(false)
    const [isConnected, setIsConnected] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const [metrics, setMetrics] = useState<ProctorMetrics>({
        integrityScore: 100,
        activeFlags: [],
        cheatScore: 0,
        framesProcessed: 0,
        tabSwitchCount: 0,
        temporalReady: false,
        staticAnalysis: null,
        temporalAnalysis: null
    })

    // Check API connectivity
    useEffect(() => {
        const checkConnection = async () => {
            try {
                const response = await fetch(`${AI_SERVICE_URL}/api/proctor/health`)
                if (response.ok) {
                    setIsConnected(true)
                }
            } catch (err) {
                console.warn('Proctoring API not available')
                setIsConnected(false)
            }
        }
        checkConnection()
    }, [])

    // Start proctoring session
    const startSession = useCallback(async (assessmentId: string, studentId: string): Promise<boolean> => {
        try {
            setError(null)

            const response = await fetch(`${AI_SERVICE_URL}/api/proctor/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    assessment_id: assessmentId,
                    student_id: studentId
                })
            })

            if (!response.ok) {
                throw new Error('Failed to start proctoring session')
            }

            const data = await response.json()
            setSessionId(data.session_id)
            setIsActive(true)

            console.log('[Proctoring] Session started:', data.session_id)
            return true

        } catch (err: any) {
            console.error('[Proctoring] Start error:', err)
            setError(err.message || 'Failed to start session')
            return false
        }
    }, [])

    // Stop proctoring session
    const stopSession = useCallback(async (): Promise<void> => {
        if (!sessionId) return

        try {
            await fetch(`${AI_SERVICE_URL}/api/proctor/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            })

            console.log('[Proctoring] Session stopped')

        } catch (err) {
            console.error('[Proctoring] Stop error:', err)
        } finally {
            setSessionId(null)
            setIsActive(false)
        }
    }, [sessionId])

    // Send frame for analysis
    const sendFrame = useCallback(async (canvas: HTMLCanvasElement, timestamp: number): Promise<void> => {
        if (!sessionId || !isActive) return

        try {
            const frameBase64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]

            const response = await fetch(`${AI_SERVICE_URL}/api/proctor/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    frame_base64: frameBase64,
                    timestamp
                })
            })

            if (response.ok) {
                const result = await response.json()

                if (result.processed) {
                    setMetrics(prev => ({
                        ...prev,
                        integrityScore: result.current_score,
                        activeFlags: result.active_flags || [],
                        framesProcessed: result.frame_count || prev.framesProcessed + 1,
                        cheatScore: result.unified_cheat_score
                            ? Math.round(result.unified_cheat_score * 100)
                            : prev.cheatScore,
                        temporalReady: result.temporal_analysis?.ready || prev.temporalReady,
                        staticAnalysis: result.static_analysis ? {
                            probability: result.static_analysis.probability,
                            isCheat: result.static_analysis.is_cheating,
                            confidence: result.static_analysis.confidence
                        } : prev.staticAnalysis,
                        temporalAnalysis: result.temporal_analysis?.ready ? {
                            probability: result.temporal_analysis.probability,
                            confidence: result.temporal_analysis.confidence
                        } : prev.temporalAnalysis
                    }))
                }
            }

        } catch (err) {
            console.error('[Proctoring] Frame send error:', err)
        }
    }, [sessionId, isActive])

    // Record tab switch
    const recordTabSwitch = useCallback(async (): Promise<void> => {
        if (!sessionId || !isActive) return

        try {
            await fetch(`${AI_SERVICE_URL}/api/proctor/tab-switch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            })

            setMetrics(prev => ({
                ...prev,
                tabSwitchCount: prev.tabSwitchCount + 1
            }))

        } catch (err) {
            console.error('[Proctoring] Tab switch error:', err)
        }
    }, [sessionId, isActive])

    // Get behavior report
    const getBehaviorReport = useCallback(async (): Promise<any> => {
        if (!sessionId) return null

        try {
            const response = await fetch(`${AI_SERVICE_URL}/api/proctor/behavior-report/${sessionId}`)

            if (response.ok) {
                return await response.json()
            }

        } catch (err) {
            console.error('[Proctoring] Behavior report error:', err)
        }

        return null
    }, [sessionId])

    return {
        sessionId,
        isActive,
        isConnected,
        metrics,
        error,
        startSession,
        stopSession,
        sendFrame,
        recordTabSwitch,
        getBehaviorReport
    }
}

// Hook for automatic frame capture and streaming
export function useFrameStreaming(
    videoRef: React.RefObject<HTMLVideoElement | null>,
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
    sendFrame: (canvas: HTMLCanvasElement, timestamp: number) => Promise<void>,
    isActive: boolean,
    fps: number = 2
) {
    const startTimeRef = useRef<number>(Date.now())
    const intervalRef = useRef<NodeJS.Timeout | null>(null)

    useEffect(() => {
        if (!isActive || !videoRef.current || !canvasRef.current) {
            if (intervalRef.current) {
                clearInterval(intervalRef.current)
                intervalRef.current = null
            }
            return
        }

        startTimeRef.current = Date.now()

        const captureAndSend = () => {
            const video = videoRef.current
            const canvas = canvasRef.current

            if (!video || !canvas || video.readyState < 2) return

            const ctx = canvas.getContext('2d')
            if (!ctx) return

            canvas.width = 640
            canvas.height = 480
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

            const timestamp = (Date.now() - startTimeRef.current) / 1000
            sendFrame(canvas, timestamp)
        }

        intervalRef.current = setInterval(captureAndSend, 1000 / fps)

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current)
                intervalRef.current = null
            }
        }
    }, [isActive, sendFrame, fps, videoRef, canvasRef])
}
