'use client'

import { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import {
    XMarkIcon,
    VideoCameraIcon,
    ClockIcon,
    ExclamationTriangleIcon,
    ShieldCheckIcon,
    ShieldExclamationIcon,
    EyeIcon,
    DocumentTextIcon,
    ArrowPathIcon
} from '@heroicons/react/24/outline'

// API configuration
const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

type SessionState = 'ready' | 'proctoring' | 'complete' | 'error'
type PermissionState = 'pending' | 'granted' | 'denied' | 'error'

interface ProctorResult {
    processed: boolean
    current_score: number
    active_flags: string[]
    frame_count?: number
    temporal_analysis?: {
        probability: number
        confidence: number
        ready: boolean
    }
    static_analysis?: {
        probability: number
        is_cheating: boolean
        confidence: number
    }
    unified_cheat_score?: number
}

function ProctoredExamContent() {
    const searchParams = useSearchParams()
    const router = useRouter()

    const assessmentId = searchParams.get('assessment_id') || 'demo'
    const studentId = searchParams.get('student_id') || 'student_1'

    const [sessionState, setSessionState] = useState<SessionState>('ready')
    const [sessionId, setSessionId] = useState<string | null>(null)
    const [permissionState, setPermissionState] = useState<PermissionState>('pending')
    const [permissionError, setPermissionError] = useState('')
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [timeElapsed, setTimeElapsed] = useState(0)

    // Proctoring metrics
    const [integrityScore, setIntegrityScore] = useState(100)
    const [activeFlags, setActiveFlags] = useState<string[]>([])
    const [cheatScore, setCheatScore] = useState(0)
    const [framesProcessed, setFramesProcessed] = useState(0)
    const [tabSwitchCount, setTabSwitchCount] = useState(0)
    const [temporalReady, setTemporalReady] = useState(false)

    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const mediaStreamRef = useRef<MediaStream | null>(null)
    const timerRef = useRef<NodeJS.Timeout | null>(null)
    const frameIntervalRef = useRef<NodeJS.Timeout | null>(null)

    // Start proctoring session
    const startSession = useCallback(async () => {
        try {
            const response = await fetch(`${AI_SERVICE_URL}/api/proctor/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    assessment_id: assessmentId,
                    student_id: studentId
                })
            })

            if (!response.ok) throw new Error('Failed to start proctoring session')

            const data = await response.json()
            setSessionId(data.session_id)
            return data.session_id
        } catch (error) {
            console.error('Error starting session:', error)
            setSessionState('error')
            return null
        }
    }, [assessmentId, studentId])

    // Stream frame to backend
    const streamFrame = useCallback(async (sessionIdParam: string) => {
        if (!videoRef.current || !canvasRef.current) return

        const canvas = canvasRef.current
        const video = videoRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Draw video frame to canvas
        canvas.width = 640
        canvas.height = 480
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        // Convert to base64 JPEG
        const frameBase64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]

        try {
            const response = await fetch(`${AI_SERVICE_URL}/api/proctor/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionIdParam,
                    frame_base64: frameBase64,
                    timestamp: timeElapsed
                })
            })

            if (response.ok) {
                const result: ProctorResult = await response.json()

                if (result.processed) {
                    setIntegrityScore(result.current_score)
                    setActiveFlags(result.active_flags || [])
                    setFramesProcessed(result.frame_count || 0)

                    if (result.unified_cheat_score !== undefined) {
                        setCheatScore(Math.round(result.unified_cheat_score * 100))
                    }

                    if (result.temporal_analysis?.ready) {
                        setTemporalReady(true)
                    }
                }
            }
        } catch (error) {
            console.error('Error streaming frame:', error)
        }
    }, [timeElapsed])

    // Handle tab visibility change
    useEffect(() => {
        const handleVisibilityChange = async () => {
            if (document.hidden && sessionId && sessionState === 'proctoring') {
                setTabSwitchCount(prev => prev + 1)

                try {
                    await fetch(`${AI_SERVICE_URL}/api/proctor/tab-switch`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: sessionId })
                    })
                } catch (error) {
                    console.error('Error recording tab switch:', error)
                }
            }
        }

        document.addEventListener('visibilitychange', handleVisibilityChange)
        return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
    }, [sessionId, sessionState])

    // Start camera
    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: true
            })
            mediaStreamRef.current = stream

            if (videoRef.current) {
                videoRef.current.srcObject = stream
                await videoRef.current.play()
            }

            setIsCameraOn(true)
            setPermissionState('granted')
            return true
        } catch (err: any) {
            console.error('Camera access denied:', err)
            setPermissionState('denied')
            setPermissionError('Camera and microphone access is required for proctored exams.')
            return false
        }
    }, [])

    // Stop camera
    const stopCamera = useCallback(() => {
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop())
            mediaStreamRef.current = null
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null
        }
        setIsCameraOn(false)
    }, [])

    // Start proctoring
    const startProctoring = useCallback(async () => {
        const cameraStarted = await startCamera()
        if (!cameraStarted) return

        const newSessionId = await startSession()
        if (!newSessionId) return

        setSessionState('proctoring')

        // Start timer
        timerRef.current = setInterval(() => {
            setTimeElapsed(prev => prev + 1)
        }, 1000)

        // Start frame streaming (2 FPS)
        frameIntervalRef.current = setInterval(() => {
            streamFrame(newSessionId)
        }, 500)
    }, [startCamera, startSession, streamFrame])

    // Stop proctoring
    const stopProctoring = useCallback(async () => {
        if (timerRef.current) clearInterval(timerRef.current)
        if (frameIntervalRef.current) clearInterval(frameIntervalRef.current)

        if (sessionId) {
            try {
                await fetch(`${AI_SERVICE_URL}/api/proctor/stop`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                })
            } catch (error) {
                console.error('Error stopping session:', error)
            }
        }

        stopCamera()
        setSessionState('complete')
    }, [sessionId, stopCamera])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current)
            if (frameIntervalRef.current) clearInterval(frameIntervalRef.current)
            stopCamera()
        }
    }, [stopCamera])

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    const getScoreColor = (score: number) => {
        if (score >= 80) return 'text-green-500'
        if (score >= 60) return 'text-yellow-500'
        if (score >= 40) return 'text-orange-500'
        return 'text-red-500'
    }

    const getScoreBg = (score: number) => {
        if (score >= 80) return 'bg-green-500'
        if (score >= 60) return 'bg-yellow-500'
        if (score >= 40) return 'bg-orange-500'
        return 'bg-red-500'
    }

    // Permission denied state
    if (permissionState === 'denied') {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
                <div className="max-w-md mx-auto">
                    <div className="bg-white rounded-3xl shadow-xl p-8 text-center">
                        <div className="w-20 h-20 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-6">
                            <ExclamationTriangleIcon className="w-10 h-10 text-red-600" />
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 mb-2">Permission Required</h1>
                        <p className="text-gray-500 mb-6">{permissionError}</p>

                        <div className="space-y-3">
                            <button
                                onClick={() => {
                                    setPermissionState('pending')
                                    setPermissionError('')
                                    startProctoring()
                                }}
                                className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-medium text-white hover:shadow-lg transition-all"
                            >
                                Try Again
                            </button>
                            <Link
                                href="/assessments"
                                className="block w-full py-3 rounded-xl border border-gray-200 font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                            >
                                Go Back
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    // Complete state
    if (sessionState === 'complete') {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
                <div className="max-w-2xl mx-auto">
                    <div className="bg-white rounded-3xl shadow-xl p-8">
                        <div className="text-center mb-8">
                            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 ${integrityScore >= 70 ? 'bg-green-100' : 'bg-orange-100'
                                }`}>
                                {integrityScore >= 70 ? (
                                    <ShieldCheckIcon className="w-10 h-10 text-green-600" />
                                ) : (
                                    <ShieldExclamationIcon className="w-10 h-10 text-orange-600" />
                                )}
                            </div>
                            <h1 className="text-2xl font-bold text-gray-900 mb-2">Exam Complete</h1>
                            <p className="text-gray-500">Proctoring session has ended</p>
                        </div>

                        {/* Final Score */}
                        <div className={`rounded-2xl p-6 text-white text-center mb-6 ${getScoreBg(integrityScore)}`}>
                            <p className="text-sm opacity-80 mb-1">Final Integrity Score</p>
                            <p className="text-5xl font-bold">{integrityScore}%</p>
                        </div>

                        {/* Stats Grid */}
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="bg-gray-50 rounded-xl p-4 text-center">
                                <p className="text-sm text-gray-500">Duration</p>
                                <p className="text-2xl font-bold text-gray-900">{formatTime(timeElapsed)}</p>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-4 text-center">
                                <p className="text-sm text-gray-500">Frames Analyzed</p>
                                <p className="text-2xl font-bold text-gray-900">{framesProcessed}</p>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-4 text-center">
                                <p className="text-sm text-gray-500">Tab Switches</p>
                                <p className="text-2xl font-bold text-gray-900">{tabSwitchCount}</p>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-4 text-center">
                                <p className="text-sm text-gray-500">Flags Triggered</p>
                                <p className="text-2xl font-bold text-gray-900">{activeFlags.length}</p>
                            </div>
                        </div>

                        {/* Flags List */}
                        {activeFlags.length > 0 && (
                            <div className="bg-orange-50 rounded-xl p-4 mb-6">
                                <h3 className="font-semibold text-orange-800 mb-2">Flags Detected</h3>
                                <ul className="space-y-1">
                                    {activeFlags.map((flag, idx) => (
                                        <li key={idx} className="text-sm text-orange-700 flex items-center gap-2">
                                            <ExclamationTriangleIcon className="w-4 h-4" />
                                            {flag}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        <Link
                            href="/assessments"
                            className="block w-full py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-medium text-white hover:shadow-lg transition-all text-center"
                        >
                            Back to Assessments
                        </Link>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-blue-900 text-white">
            {/* Header */}
            <div className="bg-black/30 backdrop-blur-sm border-b border-white/10 px-6 py-4">
                <div className="max-w-6xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/assessments" className="text-white/60 hover:text-white">
                            <XMarkIcon className="w-6 h-6" />
                        </Link>
                        <div>
                            <h1 className="font-semibold">Proctored Exam</h1>
                            <p className="text-sm text-white/60">AI-Powered Integrity Monitoring</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-6">
                        {sessionState === 'proctoring' && (
                            <>
                                <div className="flex items-center gap-2">
                                    <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                                    <span className="text-sm text-red-400">LIVE</span>
                                </div>
                                <div className="flex items-center gap-2 text-white/60">
                                    <ClockIcon className="w-4 h-4" />
                                    <span className="text-sm font-mono">{formatTime(timeElapsed)}</span>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            <div className="max-w-6xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Camera View */}
                    <div className="lg:col-span-2 space-y-4">
                        <div className="relative aspect-video bg-gray-800 rounded-2xl overflow-hidden shadow-2xl">
                            {isCameraOn ? (
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                    <VideoCameraIcon className="w-16 h-16 text-gray-600" />
                                </div>
                            )}

                            {/* Overlay badges */}
                            {sessionState === 'proctoring' && (
                                <>
                                    <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/90 px-3 py-1 rounded-full">
                                        <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                        <span className="text-xs font-medium">Recording</span>
                                    </div>

                                    <div className={`absolute top-4 right-4 px-3 py-1 rounded-full ${integrityScore >= 80 ? 'bg-green-500/90' :
                                            integrityScore >= 60 ? 'bg-yellow-500/90' : 'bg-red-500/90'
                                        }`}>
                                        <span className="text-xs font-medium">Score: {integrityScore}%</span>
                                    </div>

                                    {temporalReady && (
                                        <div className="absolute bottom-4 left-4 flex items-center gap-2 bg-blue-500/90 px-3 py-1 rounded-full">
                                            <EyeIcon className="w-4 h-4" />
                                            <span className="text-xs font-medium">AI Analysis Active</span>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>

                        {/* Hidden canvas for frame capture */}
                        <canvas ref={canvasRef} className="hidden" />

                        {/* Exam Content Placeholder */}
                        {sessionState === 'proctoring' && (
                            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
                                <div className="flex items-center gap-3 mb-4">
                                    <DocumentTextIcon className="w-5 h-5 text-blue-400" />
                                    <h3 className="font-semibold">Exam Questions</h3>
                                </div>
                                <p className="text-white/60 text-sm">
                                    Your exam questions will appear here. This is a demo of the proctoring interface.
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Metrics Sidebar */}
                    <div className="space-y-4">
                        {sessionState === 'ready' && (
                            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-center">
                                <ShieldCheckIcon className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                                <h3 className="text-xl font-semibold mb-2">Ready to Start</h3>
                                <p className="text-white/60 text-sm mb-6">
                                    Your webcam will be monitored throughout the exam to ensure academic integrity.
                                </p>
                                <button
                                    onClick={startProctoring}
                                    className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-semibold hover:shadow-lg transition-all"
                                >
                                    Start Proctored Exam
                                </button>
                            </div>
                        )}

                        {sessionState === 'proctoring' && (
                            <>
                                {/* Integrity Score */}
                                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6">
                                    <h3 className="font-semibold mb-4">Integrity Score</h3>
                                    <div className="text-center mb-4">
                                        <span className={`text-5xl font-bold ${getScoreColor(integrityScore)}`}>
                                            {integrityScore}%
                                        </span>
                                    </div>
                                    <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-500 ${getScoreBg(integrityScore)}`}
                                            style={{ width: `${integrityScore}%` }}
                                        />
                                    </div>
                                </div>

                                {/* AI Analysis */}
                                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6">
                                    <h3 className="font-semibold mb-4">AI Analysis</h3>
                                    <div className="space-y-3">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-white/60">Cheat Probability</span>
                                            <span className={cheatScore > 50 ? 'text-red-400' : 'text-green-400'}>
                                                {cheatScore}%
                                            </span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-white/60">Frames Analyzed</span>
                                            <span>{framesProcessed}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-white/60">Tab Switches</span>
                                            <span className={tabSwitchCount > 2 ? 'text-orange-400' : ''}>
                                                {tabSwitchCount}
                                            </span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-white/60">Temporal Model</span>
                                            <span className={temporalReady ? 'text-green-400' : 'text-yellow-400'}>
                                                {temporalReady ? 'Active' : 'Warming up...'}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Active Flags */}
                                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6">
                                    <h3 className="font-semibold mb-4">Active Flags</h3>
                                    {activeFlags.length === 0 ? (
                                        <p className="text-sm text-green-400 flex items-center gap-2">
                                            <ShieldCheckIcon className="w-4 h-4" />
                                            No issues detected
                                        </p>
                                    ) : (
                                        <ul className="space-y-2">
                                            {activeFlags.map((flag, idx) => (
                                                <li key={idx} className="text-sm text-orange-400 flex items-center gap-2">
                                                    <ExclamationTriangleIcon className="w-4 h-4" />
                                                    {flag}
                                                </li>
                                            ))}
                                        </ul>
                                    )}
                                </div>

                                {/* End Exam Button */}
                                <button
                                    onClick={stopProctoring}
                                    className="w-full py-3 rounded-xl bg-red-500/20 border border-red-500/40 text-red-400 font-medium hover:bg-red-500/30 transition-colors"
                                >
                                    End Exam
                                </button>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

function ExamLoading() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-blue-900 flex items-center justify-center">
            <div className="text-center text-white">
                <ArrowPathIcon className="w-12 h-12 animate-spin mx-auto mb-4" />
                <p className="text-white/60">Loading proctored exam...</p>
            </div>
        </div>
    )
}

export default function ProctoredExamPage() {
    return (
        <Suspense fallback={<ExamLoading />}>
            <ProctoredExamContent />
        </Suspense>
    )
}
