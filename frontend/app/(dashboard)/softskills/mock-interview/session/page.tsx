'use client'

import { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import {
    ArrowLeftIcon,
    MicrophoneIcon,
    StopIcon,
    CheckCircleIcon,
    XMarkIcon,
    SpeakerWaveIcon,
    VideoCameraIcon,
    ClockIcon,
    ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { useSpeechEngine, useSpeechRecognition } from '@/components/avatar/SpeechEngine'

// Dynamic import for Avatar (uses Three.js which needs client-side only)
const AvatarViewer = dynamic(() => import('@/components/avatar/AvatarViewer'), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full bg-gray-200 animate-pulse rounded-2xl flex items-center justify-center">
            <span className="text-gray-400">Loading Avatar...</span>
        </div>
    )
})

// Mock questions for demo
const DEMO_QUESTIONS = [
    "Can you explain Newton's first law of motion and give an example from everyday life?",
    "What is the relationship between force, mass, and acceleration?",
    "Describe the concept of inertia and how it affects objects at rest and in motion.",
    "How does friction affect the motion of objects?",
    "Explain the difference between speed and velocity."
]

type SessionState = 'ready' | 'speaking' | 'listening' | 'processing' | 'complete'
type PermissionState = 'pending' | 'granted' | 'denied' | 'error'

function InterviewSessionContent() {
    const searchParams = useSearchParams()
    const router = useRouter()

    const subject = searchParams.get('subject') || 'physics'
    const chapter = searchParams.get('chapter') || 'Mechanics'
    const avatarId = (searchParams.get('avatar') || 'female') as 'male' | 'female'

    const [sessionState, setSessionState] = useState<SessionState>('ready')
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
    const [answers, setAnswers] = useState<string[]>([])
    const [scores, setScores] = useState<number[]>([])
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [avatarReady, setAvatarReady] = useState(false)
    const [timeElapsed, setTimeElapsed] = useState(0)
    const [permissionState, setPermissionState] = useState<PermissionState>('pending')
    const [permissionError, setPermissionError] = useState<string>('')

    const videoRef = useRef<HTMLVideoElement>(null)
    const mediaStreamRef = useRef<MediaStream | null>(null)
    const timerRef = useRef<NodeJS.Timeout | null>(null)

    const { speak, stop: stopSpeaking, isSpeaking, isSupported: ttsSupported } = useSpeechEngine({
        onSpeakStart: () => setSessionState('speaking'),
        onSpeakEnd: () => setSessionState('listening')
    })

    const {
        isListening,
        transcript,
        startListening,
        stopListening,
        resetTranscript,
        isSupported: sttSupported
    } = useSpeechRecognition()

    const currentQuestion = DEMO_QUESTIONS[currentQuestionIndex]

    // Check permissions on mount
    useEffect(() => {
        checkPermissions()
    }, [])

    const checkPermissions = async () => {
        try {
            // Check if permissions API is available
            if (navigator.permissions) {
                const cameraPermission = await navigator.permissions.query({ name: 'camera' as PermissionName })
                const micPermission = await navigator.permissions.query({ name: 'microphone' as PermissionName })

                if (cameraPermission.state === 'denied' || micPermission.state === 'denied') {
                    setPermissionState('denied')
                    setPermissionError('Camera or microphone access was denied. Please enable permissions in your browser settings.')
                    return
                }
            }
            setPermissionState('pending')
        } catch (err) {
            // Permissions API not fully supported, will check on start
            setPermissionState('pending')
        }
    }

    // Effect to attach stream to video element when both are available
    useEffect(() => {
        if (isCameraOn && videoRef.current && mediaStreamRef.current) {
            videoRef.current.srcObject = mediaStreamRef.current
            videoRef.current.play().catch(err => {
                console.warn('Video autoplay blocked:', err)
            })
        }
    }, [isCameraOn])

    // Start camera
    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false
            })
            mediaStreamRef.current = stream
            setIsCameraOn(true)
            setPermissionState('granted')
            return true
        } catch (err: any) {
            console.error('Camera access denied:', err)
            setPermissionState('denied')
            if (err.name === 'NotAllowedError') {
                setPermissionError('Camera access was denied. Please allow camera access to continue.')
            } else if (err.name === 'NotFoundError') {
                setPermissionError('No camera found. Please connect a camera and try again.')
            } else if (err.name === 'NotReadableError') {
                setPermissionError('Camera is already in use by another application.')
            } else {
                setPermissionError('Failed to access camera. Please check your device settings.')
            }
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

    // Start the interview
    const startInterview = useCallback(async () => {
        const cameraStarted = await startCamera()
        if (!cameraStarted) return

        // Start timer
        timerRef.current = setInterval(() => {
            setTimeElapsed(prev => prev + 1)
        }, 1000)

        // Ask first question
        setSessionState('speaking')
        await speak(currentQuestion)
    }, [startCamera, speak, currentQuestion])

    // Submit current answer and move to next question
    const submitAnswer = useCallback(async () => {
        stopListening()

        // Save answer
        const answer = transcript.trim()
        setAnswers(prev => [...prev, answer])

        // Mock scoring (in production, call backend API)
        const mockScore = Math.floor(Math.random() * 30) + 70 // 70-100
        setScores(prev => [...prev, mockScore])

        resetTranscript()

        // Move to next question or complete
        if (currentQuestionIndex < DEMO_QUESTIONS.length - 1) {
            setCurrentQuestionIndex(prev => prev + 1)
            setSessionState('speaking')
            await speak(DEMO_QUESTIONS[currentQuestionIndex + 1])
        } else {
            // Interview complete
            setSessionState('complete')
            stopCamera()
            if (timerRef.current) {
                clearInterval(timerRef.current)
            }
        }
    }, [transcript, currentQuestionIndex, stopListening, resetTranscript, speak, stopCamera])

    // Format time
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCamera()
            stopSpeaking()
            if (timerRef.current) {
                clearInterval(timerRef.current)
            }
        }
    }, [stopCamera, stopSpeaking])

    // Calculate average score
    const averageScore = scores.length > 0
        ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length)
        : 0

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
                                    startInterview()
                                }}
                                className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-medium text-white hover:shadow-lg transition-all"
                            >
                                Try Again
                            </button>
                            <Link
                                href="/softskills/mock-interview"
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

    if (sessionState === 'complete') {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
                <div className="max-w-2xl mx-auto">
                    <div className="bg-white rounded-3xl shadow-xl p-8 text-center">
                        <div className="w-20 h-20 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-6">
                            <CheckCircleIcon className="w-10 h-10 text-green-600" />
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 mb-2">Interview Complete!</h1>
                        <p className="text-gray-500 mb-6">Great job completing the mock interview</p>

                        <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl p-6 text-white mb-6">
                            <p className="text-sm text-blue-200 mb-1">Overall Score</p>
                            <p className="text-5xl font-bold">{averageScore}%</p>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="bg-gray-50 rounded-xl p-4">
                                <p className="text-sm text-gray-500">Questions</p>
                                <p className="text-2xl font-bold text-gray-900">{DEMO_QUESTIONS.length}</p>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-4">
                                <p className="text-sm text-gray-500">Duration</p>
                                <p className="text-2xl font-bold text-gray-900">{formatTime(timeElapsed)}</p>
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <Link
                                href="/softskills/mock-interview"
                                className="flex-1 py-3 rounded-xl border border-gray-200 font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                            >
                                Try Again
                            </Link>
                            <Link
                                href="/softskills"
                                className="flex-1 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-medium text-white hover:shadow-lg transition-all"
                            >
                                Back to Soft Skills
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
            {/* Header */}
            <div className="bg-black/30 backdrop-blur-sm border-b border-white/10 px-6 py-4">
                <div className="max-w-6xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/softskills/mock-interview" className="text-white/60 hover:text-white">
                            <XMarkIcon className="w-6 h-6" />
                        </Link>
                        <div>
                            <h1 className="font-semibold">Mock Interview</h1>
                            <p className="text-sm text-white/60">{chapter} • {subject}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-white/60">
                            <ClockIcon className="w-4 h-4" />
                            <span className="text-sm font-mono">{formatTime(timeElapsed)}</span>
                        </div>
                        <div className="text-sm">
                            Question {currentQuestionIndex + 1} of {DEMO_QUESTIONS.length}
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-6xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Avatar Section */}
                    <div className="space-y-4">
                        <div className="aspect-square max-h-[500px] rounded-2xl overflow-hidden shadow-2xl">
                            <AvatarViewer
                                avatarId={avatarId}
                                isSpeaking={isSpeaking}
                                onReady={() => setAvatarReady(true)}
                            />
                        </div>

                        {/* Speaking indicator */}
                        {isSpeaking && (
                            <div className="flex items-center justify-center gap-2 text-blue-400">
                                <SpeakerWaveIcon className="w-5 h-5 animate-pulse" />
                                <span className="text-sm">Avatar is speaking...</span>
                            </div>
                        )}
                    </div>

                    {/* User Section */}
                    <div className="space-y-6">
                        {/* User Camera */}
                        <div className="relative aspect-video bg-gray-800 rounded-2xl overflow-hidden shadow-xl">
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
                                    <VideoCameraIcon className="w-12 h-12 text-gray-600" />
                                </div>
                            )}

                            {/* Recording indicator */}
                            {isListening && (
                                <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 px-3 py-1 rounded-full">
                                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                    <span className="text-xs font-medium">Recording</span>
                                </div>
                            )}
                        </div>

                        {/* Question Display */}
                        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6">
                            <p className="text-sm text-white/60 mb-2">Current Question:</p>
                            <p className="text-lg font-medium">{currentQuestion}</p>
                        </div>

                        {/* Transcript Display */}
                        {(isListening || transcript) && (
                            <div className="bg-white/5 rounded-2xl p-6">
                                <p className="text-sm text-white/60 mb-2">Your Answer:</p>
                                <p className="text-white/90">
                                    {transcript || <span className="text-white/40 italic">Listening...</span>}
                                </p>
                            </div>
                        )}

                        {/* Controls */}
                        <div className="flex gap-4">
                            {sessionState === 'ready' && (
                                <button
                                    onClick={startInterview}
                                    disabled={!avatarReady}
                                    className="flex-1 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all disabled:opacity-50"
                                >
                                    Start Interview
                                </button>
                            )}

                            {sessionState === 'listening' && !isListening && (
                                <button
                                    onClick={startListening}
                                    className="flex-1 py-4 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all"
                                >
                                    <MicrophoneIcon className="w-5 h-5" />
                                    Start Speaking
                                </button>
                            )}

                            {isListening && (
                                <>
                                    <button
                                        onClick={stopListening}
                                        className="flex-1 py-4 rounded-xl bg-gradient-to-r from-red-500 to-rose-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all"
                                    >
                                        <StopIcon className="w-5 h-5" />
                                        Stop Recording
                                    </button>
                                    <button
                                        onClick={submitAnswer}
                                        disabled={!transcript.trim()}
                                        className="flex-1 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all disabled:opacity-50"
                                    >
                                        <CheckCircleIcon className="w-5 h-5" />
                                        Submit Answer
                                    </button>
                                </>
                            )}

                            {sessionState === 'speaking' && (
                                <div className="flex-1 py-4 rounded-xl bg-white/10 font-medium flex items-center justify-center gap-2">
                                    <SpeakerWaveIcon className="w-5 h-5 animate-pulse" />
                                    Listen to the question...
                                </div>
                            )}
                        </div>

                        {/* Progress */}
                        <div className="flex gap-2">
                            {DEMO_QUESTIONS.map((_, idx) => (
                                <div
                                    key={idx}
                                    className={`flex-1 h-2 rounded-full ${idx < currentQuestionIndex
                                        ? 'bg-green-500'
                                        : idx === currentQuestionIndex
                                            ? 'bg-blue-500'
                                            : 'bg-white/20'
                                        }`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

// Loading fallback for Suspense
function SessionLoading() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center">
            <div className="text-center">
                <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                <p className="text-white/60">Loading interview session...</p>
            </div>
        </div>
    )
}

// Main export with Suspense boundary
export default function InterviewSessionPage() {
    return (
        <Suspense fallback={<SessionLoading />}>
            <InterviewSessionContent />
        </Suspense>
    )
}
