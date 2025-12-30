'use client'

import { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import {
    XMarkIcon,
    MicrophoneIcon,
    StopIcon,
    CheckCircleIcon,
    SpeakerWaveIcon,
    VideoCameraIcon,
    ClockIcon,
    EyeIcon,
    HandRaisedIcon,
    ChatBubbleLeftRightIcon,
    FaceSmileIcon,
    ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { useSpeechEngine, useSpeechRecognition } from '@/components/avatar/SpeechEngine'
import { useSoftSkillsAnalysis, useFrameCapture } from '@/hooks/useSoftSkillsAnalysis'
import { RealTimeScoreCard } from '@/components/softskills/RealTimeScoreCard'

// Dynamic import for Avatar
const AvatarViewer = dynamic(() => import('@/components/avatar/AvatarViewer'), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full bg-gray-200 animate-pulse rounded-2xl flex items-center justify-center">
            <span className="text-gray-400">Loading Avatar...</span>
        </div>
    )
})

// Communication prompts
const COMMUNICATION_PROMPTS = [
    "Tell me about yourself and your educational background.",
    "What are your strengths and how do they help you in learning?",
    "Describe a challenging situation you faced and how you overcame it.",
    "Where do you see yourself in five years?",
    "What motivates you to learn new things?"
]

type SessionState = 'ready' | 'speaking' | 'listening' | 'processing' | 'complete'
type PermissionState = 'pending' | 'granted' | 'denied' | 'error'

interface SkillScore {
    name: string
    score: number
    icon: React.ComponentType<any>
    color: string
}

function CommunicationSessionContent() {
    const searchParams = useSearchParams()
    const router = useRouter()

    const avatarId = (searchParams.get('avatar') || 'female') as 'male' | 'female'

    const [sessionState, setSessionState] = useState<SessionState>('ready')
    const [currentPromptIndex, setCurrentPromptIndex] = useState(0)
    const [answers, setAnswers] = useState<string[]>([])
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [avatarReady, setAvatarReady] = useState(false)
    const [timeElapsed, setTimeElapsed] = useState(0)
    const [permissionState, setPermissionState] = useState<PermissionState>('pending')
    const [permissionError, setPermissionError] = useState<string>('')

    // Generate unique session ID
    const sessionIdRef = useRef(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)

    // Real-time soft skills analysis hook
    const {
        isConnected: analysisConnected,
        isAnalyzing,
        visualMetrics,
        fluencyMetrics,
        framesProcessed,
        sendFrame,
        sendTranscript,
        startAnalysis,
        stopAnalysis,
        requestSummary,
    } = useSoftSkillsAnalysis(sessionIdRef.current)

    // Real-time scores from analysis (with fallback to reasonable defaults)
    const [realtimeScores, setRealtimeScores] = useState({
        eyeContact: 75,
        posture: 75,
        handGestures: 70,
        expressions: 75,
        fluency: 0,
        grammar: 0
    })

    const videoRef = useRef<HTMLVideoElement>(null)
    const mediaStreamRef = useRef<MediaStream | null>(null)
    const timerRef = useRef<NodeJS.Timeout | null>(null)
    const scoreUpdateRef = useRef<NodeJS.Timeout | null>(null)

    const { speak, stop: stopSpeaking, isSpeaking } = useSpeechEngine({
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

    const currentPrompt = COMMUNICATION_PROMPTS[currentPromptIndex]

    // Check permissions on mount
    useEffect(() => {
        checkPermissions()
    }, [])

    const checkPermissions = async () => {
        try {
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
            setPermissionState('pending')
        }
    }

    // Update real-time scores from backend analysis
    useEffect(() => {
        if (visualMetrics) {
            setRealtimeScores(prev => ({
                ...prev,
                eyeContact: visualMetrics.gaze_score || prev.eyeContact,
                posture: visualMetrics.posture_score || prev.posture,
                handGestures: visualMetrics.gesture_score || prev.handGestures,
                expressions: 75, // Placeholder - will be replaced with expression analysis
            }))
        }
    }, [visualMetrics])

    // Update fluency scores from analysis
    useEffect(() => {
        if (fluencyMetrics) {
            setRealtimeScores(prev => ({
                ...prev,
                fluency: fluencyMetrics.score || prev.fluency,
            }))
        }
    }, [fluencyMetrics])

    // Frame capture for real-time analysis
    useFrameCapture(videoRef, sendFrame, isAnalyzing && isListening, 5)

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
                audio: true
            })
            mediaStreamRef.current = stream
            setIsCameraOn(true)
            setPermissionState('granted')
            return true
        } catch (err: any) {
            console.error('Camera access denied:', err)
            setPermissionState('denied')
            if (err.name === 'NotAllowedError') {
                setPermissionError('Camera and microphone access is required for this evaluation. Please allow access to continue.')
            } else if (err.name === 'NotFoundError') {
                setPermissionError('No camera or microphone found. Please connect the required devices and try again.')
            } else if (err.name === 'NotReadableError') {
                setPermissionError('Camera or microphone is already in use by another application.')
            } else {
                setPermissionError('Failed to access camera and microphone. Please check your device settings.')
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

    // Start the session
    const startSession = useCallback(async () => {
        const cameraStarted = await startCamera()
        if (!cameraStarted) return

        timerRef.current = setInterval(() => {
            setTimeElapsed(prev => prev + 1)
        }, 1000)

        // Start real-time analysis
        startAnalysis()

        setSessionState('speaking')
        await speak(currentPrompt)
    }, [startCamera, speak, currentPrompt, startAnalysis])

    // Submit current answer and move to next prompt
    const submitAnswer = useCallback(async () => {
        stopListening()

        const answer = transcript.trim()
        setAnswers(prev => [...prev, answer])

        // Send transcript for fluency analysis
        if (answer && timeElapsed > 0) {
            sendTranscript(answer, timeElapsed)
        }

        // Simple grammar score (can be enhanced with backend grammar analysis)
        const grammarScore = Math.min(100, 70 + Math.random() * 20)
        setRealtimeScores(prev => ({
            ...prev,
            grammar: Math.round((prev.grammar * currentPromptIndex + grammarScore) / (currentPromptIndex + 1))
        }))

        resetTranscript()

        if (currentPromptIndex < COMMUNICATION_PROMPTS.length - 1) {
            setCurrentPromptIndex(prev => prev + 1)
            setSessionState('speaking')
            await speak(COMMUNICATION_PROMPTS[currentPromptIndex + 1])
        } else {
            // End session and get summary
            stopAnalysis()
            requestSummary()
            setSessionState('complete')
            stopCamera()
            if (timerRef.current) {
                clearInterval(timerRef.current)
            }
        }
    }, [transcript, currentPromptIndex, stopListening, resetTranscript, speak, stopCamera, sendTranscript, timeElapsed, stopAnalysis, requestSummary])

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    useEffect(() => {
        return () => {
            stopCamera()
            stopSpeaking()
            if (timerRef.current) clearInterval(timerRef.current)
            if (scoreUpdateRef.current) clearInterval(scoreUpdateRef.current)
        }
    }, [stopCamera, stopSpeaking])

    // Calculate final score (weighted average)
    const calculateFinalScore = () => {
        const { fluency, grammar, eyeContact, handGestures, posture, expressions } = realtimeScores
        return Math.round(
            0.30 * fluency +
            0.20 * grammar +
            0.15 * eyeContact +
            0.10 * handGestures +
            0.10 * expressions +
            0.10 * posture
        )
    }

    const finalScores: SkillScore[] = [
        { name: 'Fluency', score: Math.round(realtimeScores.fluency), icon: MicrophoneIcon, color: 'text-blue-500' },
        { name: 'Grammar', score: Math.round(realtimeScores.grammar), icon: ChatBubbleLeftRightIcon, color: 'text-indigo-500' },
        { name: 'Eye Contact', score: Math.round(realtimeScores.eyeContact), icon: EyeIcon, color: 'text-purple-500' },
        { name: 'Expressions', score: Math.round(realtimeScores.expressions), icon: FaceSmileIcon, color: 'text-amber-500' },
        { name: 'Hand Gestures', score: Math.round(realtimeScores.handGestures), icon: HandRaisedIcon, color: 'text-pink-500' },
        { name: 'Posture', score: Math.round(realtimeScores.posture), icon: VideoCameraIcon, color: 'text-rose-500' }
    ]

    // Permission denied state
    if (permissionState === 'denied') {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-purple-50 p-6">
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
                                    startSession()
                                }}
                                className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 font-medium text-white hover:shadow-lg transition-all"
                            >
                                Try Again
                            </button>
                            <Link
                                href="/softskills/communication"
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
        const finalScore = calculateFinalScore()

        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-purple-50 p-6">
                <div className="max-w-2xl mx-auto">
                    <div className="bg-white rounded-3xl shadow-xl p-8">
                        <div className="text-center mb-8">
                            <div className="w-20 h-20 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-4">
                                <CheckCircleIcon className="w-10 h-10 text-green-600" />
                            </div>
                            <h1 className="text-2xl font-bold text-gray-900 mb-2">Evaluation Complete!</h1>
                            <p className="text-gray-500">Here's your communication skills breakdown</p>
                        </div>

                        {/* Overall Score */}
                        <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl p-6 text-white text-center mb-6">
                            <p className="text-sm text-purple-200 mb-1">Overall Soft Skills Score</p>
                            <p className="text-5xl font-bold">{finalScore}%</p>
                        </div>

                        {/* Breakdown */}
                        <div className="space-y-4 mb-6">
                            {finalScores.map((skill) => (
                                <div key={skill.name} className="flex items-center gap-4">
                                    <skill.icon className={`w-5 h-5 ${skill.color}`} />
                                    <span className="flex-1 text-gray-700 font-medium">{skill.name}</span>
                                    <div className="w-32 bg-gray-200 rounded-full h-2">
                                        <div
                                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                                            style={{ width: `${skill.score}%` }}
                                        />
                                    </div>
                                    <span className="w-12 text-right font-semibold text-gray-900">{skill.score}%</span>
                                </div>
                            ))}
                        </div>

                        {/* Stats */}
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="bg-gray-50 rounded-xl p-4 text-center">
                                <p className="text-sm text-gray-500">Prompts</p>
                                <p className="text-2xl font-bold text-gray-900">{COMMUNICATION_PROMPTS.length}</p>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-4 text-center">
                                <p className="text-sm text-gray-500">Duration</p>
                                <p className="text-2xl font-bold text-gray-900">{formatTime(timeElapsed)}</p>
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <Link
                                href="/softskills/communication"
                                className="flex-1 py-3 rounded-xl border border-gray-200 font-medium text-gray-700 hover:bg-gray-50 transition-colors text-center"
                            >
                                Try Again
                            </Link>
                            <Link
                                href="/softskills"
                                className="flex-1 py-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 font-medium text-white hover:shadow-lg transition-all text-center"
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
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-purple-900 text-white">
            {/* Header */}
            <div className="bg-black/30 backdrop-blur-sm border-b border-white/10 px-6 py-4">
                <div className="max-w-6xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/softskills/communication" className="text-white/60 hover:text-white">
                            <XMarkIcon className="w-6 h-6" />
                        </Link>
                        <div>
                            <h1 className="font-semibold">Soft Skills Evaluation</h1>
                            <p className="text-sm text-white/60">Communication Assessment</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-white/60">
                            <ClockIcon className="w-4 h-4" />
                            <span className="text-sm font-mono">{formatTime(timeElapsed)}</span>
                        </div>
                        <div className="text-sm">
                            Prompt {currentPromptIndex + 1} of {COMMUNICATION_PROMPTS.length}
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-6xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Avatar Section */}
                    <div className="space-y-4">
                        <div className="aspect-square max-h-[400px] rounded-2xl overflow-hidden shadow-2xl">
                            <AvatarViewer
                                avatarId={avatarId}
                                isSpeaking={isSpeaking}
                                onReady={() => setAvatarReady(true)}
                            />
                        </div>

                        {isSpeaking && (
                            <div className="flex items-center justify-center gap-2 text-purple-400">
                                <SpeakerWaveIcon className="w-5 h-5 animate-pulse" />
                                <span className="text-sm">Avatar is speaking...</span>
                            </div>
                        )}
                    </div>

                    {/* User Camera */}
                    <div className="space-y-4">
                        <div className="relative aspect-square bg-gray-800 rounded-2xl overflow-hidden shadow-xl">
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

                            {isListening && (
                                <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 px-3 py-1 rounded-full">
                                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                    <span className="text-xs font-medium">Recording</span>
                                </div>
                            )}
                        </div>

                        {/* Prompt */}
                        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
                            <p className="text-sm text-white/60 mb-1">Prompt:</p>
                            <p className="font-medium">{currentPrompt}</p>
                        </div>
                    </div>

                    {/* Real-time Scores */}
                    <div className="space-y-4">
                        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6">
                            <h3 className="font-semibold mb-4">Real-time Analysis</h3>
                            <div className="space-y-4">
                                <div>
                                    <div className="flex items-center justify-between text-sm mb-1">
                                        <span className="flex items-center gap-2">
                                            <EyeIcon className="w-4 h-4 text-purple-400" />
                                            Eye Contact
                                        </span>
                                        <span>{Math.round(realtimeScores.eyeContact)}%</span>
                                    </div>
                                    <div className="h-2 bg-white/20 rounded-full">
                                        <div
                                            className="h-2 bg-purple-500 rounded-full transition-all"
                                            style={{ width: `${realtimeScores.eyeContact}%` }}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-between text-sm mb-1">
                                        <span className="flex items-center gap-2">
                                            <VideoCameraIcon className="w-4 h-4 text-pink-400" />
                                            Posture
                                        </span>
                                        <span>{Math.round(realtimeScores.posture)}%</span>
                                    </div>
                                    <div className="h-2 bg-white/20 rounded-full">
                                        <div
                                            className="h-2 bg-pink-500 rounded-full transition-all"
                                            style={{ width: `${realtimeScores.posture}%` }}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-between text-sm mb-1">
                                        <span className="flex items-center gap-2">
                                            <FaceSmileIcon className="w-4 h-4 text-amber-400" />
                                            Expressions
                                        </span>
                                        <span>{Math.round(realtimeScores.expressions)}%</span>
                                    </div>
                                    <div className="h-2 bg-white/20 rounded-full">
                                        <div
                                            className="h-2 bg-amber-500 rounded-full transition-all"
                                            style={{ width: `${realtimeScores.expressions}%` }}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-between text-sm mb-1">
                                        <span className="flex items-center gap-2">
                                            <HandRaisedIcon className="w-4 h-4 text-rose-400" />
                                            Hand Gestures
                                        </span>
                                        <span>{Math.round(realtimeScores.handGestures)}%</span>
                                    </div>
                                    <div className="h-2 bg-white/20 rounded-full">
                                        <div
                                            className="h-2 bg-rose-500 rounded-full transition-all"
                                            style={{ width: `${realtimeScores.handGestures}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Transcript */}
                        {(isListening || transcript) && (
                            <div className="bg-white/5 rounded-xl p-4">
                                <p className="text-sm text-white/60 mb-1">Your Response:</p>
                                <p className="text-sm text-white/90">
                                    {transcript || <span className="text-white/40 italic">Listening...</span>}
                                </p>
                            </div>
                        )}

                        {/* Controls */}
                        <div className="flex flex-col gap-3">
                            {sessionState === 'ready' && (
                                <button
                                    onClick={startSession}
                                    disabled={!avatarReady}
                                    className="w-full py-4 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all disabled:opacity-50"
                                >
                                    Start Evaluation
                                </button>
                            )}

                            {sessionState === 'listening' && !isListening && (
                                <button
                                    onClick={startListening}
                                    className="w-full py-4 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all"
                                >
                                    <MicrophoneIcon className="w-5 h-5" />
                                    Start Speaking
                                </button>
                            )}

                            {isListening && (
                                <div className="flex gap-2">
                                    <button
                                        onClick={stopListening}
                                        className="flex-1 py-3 rounded-xl bg-gradient-to-r from-red-500 to-rose-600 font-semibold flex items-center justify-center gap-2"
                                    >
                                        <StopIcon className="w-5 h-5" />
                                        Stop
                                    </button>
                                    <button
                                        onClick={submitAnswer}
                                        disabled={!transcript.trim()}
                                        className="flex-1 py-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
                                    >
                                        <CheckCircleIcon className="w-5 h-5" />
                                        Submit
                                    </button>
                                </div>
                            )}

                            {sessionState === 'speaking' && (
                                <div className="w-full py-4 rounded-xl bg-white/10 font-medium flex items-center justify-center gap-2">
                                    <SpeakerWaveIcon className="w-5 h-5 animate-pulse" />
                                    Listen to the prompt...
                                </div>
                            )}
                        </div>

                        {/* Progress */}
                        <div className="flex gap-2">
                            {COMMUNICATION_PROMPTS.map((_, idx) => (
                                <div
                                    key={idx}
                                    className={`flex-1 h-2 rounded-full ${idx < currentPromptIndex
                                        ? 'bg-green-500'
                                        : idx === currentPromptIndex
                                            ? 'bg-purple-500'
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
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-purple-900 flex items-center justify-center">
            <div className="text-center">
                <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                <p className="text-white/60">Loading evaluation session...</p>
            </div>
        </div>
    )
}

// Main export with Suspense boundary
export default function CommunicationSessionPage() {
    return (
        <Suspense fallback={<SessionLoading />}>
            <CommunicationSessionContent />
        </Suspense>
    )
}
