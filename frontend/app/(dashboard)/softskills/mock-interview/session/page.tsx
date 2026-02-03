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
    ExclamationTriangleIcon,
    SparklesIcon
} from '@heroicons/react/24/outline'
import { useSpeechEngine, useSpeechRecognition } from '@/components/avatar/SpeechEngine'

// Professional 3D Avatar with TalkingHead.js lip-sync
const TalkingHeadAvatar = dynamic(() => import('@/components/avatar/TalkingHeadAvatar'), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full bg-gradient-to-b from-slate-100 to-slate-200 animate-pulse rounded-2xl flex items-center justify-center">
            <span className="text-gray-400">Loading 3D Avatar...</span>
        </div>
    )
})

// Question interface from backend
interface InterviewQuestion {
    id: string
    question: string
    topic_id: string
    topic_name: string
    difficulty: string
}

// Evaluation result from backend
interface AnswerEvaluation {
    question_id: string
    score: number
    concept_scores: Record<string, number>
    covered_concepts: string[]
    missed_concepts: string[]
    feedback: string
    expected_answer_summary: string
}

type SessionState = 'loading' | 'ready' | 'speaking' | 'listening' | 'processing' | 'complete' | 'error'
type PermissionState = 'pending' | 'granted' | 'denied' | 'error'

function InterviewSessionContent() {
    const searchParams = useSearchParams()
    const router = useRouter()

    // Read correct URL params from mock interview selection page
    // New multi-topic params
    const topicIdsParam = searchParams.get('topics') || ''
    const topicNamesParam = searchParams.get('topic_names') || ''
    // Backward compatibility with single topic
    const legacyTopicId = searchParams.get('topic') || ''
    const legacyTopicName = searchParams.get('topic_name') || ''

    // Parse topic IDs and names
    const topicIds = topicIdsParam ? topicIdsParam.split(',') : (legacyTopicId ? [legacyTopicId] : [])
    const topicNames = topicNamesParam ? topicNamesParam.split(',') : (legacyTopicName ? [legacyTopicName] : ['General Topic'])

    const displayTopicName = topicNames.length > 1
        ? `${topicNames[0]} (+${topicNames.length - 1} more)`
        : topicNames[0] || 'General Topic'

    const classrooms = searchParams.get('classrooms') || 'all'
    const avatarId = (searchParams.get('avatar') || 'female') as 'male' | 'female'

    // Legacy params (fallback for backward compatibility)
    const subject = displayTopicName
    const chapter = displayTopicName

    const [sessionState, setSessionState] = useState<SessionState>('loading')
    const [sessionId, setSessionId] = useState<string>('')
    const [questions, setQuestions] = useState<InterviewQuestion[]>([])
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
    const [answers, setAnswers] = useState<string[]>([])
    const [evaluations, setEvaluations] = useState<AnswerEvaluation[]>([])
    const [isEvaluating, setIsEvaluating] = useState(false)
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [avatarReady, setAvatarReady] = useState(false)
    const [timeElapsed, setTimeElapsed] = useState(0)
    const [permissionState, setPermissionState] = useState<PermissionState>('pending')
    const [permissionError, setPermissionError] = useState<string>('')
    const [textToSpeak, setTextToSpeak] = useState<string>('')
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [loadingError, setLoadingError] = useState<string>('')
    const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false)
    const [manualAnswer, setManualAnswer] = useState<string>('')  // Text input fallback for STT

    const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'


    const videoRef = useRef<HTMLVideoElement>(null)
    const mediaStreamRef = useRef<MediaStream | null>(null)
    const timerRef = useRef<NodeJS.Timeout | null>(null)
    const answerStartTimeRef = useRef<number>(0)
    const isInitializingRef = useRef<boolean>(false)
    const hasInitializedRef = useRef<boolean>(false)


    // Speech function - sets text for TalkingHead avatar to speak with lip sync
    const speak = useCallback((text: string) => {
        console.log('[MockInterview] Triggering TalkingHead speech:', text)
        setTextToSpeak(text)
    }, [])

    const stopSpeaking = useCallback(() => {
        console.log('[MockInterview] Stopping speech')
        setTextToSpeak('')
        setIsSpeaking(false)
    }, [])

    // TalkingHead speech callbacks
    const handleSpeechStart = useCallback(() => {
        console.log('[MockInterview] Avatar started speaking')
        setIsSpeaking(true)
        setSessionState('speaking')
    }, [])

    const handleSpeechEnd = useCallback(() => {
        console.log('[MockInterview] Avatar finished speaking')
        setIsSpeaking(false)
        setTextToSpeak('')
        setSessionState('listening')
        answerStartTimeRef.current = Date.now()
    }, [])


    const {
        isListening,
        transcript,
        startListening,
        stopListening,
        resetTranscript,
        isSupported: sttSupported,
        error: sttError
    } = useSpeechRecognition()

    const currentQuestion = questions[currentQuestionIndex]

    // Initialize session - fetch questions from backend
    useEffect(() => {
        const initSession = async () => {
            // Prevent duplicate calls
            if (isInitializingRef.current || hasInitializedRef.current) {
                console.log('[MockInterview] Skipping duplicate init call')
                return
            }
            isInitializingRef.current = true

            if (topicIds.length === 0) {
                setLoadingError('No topics selected. Please go back and select topics.')
                setSessionState('error')
                isInitializingRef.current = false
                return
            }

            try {
                const token = localStorage.getItem('accessToken')
                if (!token) {
                    setLoadingError('Not authenticated. Please log in.')
                    setSessionState('error')
                    isInitializingRef.current = false
                    return
                }

                console.log('[MockInterview] Starting session with topics:', topicIds)

                const response = await fetch(`${AI_SERVICE_URL}/api/mock-interview/start-topic-interview`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        user_id: 'current-user', // Will be extracted from token on backend
                        topic_ids: topicIds,
                        avatar: avatarId,
                        questions_per_topic: 3,
                        token: token
                    })
                })

                if (response.status === 404) {
                    // No questions available - offer to generate
                    setIsGeneratingQuestions(true)
                    setLoadingError('No questions available for these topics yet. Would you like to generate them?')
                    setSessionState('error')
                    return
                }

                if (!response.ok) {
                    throw new Error(`Failed to start session: ${response.status}`)
                }

                const data = await response.json()
                console.log('[MockInterview] Session started:', data)

                setSessionId(data.session_id)
                // Build full questions array from batch response
                // The backend returns questions one at a time, but we need them all for progress display
                // For now, we'll fetch them progressively
                setQuestions([{
                    id: data.question.id,
                    question: data.question.question,
                    topic_id: data.question.topic_id,
                    topic_name: data.question.topic_name,
                    difficulty: data.question.difficulty
                }])
                setSessionState('ready')
                hasInitializedRef.current = true
                isInitializingRef.current = false

            } catch (error) {
                console.error('[MockInterview] Session init error:', error)
                setLoadingError('Failed to load interview questions. Please try again.')
                setSessionState('error')
                isInitializingRef.current = false
            }
        }

        initSession()
    }, [topicIds, avatarId, AI_SERVICE_URL])

    // Check permissions on mount
    useEffect(() => {
        checkPermissions()
    }, [])

    // Debug: Log STT support status
    useEffect(() => {
        console.log('[MockInterview] STT Supported:', sttSupported)
        console.log('[MockInterview] STT Error:', sttError)
        if (!sttSupported) {
            console.warn('[MockInterview] Speech recognition is NOT supported in this browser. User can use text input instead.')
        }
    }, [sttSupported, sttError])

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
        if (currentQuestion) {
            setSessionState('speaking')
            await speak(currentQuestion.question)
        }
    }, [startCamera, speak, currentQuestion])

    // Submit answer using new API
    const submitAnswer = useCallback(async () => {
        console.log('[MockInterview] Submit answer called')
        console.log('[MockInterview] Current transcript:', transcript)
        console.log('[MockInterview] Manual answer:', manualAnswer)

        stopListening()
        setIsEvaluating(true)
        setSessionState('processing')

        // Use transcript from speech or manual text input as fallback
        const answer = transcript.trim() || manualAnswer.trim()
        setAnswers(prev => [...prev, answer])
        setManualAnswer('')  // Clear manual input
        console.log('[MockInterview] Answer saved:', answer)

        const responseTime = Math.round((Date.now() - answerStartTimeRef.current) / 1000)

        try {
            const token = localStorage.getItem('accessToken')

            const response = await fetch(`${AI_SERVICE_URL}/api/mock-interview/submit-topic-answer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    question_id: currentQuestion.id,
                    answer_text: answer,
                    response_time_seconds: responseTime,
                    token: token
                })
            })

            if (!response.ok) {
                throw new Error(`Evaluation failed: ${response.status}`)
            }

            const data = await response.json()
            console.log('[MockInterview] Evaluation result:', data)

            // Store evaluation
            setEvaluations(prev => [...prev, data.evaluation])
            setIsEvaluating(false)
            resetTranscript()

            // Check if more questions
            if (!data.is_complete && data.next_question) {
                console.log('[MockInterview] Moving to next question')
                setQuestions(prev => [...prev, data.next_question])
                setCurrentQuestionIndex(prev => prev + 1)
                setSessionState('speaking')
                await speak(data.next_question.question)
            } else {
                // Interview complete
                console.log('[MockInterview] Interview complete!')
                setSessionState('complete')
                stopCamera()
                if (timerRef.current) {
                    clearInterval(timerRef.current)
                }
            }

        } catch (error) {
            console.error('[MockInterview] Evaluation failed:', error)
            // Fallback - still allow progress
            setIsEvaluating(false)
            resetTranscript()

            // Create fallback evaluation
            const wordCount = answer.split(' ').length
            const fallbackScore = Math.min(100, 40 + wordCount * 2)
            setEvaluations(prev => [...prev, {
                question_id: currentQuestion.id,
                score: fallbackScore,
                concept_scores: {},
                covered_concepts: [],
                missed_concepts: [],
                feedback: 'Answer recorded. Keep practicing for better results!',
                expected_answer_summary: ''
            }])

            if (currentQuestionIndex < questions.length - 1) {
                setCurrentQuestionIndex(prev => prev + 1)
                setSessionState('speaking')
                await speak(questions[currentQuestionIndex + 1].question)
            } else {
                setSessionState('complete')
                stopCamera()
                if (timerRef.current) {
                    clearInterval(timerRef.current)
                }
            }
        }
    }, [transcript, currentQuestion, sessionId, currentQuestionIndex, questions, stopListening, resetTranscript, speak, stopCamera, AI_SERVICE_URL])

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

    // Calculate average score from evaluations
    const averageScore = evaluations.length > 0
        ? Math.round(evaluations.reduce((sum, e) => sum + e.score, 0) / evaluations.length)
        : 0

    // Loading state
    if (sessionState === 'loading') {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-white/60">Loading interview questions...</p>
                </div>
            </div>
        )
    }

    // Error state
    if (sessionState === 'error') {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
                <div className="max-w-md mx-auto">
                    <div className="bg-white rounded-3xl shadow-xl p-8 text-center">
                        <div className="w-20 h-20 rounded-full bg-amber-100 flex items-center justify-center mx-auto mb-6">
                            <ExclamationTriangleIcon className="w-10 h-10 text-amber-600" />
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 mb-2">Unable to Start</h1>
                        <p className="text-gray-500 mb-6">{loadingError}</p>

                        {isGeneratingQuestions && (
                            <button
                                onClick={async () => {
                                    // TODO: Trigger question generation
                                    setIsGeneratingQuestions(false)
                                    setSessionState('loading')
                                    // Retry init
                                }}
                                className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-500 to-indigo-600 font-medium text-white hover:shadow-lg transition-all mb-3 flex items-center justify-center gap-2"
                            >
                                <SparklesIcon className="w-5 h-5" />
                                Generate Questions
                            </button>
                        )}

                        <Link
                            href="/softskills/mock-interview"
                            className="block w-full py-3 rounded-xl border border-gray-200 font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                        >
                            Go Back
                        </Link>
                    </div>
                </div>
            </div>
        )
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
                                <p className="text-2xl font-bold text-gray-900">{evaluations.length}</p>
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
                            Question {currentQuestionIndex + 1} of {questions.length || '...'}
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-6xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Avatar Section */}
                    <div className="space-y-4">
                        <div className="aspect-[3/4] max-h-[600px] rounded-2xl overflow-hidden shadow-2xl relative">
                            <TalkingHeadAvatar
                                avatarId={avatarId}
                                isSpeaking={isSpeaking}
                                textToSpeak={textToSpeak}
                                onReady={() => setAvatarReady(true)}
                                onSpeechStart={handleSpeechStart}
                                onSpeechEnd={handleSpeechEnd}
                            />
                        </div>
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
                            <p className="text-sm text-white/60 mb-2">Current Question{currentQuestion?.topic_name ? ` (${currentQuestion.topic_name})` : ''}:</p>
                            <p className="text-lg font-medium">{currentQuestion?.question || 'Loading...'}</p>
                        </div>

                        {/* STT Warning */}
                        {!sttSupported && (
                            <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-xl p-4">
                                <p className="text-yellow-200 text-sm font-medium">⚠️ Speech Recognition Not Supported</p>
                                <p className="text-yellow-200/70 text-xs mt-1">
                                    Your browser doesn't support speech recognition. Please use Chrome or Edge for the best experience,
                                    or type your answer in the text box below.
                                </p>
                            </div>
                        )}

                        {/* STT Error */}
                        {sttError && sttError !== 'no-speech' && (
                            <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4">
                                <p className="text-red-200 text-sm font-medium">🎤 Microphone Error: {sttError}</p>
                                <p className="text-red-200/70 text-xs mt-1">
                                    {sttError === 'not-allowed'
                                        ? 'Microphone permission denied. Please allow microphone access and refresh the page.'
                                        : 'Please check your microphone settings and try again.'}
                                </p>
                            </div>
                        )}

                        {/* Transcript Display */}
                        {(isListening || transcript) && (
                            <div className="bg-white/5 rounded-2xl p-6">
                                <p className="text-sm text-white/60 mb-2">Your Answer:</p>
                                <p className="text-white/90">
                                    {transcript || <span className="text-white/40 italic">Listening...</span>}
                                </p>
                            </div>
                        )}

                        {/* Text Input Fallback (when STT not available or user prefers typing) */}
                        {sessionState === 'listening' && (
                            <div className="bg-white/5 rounded-2xl p-6 space-y-4">
                                <p className="text-sm text-white/60 mb-2">
                                    {sttSupported ? 'Or type your answer:' : 'Type your answer:'}
                                </p>
                                <textarea
                                    value={manualAnswer}
                                    onChange={(e) => setManualAnswer(e.target.value)}
                                    placeholder="Type your answer here..."
                                    className="w-full h-32 bg-white/10 backdrop-blur-sm rounded-xl p-4 text-white placeholder-white/40 border border-white/10 focus:border-blue-500 focus:outline-none resize-none"
                                />
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
                                        disabled={!transcript.trim() && !manualAnswer.trim()}
                                        className="flex-1 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all disabled:opacity-50"
                                    >
                                        <CheckCircleIcon className="w-5 h-5" />
                                        Submit Answer
                                    </button>
                                </>
                            )}

                            {/* Submit button for text-only input (when not listening but has typed answer) */}
                            {sessionState === 'listening' && !isListening && manualAnswer.trim() && (
                                <button
                                    onClick={submitAnswer}
                                    className="flex-1 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 font-semibold flex items-center justify-center gap-2 hover:shadow-lg transition-all"
                                >
                                    <CheckCircleIcon className="w-5 h-5" />
                                    Submit Typed Answer
                                </button>
                            )}

                            {sessionState === 'speaking' && (
                                <div className="flex-1 py-4 rounded-xl bg-white/10 font-medium flex items-center justify-center gap-2">
                                    <SpeakerWaveIcon className="w-5 h-5 animate-pulse" />
                                    Listen to the question...
                                </div>
                            )}

                            {sessionState === 'processing' && (
                                <div className="flex-1 py-4 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 font-medium flex items-center justify-center gap-2">
                                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    Evaluating your answer...
                                </div>
                            )}
                        </div>


                        {/* Progress */}
                        <div className="flex gap-2">
                            {questions.map((_, idx) => (
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
        </div >
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
