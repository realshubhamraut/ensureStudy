'use client'

import { useState, useCallback, useRef, useEffect } from 'react'

interface SpeechEngineProps {
    onSpeakStart?: () => void
    onSpeakEnd?: () => void
    onPhoneme?: (phoneme: string) => void
}

interface SpeechEngineReturn {
    speak: (text: string) => Promise<void>
    stop: () => void
    isSpeaking: boolean
    isSupported: boolean
}

// Phoneme to viseme mapping for lip sync
const PHONEME_TO_VISEME: Record<string, string> = {
    // Vowels
    'a': 'viseme_aa',
    'e': 'viseme_ee',
    'i': 'viseme_ih',
    'o': 'viseme_oh',
    'u': 'viseme_ou',
    // Consonants
    'p': 'viseme_pp',
    'b': 'viseme_pp',
    'm': 'viseme_pp',
    'f': 'viseme_ff',
    'v': 'viseme_ff',
    't': 'viseme_tt',
    'd': 'viseme_dd',
    's': 'viseme_ss',
    'z': 'viseme_ss',
    'n': 'viseme_nn',
    'l': 'viseme_nn',
    'r': 'viseme_rr',
    'k': 'viseme_kk',
    'g': 'viseme_kk',
    'ch': 'viseme_ch',
    'sh': 'viseme_ch',
    'th': 'viseme_th',
    // Default
    ' ': 'viseme_sil'
}

/**
 * Custom hook for Web Speech API text-to-speech with phoneme callbacks
 */
export function useSpeechEngine({ onSpeakStart, onSpeakEnd, onPhoneme }: SpeechEngineProps = {}): SpeechEngineReturn {
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [isSupported, setIsSupported] = useState(false)
    const [voicesLoaded, setVoicesLoaded] = useState(false)
    const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
    const phonemeIntervalRef = useRef<NodeJS.Timeout | null>(null)

    // Check support and load voices
    useEffect(() => {
        if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
            setIsSupported(false)
            return
        }

        setIsSupported(true)

        // Voices may load asynchronously
        const loadVoices = () => {
            const voices = window.speechSynthesis.getVoices()
            if (voices.length > 0) {
                setVoicesLoaded(true)
                console.log('Loaded', voices.length, 'voices')
            }
        }

        // Try loading immediately
        loadVoices()

        // Also listen for voice changes (Chrome loads voices async)
        window.speechSynthesis.onvoiceschanged = loadVoices

        // Workaround: Chrome sometimes needs a dummy call to init
        const initSpeech = () => {
            const utterance = new SpeechSynthesisUtterance('')
            utterance.volume = 0
            window.speechSynthesis.speak(utterance)
            window.speechSynthesis.cancel()
        }

        // Try after a short delay
        const timer = setTimeout(() => {
            initSpeech()
            loadVoices()
        }, 100)

        return () => {
            clearTimeout(timer)
            window.speechSynthesis.onvoiceschanged = null
        }
    }, [])

    // Simple phoneme simulation based on text
    const simulatePhonemes = useCallback((text: string, duration: number) => {
        const chars = text.toLowerCase().split('')
        const charDuration = duration / chars.length
        let index = 0

        phonemeIntervalRef.current = setInterval(() => {
            if (index < chars.length) {
                const char = chars[index]
                const viseme = PHONEME_TO_VISEME[char] || 'viseme_sil'
                onPhoneme?.(viseme)
                index++
            } else {
                if (phonemeIntervalRef.current) {
                    clearInterval(phonemeIntervalRef.current)
                }
            }
        }, charDuration * 1000)
    }, [onPhoneme])

    const speak = useCallback(async (text: string): Promise<void> => {
        if (!isSupported) {
            console.warn('Speech synthesis not supported')
            // Still trigger callbacks for UI flow
            onSpeakStart?.()
            await new Promise(resolve => setTimeout(resolve, 2000))
            onSpeakEnd?.()
            return Promise.resolve()
        }

        // Cancel any current speech
        window.speechSynthesis.cancel()

        // Small delay to ensure cancel completes
        await new Promise(resolve => setTimeout(resolve, 50))

        return new Promise((resolve) => {
            const utterance = new SpeechSynthesisUtterance(text)
            utteranceRef.current = utterance

            // Get voices
            const voices = window.speechSynthesis.getVoices()

            if (voices.length === 0) {
                console.warn('No voices available, using default')
            } else {
                // Prefer Google or Microsoft voices for better quality
                const preferredVoice = voices.find(v =>
                    v.lang.startsWith('en') && (
                        v.name.includes('Google') ||
                        v.name.includes('Microsoft') ||
                        v.name.includes('Natural') ||
                        v.name.includes('Samantha') ||
                        v.name.includes('Daniel')
                    )
                ) || voices.find(v => v.lang.startsWith('en')) || voices[0]

                if (preferredVoice) {
                    utterance.voice = preferredVoice
                    console.log('Using voice:', preferredVoice.name)
                }
            }

            utterance.rate = 0.9  // Slightly slower for clarity
            utterance.pitch = 1.0
            utterance.volume = 1.0

            // Estimate duration (rough approximation)
            const wordsPerMinute = 150
            const words = text.split(' ').length
            const estimatedDuration = (words / wordsPerMinute) * 60

            utterance.onstart = () => {
                console.log('TTS started')
                setIsSpeaking(true)
                onSpeakStart?.()
                simulatePhonemes(text, estimatedDuration)
            }

            utterance.onend = () => {
                console.log('TTS ended')
                setIsSpeaking(false)
                onSpeakEnd?.()
                if (phonemeIntervalRef.current) {
                    clearInterval(phonemeIntervalRef.current)
                }
                resolve()
            }

            utterance.onerror = (event) => {
                console.error('Speech synthesis error:', event.error)
                setIsSpeaking(false)
                onSpeakEnd?.()
                if (phonemeIntervalRef.current) {
                    clearInterval(phonemeIntervalRef.current)
                }
                resolve()
            }

            // Speak with a small delay to ensure everything is ready
            setTimeout(() => {
                window.speechSynthesis.speak(utterance)
            }, 100)
        })
    }, [isSupported, simulatePhonemes, onSpeakStart, onSpeakEnd])

    const stop = useCallback(() => {
        if (isSupported) {
            window.speechSynthesis.cancel()
            setIsSpeaking(false)
            if (phonemeIntervalRef.current) {
                clearInterval(phonemeIntervalRef.current)
            }
        }
    }, [isSupported])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (phonemeIntervalRef.current) {
                clearInterval(phonemeIntervalRef.current)
            }
            if (isSupported) {
                window.speechSynthesis.cancel()
            }
        }
    }, [isSupported])

    return { speak, stop, isSpeaking, isSupported }
}

/**
 * Hook for speech recognition (listening to user)
 */
export function useSpeechRecognition() {
    const [isListening, setIsListening] = useState(false)
    const [transcript, setTranscript] = useState('')
    const [interimTranscript, setInterimTranscript] = useState('')
    const [isSupported, setIsSupported] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const recognitionRef = useRef<any>(null)
    const shouldListenRef = useRef(false)
    const restartAttemptsRef = useRef(0)
    const maxRestartAttempts = 3
    const hasNetworkErrorRef = useRef(false)

    useEffect(() => {
        if (typeof window === 'undefined') {
            return
        }

        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition

        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported in this browser')
            setIsSupported(false)
            return
        }

        setIsSupported(true)

        const recognition = new SpeechRecognition()
        recognitionRef.current = recognition

        // Configure recognition
        recognition.continuous = true
        recognition.interimResults = true
        recognition.lang = 'en-US'
        recognition.maxAlternatives = 1

        recognition.onstart = () => {
            console.log('Speech recognition started')
            setIsListening(true)
            setError(null)
        }

        recognition.onresult = (event: any) => {
            let finalTranscript = ''
            let interim = ''

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i]
                const transcriptPart = result[0].transcript

                if (result.isFinal) {
                    finalTranscript += transcriptPart
                    console.log('Final transcript:', transcriptPart)
                } else {
                    interim += transcriptPart
                }
            }

            // Update interim transcript for real-time display
            setInterimTranscript(interim)

            // Append final transcript
            if (finalTranscript) {
                setTranscript(prev => {
                    const newTranscript = prev ? prev + ' ' + finalTranscript : finalTranscript
                    return newTranscript.trim()
                })
                setInterimTranscript('')
            }
        }

        recognition.onerror = (event: any) => {
            console.error('Speech recognition error:', event.error)
            setError(event.error)

            // Don't stop for no-speech error, just continue
            if (event.error === 'no-speech') {
                console.log('No speech detected, continuing to listen...')
                return
            }

            // Track network errors - don't auto-restart on these
            if (event.error === 'network') {
                console.log('Network error - stopping auto-restart (Web Speech API needs internet)')
                hasNetworkErrorRef.current = true
                shouldListenRef.current = false
                setIsListening(false)
                return
            }

            // For permission errors, stop completely
            if (event.error === 'not-allowed') {
                setIsListening(false)
                shouldListenRef.current = false
            }
        }

        recognition.onend = () => {
            console.log('Speech recognition ended')

            // Don't restart if we had network errors
            if (hasNetworkErrorRef.current) {
                console.log('Not restarting due to previous network error')
                setIsListening(false)
                return
            }

            // Auto-restart if we should still be listening (with limits)
            if (shouldListenRef.current && restartAttemptsRef.current < maxRestartAttempts) {
                restartAttemptsRef.current++
                const delay = Math.min(100 * Math.pow(2, restartAttemptsRef.current), 2000) // Exponential backoff
                console.log(`Auto-restarting speech recognition (attempt ${restartAttemptsRef.current}/${maxRestartAttempts})...`)
                try {
                    setTimeout(() => {
                        if (shouldListenRef.current && recognitionRef.current && !hasNetworkErrorRef.current) {
                            recognitionRef.current.start()
                        }
                    }, delay)
                } catch (e) {
                    console.error('Failed to restart recognition:', e)
                    setIsListening(false)
                    shouldListenRef.current = false
                }
            } else if (restartAttemptsRef.current >= maxRestartAttempts) {
                console.log('Max restart attempts reached, stopping')
                setIsListening(false)
                shouldListenRef.current = false
            } else {
                setIsListening(false)
            }
        }

        return () => {
            shouldListenRef.current = false
            if (recognitionRef.current) {
                try {
                    recognitionRef.current.stop()
                } catch (e) {
                    // Ignore errors when stopping
                }
            }
        }
    }, [])

    const startListening = useCallback(() => {
        if (!recognitionRef.current) {
            console.error('Speech recognition not initialized')
            return
        }

        if (isListening) {
            console.log('Already listening')
            return
        }

        try {
            setTranscript('')
            setInterimTranscript('')
            setError(null)
            restartAttemptsRef.current = 0  // Reset restart counter
            hasNetworkErrorRef.current = false  // Reset network error flag
            shouldListenRef.current = true
            recognitionRef.current.start()
            console.log('Started listening')
        } catch (e: any) {
            console.error('Failed to start speech recognition:', e)
            setError(e.message)
            shouldListenRef.current = false
        }
    }, [isListening])

    const stopListening = useCallback(() => {
        shouldListenRef.current = false

        if (recognitionRef.current) {
            try {
                recognitionRef.current.stop()
            } catch (e) {
                // Ignore errors when stopping
            }
        }
        setIsListening(false)
        setInterimTranscript('')
        console.log('Stopped listening')
    }, [])

    const resetTranscript = useCallback(() => {
        setTranscript('')
        setInterimTranscript('')
    }, [])

    // Combine final and interim transcripts for display
    const fullTranscript = interimTranscript
        ? (transcript ? transcript + ' ' + interimTranscript : interimTranscript)
        : transcript

    return {
        isListening,
        transcript: fullTranscript,
        finalTranscript: transcript,
        interimTranscript,
        startListening,
        stopListening,
        resetTranscript,
        isSupported,
        error
    }
}

/**
 * Hook for local Whisper-based speech recognition (fallback when Web Speech API fails)
 * Records audio and sends to local Whisper API for transcription
 */
export function useLocalWhisperSTT() {
    const [isRecording, setIsRecording] = useState(false)
    const [isTranscribing, setIsTranscribing] = useState(false)
    const [transcript, setTranscript] = useState('')
    const [error, setError] = useState<string | null>(null)
    const [isAvailable, setIsAvailable] = useState(false)

    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const audioChunksRef = useRef<Blob[]>([])
    const streamRef = useRef<MediaStream | null>(null)

    const AI_SERVICE_URL = typeof window !== 'undefined'
        ? (process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'https://localhost:8001')
        : 'https://localhost:8001'

    // Check if local Whisper STT is available
    useEffect(() => {
        const checkAvailability = async () => {
            try {
                const response = await fetch(`${AI_SERVICE_URL}/api/stt/status`, {
                    method: 'GET'
                })
                if (response.ok) {
                    const data = await response.json()
                    setIsAvailable(data.available)
                    console.log('[LocalWhisperSTT] Service available:', data.available)
                }
            } catch (err) {
                console.log('[LocalWhisperSTT] Service not reachable')
                setIsAvailable(false)
            }
        }
        checkAvailability()
    }, [AI_SERVICE_URL])

    const startRecording = useCallback(async () => {
        try {
            setError(null)
            setTranscript('')
            audioChunksRef.current = []

            // Get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            })
            streamRef.current = stream

            // Create MediaRecorder with best available format
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/webm')
                    ? 'audio/webm'
                    : 'audio/mp4'

            const recorder = new MediaRecorder(stream, { mimeType })
            mediaRecorderRef.current = recorder

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data)
                }
            }

            recorder.start(1000) // Collect chunks every second
            setIsRecording(true)
            console.log('[LocalWhisperSTT] Recording started')

        } catch (err: any) {
            console.error('[LocalWhisperSTT] Failed to start recording:', err)
            setError(err.message || 'Failed to access microphone')
        }
    }, [])

    const stopRecording = useCallback(async () => {
        return new Promise<string>((resolve) => {
            if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') {
                setIsRecording(false)
                resolve(transcript)
                return
            }

            const recorder = mediaRecorderRef.current

            recorder.onstop = async () => {
                setIsRecording(false)

                // Stop all tracks
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop())
                    streamRef.current = null
                }

                // Create audio blob
                const audioBlob = new Blob(audioChunksRef.current, { type: recorder.mimeType })
                console.log('[LocalWhisperSTT] Recording stopped, size:', audioBlob.size)

                if (audioBlob.size < 1000) {
                    console.warn('[LocalWhisperSTT] Audio too short')
                    setError('Recording too short')
                    resolve('')
                    return
                }

                // Send to Whisper API
                setIsTranscribing(true)
                try {
                    const formData = new FormData()
                    formData.append('audio', audioBlob, 'recording.webm')
                    formData.append('language', 'en')

                    console.log('[LocalWhisperSTT] Sending to Whisper API...')
                    const response = await fetch(`${AI_SERVICE_URL}/api/stt/transcribe`, {
                        method: 'POST',
                        body: formData
                    })

                    if (!response.ok) {
                        throw new Error(`Transcription failed: ${response.status}`)
                    }

                    const data = await response.json()
                    console.log('[LocalWhisperSTT] Transcription result:', data.text)
                    setTranscript(data.text)
                    setIsTranscribing(false)
                    resolve(data.text)

                } catch (err: any) {
                    console.error('[LocalWhisperSTT] Transcription failed:', err)
                    setError(err.message || 'Transcription failed')
                    setIsTranscribing(false)
                    resolve('')
                }
            }

            recorder.stop()
        })
    }, [AI_SERVICE_URL, transcript])

    const resetTranscript = useCallback(() => {
        setTranscript('')
        setError(null)
    }, [])

    const cancelRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
        }
        setIsRecording(false)
        audioChunksRef.current = []
    }, [])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            cancelRecording()
        }
    }, [cancelRecording])

    return {
        isRecording,
        isTranscribing,
        isProcessing: isRecording || isTranscribing,
        transcript,
        startRecording,
        stopRecording,
        resetTranscript,
        cancelRecording,
        isAvailable,
        error
    }
}

export default useSpeechEngine

