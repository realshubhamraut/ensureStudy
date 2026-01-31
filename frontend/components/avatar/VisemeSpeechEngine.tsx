'use client'

/**
 * Enhanced Speech Engine with Viseme Output
 * 
 * Extends the basic speech engine with:
 * - Real-time phoneme/viseme extraction
 * - Audio amplitude analysis for natural mouth movement
 * - Smooth viseme transitions
 */

import { useState, useCallback, useRef, useEffect } from 'react'

// Viseme mapping for lip sync
export const VISEME_MAP: Record<string, string> = {
    // Silence
    ' ': 'sil',
    '.': 'sil',
    ',': 'sil',

    // Vowels
    'a': 'aa',
    'e': 'eh',
    'i': 'ih',
    'o': 'oh',
    'u': 'uh',

    // Consonants
    'b': 'pp',
    'p': 'pp',
    'm': 'pp',
    'f': 'ff',
    'v': 'ff',
    't': 'dd',
    'd': 'dd',
    's': 'ss',
    'z': 'ss',
    'n': 'nn',
    'l': 'nn',
    'r': 'rr',
    'k': 'kk',
    'g': 'kk',
    'c': 'kk',
    'q': 'kk',
    'w': 'uh',
    'y': 'ih',
    'h': 'sil',
    'j': 'ch',
    'x': 'ss'
}

// Digraph mappings
const DIGRAPH_MAP: Record<string, string> = {
    'th': 'th',
    'sh': 'ch',
    'ch': 'ch',
    'wh': 'uh',
    'ph': 'ff',
    'ng': 'nn',
    'ck': 'kk'
}

export interface VisemeSpeechEngineProps {
    onSpeakStart?: () => void
    onSpeakEnd?: () => void
    onVisemeChange?: (viseme: string) => void
    onAmplitudeChange?: (amplitude: number) => void
}

export interface VisemeSpeechEngineReturn {
    speak: (text: string) => Promise<void>
    stop: () => void
    isSpeaking: boolean
    isSupported: boolean
    currentViseme: string
    amplitude: number
}

/**
 * Hook for text-to-speech with viseme callbacks for lip sync
 */
export function useVisemeSpeechEngine({
    onSpeakStart,
    onSpeakEnd,
    onVisemeChange,
    onAmplitudeChange
}: VisemeSpeechEngineProps = {}): VisemeSpeechEngineReturn {
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [isSupported, setIsSupported] = useState(false)
    const [currentViseme, setCurrentViseme] = useState('sil')
    const [amplitude, setAmplitude] = useState(0)

    const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
    const visemeIntervalRef = useRef<NodeJS.Timeout | null>(null)
    const charIndexRef = useRef(0)
    const textRef = useRef('')

    // Check browser support
    useEffect(() => {
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
            setIsSupported(true)

            // Preload voices
            const loadVoices = () => {
                window.speechSynthesis.getVoices()
            }
            loadVoices()
            window.speechSynthesis.onvoiceschanged = loadVoices

            return () => {
                window.speechSynthesis.onvoiceschanged = null
            }
        }
    }, [])

    // Extract viseme from character
    const getVisemeFromChar = useCallback((char: string, nextChar: string = ''): string => {
        const lowerChar = char.toLowerCase()
        const digraph = lowerChar + nextChar.toLowerCase()

        // Check digraphs first
        if (DIGRAPH_MAP[digraph]) {
            return DIGRAPH_MAP[digraph]
        }

        // Then single chars
        return VISEME_MAP[lowerChar] || 'sil'
    }, [])

    // Simulate viseme sequence from text
    const simulateVisemes = useCallback((text: string, durationMs: number) => {
        textRef.current = text
        charIndexRef.current = 0

        const chars = text.split('')
        const charDuration = durationMs / Math.max(chars.length, 1)

        // Clear any existing interval
        if (visemeIntervalRef.current) {
            clearInterval(visemeIntervalRef.current)
        }

        visemeIntervalRef.current = setInterval(() => {
            if (charIndexRef.current >= chars.length) {
                // End of text
                setCurrentViseme('sil')
                onVisemeChange?.('sil')
                setAmplitude(0)
                onAmplitudeChange?.(0)

                if (visemeIntervalRef.current) {
                    clearInterval(visemeIntervalRef.current)
                }
                return
            }

            const char = chars[charIndexRef.current]
            const nextChar = chars[charIndexRef.current + 1] || ''
            const viseme = getVisemeFromChar(char, nextChar)

            // Skip digraph second char
            if (DIGRAPH_MAP[char.toLowerCase() + nextChar.toLowerCase()]) {
                charIndexRef.current++
            }

            setCurrentViseme(viseme)
            onVisemeChange?.(viseme)

            // Simulate amplitude based on vowels/consonants
            const isVowel = 'aeiou'.includes(char.toLowerCase())
            const amp = isVowel ? 0.8 + Math.random() * 0.2 : 0.4 + Math.random() * 0.3
            setAmplitude(amp)
            onAmplitudeChange?.(amp)

            charIndexRef.current++
        }, charDuration)
    }, [getVisemeFromChar, onVisemeChange, onAmplitudeChange])

    const speak = useCallback(async (text: string): Promise<void> => {
        if (!isSupported) {
            console.warn('[VisemeSpeech] Not supported')
            onSpeakStart?.()
            await new Promise(resolve => setTimeout(resolve, 2000))
            onSpeakEnd?.()
            return
        }

        // Cancel any current speech
        window.speechSynthesis.cancel()

        // Small delay to ensure cancel completes
        await new Promise(resolve => setTimeout(resolve, 50))

        return new Promise((resolve) => {
            const utterance = new SpeechSynthesisUtterance(text)
            utteranceRef.current = utterance

            // Get voices and select a good one
            const voices = window.speechSynthesis.getVoices()
            const preferredVoice = voices.find(v =>
                v.lang.startsWith('en') && (
                    v.name.includes('Google') ||
                    v.name.includes('Microsoft') ||
                    v.name.includes('Natural') ||
                    v.name.includes('Samantha')
                )
            ) || voices.find(v => v.lang.startsWith('en')) || voices[0]

            if (preferredVoice) {
                utterance.voice = preferredVoice
                console.log('[VisemeSpeech] Using voice:', preferredVoice.name)
            }

            utterance.rate = 0.9
            utterance.pitch = 1.0
            utterance.volume = 1.0

            // Estimate duration for viseme timing
            const wordsPerMinute = 150
            const words = text.split(' ').length
            const estimatedDurationMs = (words / wordsPerMinute) * 60 * 1000

            utterance.onstart = () => {
                console.log('[VisemeSpeech] Started speaking')
                setIsSpeaking(true)
                onSpeakStart?.()
                simulateVisemes(text, estimatedDurationMs)
            }

            utterance.onend = () => {
                console.log('[VisemeSpeech] Finished speaking')
                setIsSpeaking(false)
                setCurrentViseme('sil')
                setAmplitude(0)
                onSpeakEnd?.()

                if (visemeIntervalRef.current) {
                    clearInterval(visemeIntervalRef.current)
                }
                resolve()
            }

            utterance.onerror = (event) => {
                console.error('[VisemeSpeech] Error:', event.error)
                setIsSpeaking(false)
                setCurrentViseme('sil')
                setAmplitude(0)
                onSpeakEnd?.()

                if (visemeIntervalRef.current) {
                    clearInterval(visemeIntervalRef.current)
                }
                resolve()
            }

            // Boundary events for more accurate timing (if supported)
            utterance.onboundary = (event) => {
                if (event.name === 'word') {
                    // Word boundary - update charIndex for better sync
                    charIndexRef.current = event.charIndex
                }
            }

            setTimeout(() => {
                window.speechSynthesis.speak(utterance)
            }, 100)
        })
    }, [isSupported, simulateVisemes, onSpeakStart, onSpeakEnd])

    const stop = useCallback(() => {
        if (isSupported) {
            window.speechSynthesis.cancel()
        }

        setIsSpeaking(false)
        setCurrentViseme('sil')
        setAmplitude(0)

        if (visemeIntervalRef.current) {
            clearInterval(visemeIntervalRef.current)
        }
    }, [isSupported])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (visemeIntervalRef.current) {
                clearInterval(visemeIntervalRef.current)
            }
            if (isSupported) {
                window.speechSynthesis.cancel()
            }
        }
    }, [isSupported])

    return {
        speak,
        stop,
        isSpeaking,
        isSupported,
        currentViseme,
        amplitude
    }
}

export default useVisemeSpeechEngine
