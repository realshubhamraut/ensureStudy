'use client'

/**
 * useAudioViseme - Real-time audio analysis for viseme selection
 * 
 * Analyzes audio amplitude and frequency from TTS output
 * to determine which mouth shape (viseme) to display.
 * 
 * Extremely lightweight - uses native Web Audio API
 */

import { useRef, useCallback, useEffect, useState } from 'react'

// Viseme types based on Preston Blair phoneme set
export type VisemeType = 'rest' | 'A' | 'E' | 'O' | 'U' | 'M'

interface UseAudioVisemeReturn {
    currentViseme: VisemeType
    intensity: number // 0-1 for animation blending
    connectAudio: (audioElement: HTMLAudioElement) => void
    disconnectAudio: () => void
    isAnalyzing: boolean
}

export function useAudioViseme(): UseAudioVisemeReturn {
    const [currentViseme, setCurrentViseme] = useState<VisemeType>('rest')
    const [intensity, setIntensity] = useState(0)
    const [isAnalyzing, setIsAnalyzing] = useState(false)

    const audioContextRef = useRef<AudioContext | null>(null)
    const analyserRef = useRef<AnalyserNode | null>(null)
    const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)
    const animationRef = useRef<number>(0)
    const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null)

    // Viseme selection based on amplitude bands
    const selectViseme = useCallback((amplitude: number, lowFreq: number, highFreq: number): VisemeType => {
        if (amplitude < 0.05) return 'rest'

        // High amplitude = wide open mouth
        if (amplitude > 0.7) {
            return highFreq > lowFreq ? 'A' : 'O'
        }

        // Medium amplitude with high frequencies = E (smile-like)
        if (amplitude > 0.4) {
            if (highFreq > lowFreq * 1.2) return 'E'
            return 'A'
        }

        // Lower amplitude
        if (amplitude > 0.2) {
            if (lowFreq > highFreq) return 'O'
            return 'U'
        }

        // Very low but audible = M (closed but speaking)
        return 'M'
    }, [])

    // Analysis loop
    const analyze = useCallback(() => {
        if (!analyserRef.current || !dataArrayRef.current) return

        analyserRef.current.getByteFrequencyData(dataArrayRef.current)

        const data = dataArrayRef.current
        const bufferLength = data.length

        // Calculate overall amplitude
        let sum = 0
        for (let i = 0; i < bufferLength; i++) {
            sum += data[i]
        }
        const avgAmplitude = sum / bufferLength / 255

        // Calculate low frequency energy (voice fundamental)
        let lowSum = 0
        const lowEnd = Math.floor(bufferLength * 0.15) // ~0-300Hz
        for (let i = 0; i < lowEnd; i++) {
            lowSum += data[i]
        }
        const lowFreq = lowSum / lowEnd / 255

        // Calculate high frequency energy (consonants, sibilants)
        let highSum = 0
        const highStart = Math.floor(bufferLength * 0.3)
        const highEnd = Math.floor(bufferLength * 0.6)
        for (let i = highStart; i < highEnd; i++) {
            highSum += data[i]
        }
        const highFreq = highSum / (highEnd - highStart) / 255

        // Select viseme
        const viseme = selectViseme(avgAmplitude, lowFreq, highFreq)
        setCurrentViseme(viseme)
        setIntensity(Math.min(1, avgAmplitude * 2))

        animationRef.current = requestAnimationFrame(analyze)
    }, [selectViseme])

    // Connect to audio element
    const connectAudio = useCallback((audioElement: HTMLAudioElement) => {
        try {
            // Create audio context if needed
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
            }

            const ctx = audioContextRef.current

            // Resume if suspended
            if (ctx.state === 'suspended') {
                ctx.resume()
            }

            // Create analyser
            analyserRef.current = ctx.createAnalyser()
            analyserRef.current.fftSize = 256
            analyserRef.current.smoothingTimeConstant = 0.5

            const bufferLength = analyserRef.current.frequencyBinCount
            dataArrayRef.current = new Uint8Array(bufferLength)

            // Connect source
            // Check if already connected to avoid error
            if (!sourceRef.current) {
                sourceRef.current = ctx.createMediaElementSource(audioElement)
            }

            sourceRef.current.connect(analyserRef.current)
            analyserRef.current.connect(ctx.destination)

            setIsAnalyzing(true)
            animationRef.current = requestAnimationFrame(analyze)
        } catch (error) {
            console.warn('Audio analysis not available:', error)
        }
    }, [analyze])

    // Disconnect audio
    const disconnectAudio = useCallback(() => {
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current)
        }

        setIsAnalyzing(false)
        setCurrentViseme('rest')
        setIntensity(0)
    }, [])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [])

    return {
        currentViseme,
        intensity,
        connectAudio,
        disconnectAudio,
        isAnalyzing
    }
}

/**
 * Simple amplitude-based viseme hook for speechSynthesis
 * Since Web Speech API doesn't expose audio, we simulate based on timing
 */
export function useSpeechViseme(): {
    currentViseme: VisemeType
    intensity: number
    startSpeaking: () => void
    stopSpeaking: () => void
} {
    const [currentViseme, setCurrentViseme] = useState<VisemeType>('rest')
    const [intensity, setIntensity] = useState(0)
    const animationRef = useRef<number>(0)
    const phaseRef = useRef(0)
    const isSpeakingRef = useRef(false)

    // Viseme sequence for natural speech simulation
    const VISEME_SEQUENCE: VisemeType[] = ['M', 'A', 'E', 'O', 'U', 'A', 'M', 'E']

    const animate = useCallback(() => {
        if (!isSpeakingRef.current) {
            setCurrentViseme('rest')
            setIntensity(0)
            return
        }

        phaseRef.current += 0.15 // Speed of mouth movement

        // Natural variation using sine waves
        const time = phaseRef.current
        const wave1 = Math.sin(time * 2.5)
        const wave2 = Math.sin(time * 4.1)
        const wave3 = Math.sin(time * 7.3)

        // Combined intensity
        const rawIntensity = 0.3 + 0.3 * wave1 + 0.2 * wave2 + 0.1 * wave3
        const clampedIntensity = Math.max(0, Math.min(1, rawIntensity))
        setIntensity(clampedIntensity)

        // Select viseme based on intensity
        let viseme: VisemeType
        if (clampedIntensity > 0.7) {
            viseme = Math.random() > 0.5 ? 'A' : 'O'
        } else if (clampedIntensity > 0.5) {
            viseme = Math.random() > 0.5 ? 'E' : 'A'
        } else if (clampedIntensity > 0.3) {
            viseme = Math.random() > 0.5 ? 'U' : 'E'
        } else if (clampedIntensity > 0.1) {
            viseme = 'M'
        } else {
            viseme = 'rest'
        }

        setCurrentViseme(viseme)
        animationRef.current = requestAnimationFrame(animate)
    }, [])

    const startSpeaking = useCallback(() => {
        isSpeakingRef.current = true
        phaseRef.current = 0
        animationRef.current = requestAnimationFrame(animate)
    }, [animate])

    const stopSpeaking = useCallback(() => {
        isSpeakingRef.current = false
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current)
        }
        setCurrentViseme('rest')
        setIntensity(0)
    }, [])

    useEffect(() => {
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [])

    return {
        currentViseme,
        intensity,
        startSpeaking,
        stopSpeaking
    }
}
