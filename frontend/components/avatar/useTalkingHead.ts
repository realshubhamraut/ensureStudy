'use client'

/**
 * useTalkingHead - React hook for TalkingHead.js integration
 * 
 * Uses CDN-loaded TalkingHead.js to avoid Next.js bundling issues.
 * Provides a clean interface for controlling the 3D talking avatar.
 */

import { useRef, useCallback, useState, useEffect } from 'react'

// TalkingHead will be loaded via CDN
declare global {
    interface Window {
        TalkingHead?: any
        talkingHeadLoaded?: boolean
        talkingHeadLoadPromise?: Promise<void>
    }
}

// TalkingHead configuration options
export interface TalkingHeadConfig {
    cameraView?: 'full' | 'mid' | 'upper' | 'head'
    avatarMood?: 'neutral' | 'happy' | 'angry' | 'sad' | 'fear' | 'disgust' | 'love' | 'sleep'
    ttsLang?: string
    ttsVoice?: string
    ttsRate?: number
    lipsyncLang?: string
    avatarSpeakingEyeContact?: number
    avatarSpeakingHeadMove?: number
    avatarIdleEyeContact?: number
}

export interface UseTalkingHeadReturn {
    isLoaded: boolean
    isSpeaking: boolean
    error: string | null
    initAvatar: (container: HTMLElement, avatarUrl: string, config?: TalkingHeadConfig) => Promise<void>
    speak: (text: string) => Promise<void>
    stopSpeaking: () => void
    setMood: (mood: string) => void
    lookAtCamera: (duration?: number) => void
    dispose: () => void
}

// Default interviewer settings for professional appearance
const DEFAULT_CONFIG: TalkingHeadConfig = {
    cameraView: 'head',
    avatarMood: 'neutral',
    ttsLang: 'en-US',
    ttsVoice: 'en-US-Standard-C',
    ttsRate: 0.95,
    lipsyncLang: 'en',
    avatarSpeakingEyeContact: 0.8,
    avatarSpeakingHeadMove: 0.3,
    avatarIdleEyeContact: 0.3
}

// CDN URLs
const THREE_CDN = 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js'
const TALKINGHEAD_CDN = 'https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@1.7/modules/talkinghead.mjs'

// Load TalkingHead via import map
async function loadTalkingHead(): Promise<any> {
    // Check if already loaded
    if (typeof window !== 'undefined' && window.TalkingHead) {
        return window.TalkingHead
    }

    // Return existing promise if loading
    if (typeof window !== 'undefined' && window.talkingHeadLoadPromise) {
        await window.talkingHeadLoadPromise
        return window.TalkingHead
    }

    // Create loading promise
    window.talkingHeadLoadPromise = new Promise<void>((resolve, reject) => {
        // Add import map if not exists
        if (!document.querySelector('script[type="importmap"]')) {
            const importMap = document.createElement('script')
            importMap.type = 'importmap'
            importMap.textContent = JSON.stringify({
                imports: {
                    'three': THREE_CDN + '/+esm',
                    'three/addons/': 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/',
                    'talkinghead': TALKINGHEAD_CDN
                }
            })
            document.head.appendChild(importMap)
        }

        // Dynamically import TalkingHead
        const script = document.createElement('script')
        script.type = 'module'
        script.textContent = `
            import { TalkingHead } from 'talkinghead';
            window.TalkingHead = TalkingHead;
            window.talkingHeadLoaded = true;
            window.dispatchEvent(new CustomEvent('talkinghead-loaded'));
        `

        const handleLoad = () => {
            window.removeEventListener('talkinghead-loaded', handleLoad)
            resolve()
        }

        window.addEventListener('talkinghead-loaded', handleLoad)

        setTimeout(() => {
            if (!window.talkingHeadLoaded) {
                reject(new Error('TalkingHead loading timeout'))
            }
        }, 30000)

        document.head.appendChild(script)
    })

    await window.talkingHeadLoadPromise
    return window.TalkingHead
}

export function useTalkingHead(): UseTalkingHeadReturn {
    const headRef = useRef<any>(null)
    const [isLoaded, setIsLoaded] = useState(false)
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Initialize the TalkingHead avatar
    const initAvatar = useCallback(async (
        container: HTMLElement,
        avatarUrl: string,
        config: TalkingHeadConfig = {}
    ) => {
        try {
            setError(null)

            // Load TalkingHead from CDN
            const TalkingHead = await loadTalkingHead()

            if (!TalkingHead) {
                throw new Error('Failed to load TalkingHead library')
            }

            const mergedConfig = { ...DEFAULT_CONFIG, ...config }

            // Create TalkingHead instance
            headRef.current = new TalkingHead(container, {
                lipsyncModules: ['en'],
                cameraView: mergedConfig.cameraView,
                avatarMood: mergedConfig.avatarMood,
                avatarSpeakingEyeContact: mergedConfig.avatarSpeakingEyeContact,
                avatarSpeakingHeadMove: mergedConfig.avatarSpeakingHeadMove,
                avatarIdleEyeContact: mergedConfig.avatarIdleEyeContact,
                // Lighting for professional look
                lightAmbientColor: 0xffffff,
                lightAmbientIntensity: 2.5,
                lightDirectColor: 0xffffff,
                lightDirectIntensity: 20,
                lightDirectPhi: 0.5,
                lightDirectTheta: 2,
                // Model settings
                modelPixelRatio: window.devicePixelRatio || 1,
                modelFPS: 30,
                // Enable draco compression
                dracoEnabled: true,
            })

            // Load the avatar
            await headRef.current.showAvatar({
                url: avatarUrl,
                body: 'M',
                avatarMood: mergedConfig.avatarMood,
                ttsLang: mergedConfig.ttsLang,
                ttsVoice: mergedConfig.ttsVoice,
                ttsRate: mergedConfig.ttsRate,
                lipsyncLang: mergedConfig.lipsyncLang,
                baseline: {
                    headRotateX: -0.03,
                    eyeBlinkLeft: 0.1,
                    eyeBlinkRight: 0.1
                }
            })

            // Set camera view for interviewer framing
            headRef.current.setView(mergedConfig.cameraView, {
                cameraY: 0.03,
                cameraRotateX: 0.02
            })

            setIsLoaded(true)
        } catch (err) {
            console.error('Failed to initialize TalkingHead:', err)
            setError(err instanceof Error ? err.message : 'Failed to load avatar')
            setIsLoaded(false)
        }
    }, [])

    // Speak text with lip sync
    const speak = useCallback(async (text: string) => {
        if (!headRef.current || !isLoaded) {
            console.warn('TalkingHead not initialized')
            return
        }

        try {
            setIsSpeaking(true)

            await headRef.current.speakText(text, {
                avatarMood: 'neutral',
            })

            setIsSpeaking(false)
        } catch (err) {
            console.error('Speech error:', err)
            setIsSpeaking(false)
        }
    }, [isLoaded])

    // Stop speaking
    const stopSpeaking = useCallback(() => {
        if (headRef.current) {
            headRef.current.stopSpeaking?.()
            setIsSpeaking(false)
        }
    }, [])

    // Set avatar mood
    const setMood = useCallback((mood: string) => {
        if (headRef.current) {
            headRef.current.setMood(mood)
        }
    }, [])

    // Make avatar look at camera
    const lookAtCamera = useCallback((duration: number = 500) => {
        if (headRef.current) {
            headRef.current.lookAtCamera(duration)
        }
    }, [])

    // Cleanup
    const dispose = useCallback(() => {
        if (headRef.current) {
            headRef.current = null
            setIsLoaded(false)
            setIsSpeaking(false)
        }
    }, [])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            dispose()
        }
    }, [dispose])

    return {
        isLoaded,
        isSpeaking,
        error,
        initAvatar,
        speak,
        stopSpeaking,
        setMood,
        lookAtCamera,
        dispose
    }
}
