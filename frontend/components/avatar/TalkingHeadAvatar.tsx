'use client'

/**
 * TalkingHeadAvatar - Professional 3D Interviewer Avatar
 * 
 * Uses TalkingHead.js for real-time lip sync with 3D avatars.
 * Features: Eye contact, subtle head movements, natural expressions.
 */

import { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react'

interface TalkingHeadAvatarProps {
    avatarId: 'male' | 'female'
    isSpeaking?: boolean
    textToSpeak?: string
    onReady?: () => void
    onSpeechStart?: () => void
    onSpeechEnd?: () => void
}

// Avatar configurations using TalkingHead demo avatars from GitHub
const AVATAR_CONFIG = {
    male: {
        name: 'Alex',
        // TalkingHead demo avatar - brunette (works with all visemes)
        url: 'https://raw.githubusercontent.com/met4citizen/TalkingHead/main/avatars/brunette.glb',
        body: 'F', // brunette is female body type
    },
    female: {
        name: 'Sara',
        // TalkingHead demo avatar - avaturn
        url: 'https://raw.githubusercontent.com/met4citizen/TalkingHead/main/avatars/avaturn.glb',
        body: 'F',
    }
}

// TalkingHead CDN URLs
const CDN_BASE = 'https://cdn.jsdelivr.net'

export default function TalkingHeadAvatar({
    avatarId,
    isSpeaking = false,
    textToSpeak,
    onReady,
    onSpeechStart,
    onSpeechEnd
}: TalkingHeadAvatarProps) {
    const containerRef = useRef<HTMLDivElement>(null)
    const headRef = useRef<any>(null)
    const [isLoaded, setIsLoaded] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [loadingProgress, setLoadingProgress] = useState(0)
    const lastSpokenTextRef = useRef<string>('')
    const initializingRef = useRef(false)

    const config = AVATAR_CONFIG[avatarId]

    // Initialize TalkingHead via script injection
    useEffect(() => {
        if (!containerRef.current || initializingRef.current) return
        initializingRef.current = true

        const init = async () => {
            try {
                // Create import map for module resolution
                if (!document.getElementById('talkinghead-importmap')) {
                    const importMap = document.createElement('script')
                    importMap.id = 'talkinghead-importmap'
                    importMap.type = 'importmap'
                    importMap.textContent = JSON.stringify({
                        imports: {
                            'three': `${CDN_BASE}/npm/three@0.180.0/build/three.module.js`,
                            'three/addons/': `${CDN_BASE}/npm/three@0.180.0/examples/jsm/`,
                            'talkinghead': `${CDN_BASE}/gh/met4citizen/TalkingHead@1.7/modules/talkinghead.mjs`
                        }
                    })
                    document.head.insertBefore(importMap, document.head.firstChild)
                }

                // Load TalkingHead via dynamic import script
                await new Promise<void>((resolve, reject) => {
                    if ((window as any).TalkingHead) {
                        resolve()
                        return
                    }

                    const loaderScript = document.createElement('script')
                    loaderScript.id = 'talkinghead-loader'
                    loaderScript.type = 'module'
                    loaderScript.textContent = `
                        import { TalkingHead } from 'talkinghead';
                        window.TalkingHead = TalkingHead;
                        window.dispatchEvent(new CustomEvent('talkinghead-ready'));
                    `

                    const handleReady = () => {
                        window.removeEventListener('talkinghead-ready', handleReady)
                        resolve()
                    }

                    window.addEventListener('talkinghead-ready', handleReady)

                    // Timeout after 30 seconds
                    setTimeout(() => {
                        if (!(window as any).TalkingHead) {
                            reject(new Error('TalkingHead load timeout'))
                        }
                    }, 30000)

                    document.head.appendChild(loaderScript)
                })

                const TalkingHead = (window as any).TalkingHead
                if (!TalkingHead) {
                    throw new Error('TalkingHead not available')
                }

                // Create TalkingHead instance
                headRef.current = new TalkingHead(containerRef.current, {
                    lipsyncModules: ['en'],
                    cameraView: 'head',
                    avatarMood: 'neutral',
                    avatarSpeakingEyeContact: 0.8,
                    avatarSpeakingHeadMove: 0.3,
                    avatarIdleEyeContact: 0.3,
                    lightAmbientColor: 0xffffff,
                    lightAmbientIntensity: 2.5,
                    lightDirectColor: 0xffffff,
                    lightDirectIntensity: 20,
                    lightDirectPhi: 0.5,
                    lightDirectTheta: 2,
                    modelPixelRatio: window.devicePixelRatio || 1,
                    modelFPS: 30,
                    dracoEnabled: true,
                })

                // Load avatar
                await headRef.current.showAvatar(
                    {
                        url: config.url,
                        body: config.body,
                        avatarMood: 'neutral',
                        lipsyncLang: 'en',
                        ttsLang: 'en-GB',
                        ttsVoice: config.body === 'M' ? 'en-GB-Standard-B' : 'en-GB-Standard-A',
                        baseline: {
                            headRotateX: -0.03,
                            eyeBlinkLeft: 0.1,
                            eyeBlinkRight: 0.1
                        }
                    },
                    (progress: number) => {
                        setLoadingProgress(Math.round(progress * 100))
                    }
                )

                // Set camera view
                headRef.current.setView('head', {
                    cameraY: 0.03,
                    cameraRotateX: 0.02
                })

                setIsLoaded(true)
                setError(null)
                onReady?.()
            } catch (err) {
                console.error('Avatar init error:', err)
                setError(err instanceof Error ? err.message : 'Failed to load avatar')
                initializingRef.current = false
            }
        }

        init()

        return () => {
            if (headRef.current) {
                headRef.current = null
            }
        }
    }, [avatarId, config.body, config.url, onReady])

    // Handle speaking via textToSpeak prop - AWS Polly TTS with viseme lip sync
    useEffect(() => {
        if (!isLoaded || !headRef.current) return

        if (textToSpeak && textToSpeak !== lastSpokenTextRef.current) {
            lastSpokenTextRef.current = textToSpeak
            onSpeechStart?.()

            const speakWithPolly = async () => {
                const head = headRef.current
                if (!head) return

                try {
                    // Get AI service URL
                    const aiServiceUrl = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

                    console.log('[TalkingHead] Requesting Polly TTS...')

                    // Call Polly TTS API
                    const response = await fetch(`${aiServiceUrl}/api/tts/synthesize`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: textToSpeak,
                            voice: avatarId === 'male' ? 'male' : 'female'
                        })
                    })

                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}))
                        console.warn('[TalkingHead] Polly API error, falling back to browser TTS:', errorData)
                        // Fall back to browser TTS
                        await speakWithBrowserTTS(head, textToSpeak)
                        return
                    }

                    const data = await response.json()
                    const { audio_base64, visemes, duration_ms } = data

                    console.log(`[TalkingHead] Got ${visemes.length} visemes, duration: ${duration_ms}ms`)

                    // Create audio from base64
                    const audioBlob = base64ToBlob(audio_base64, 'audio/mp3')
                    const audioUrl = URL.createObjectURL(audioBlob)
                    const audio = new Audio(audioUrl)

                    // Start TalkingHead streaming for viseme-based lip sync
                    if (head.streamStart) {
                        head.streamStart({
                            lipsyncLang: 'en',
                            lipsyncType: 'visemes',
                            waitForAudioChunks: false,
                            mood: 'neutral'
                        })
                    }

                    // Schedule visemes based on timing
                    const visemeTimeouts: NodeJS.Timeout[] = []
                    visemes.forEach((v: { time: number; value: string }) => {
                        const timeout = setTimeout(() => {
                            try {
                                // Apply viseme to TalkingHead
                                if (head.setMorphTarget) {
                                    // Reset all visemes first
                                    const oculusVisemes = ['PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'aa', 'E', 'I', 'O', 'U', 'sil']
                                    oculusVisemes.forEach(vis => {
                                        try { head.setMorphTarget(`viseme_${vis}`, 0) } catch { }
                                    })
                                    // Set current viseme
                                    head.setMorphTarget(`viseme_${v.value}`, 1)
                                }
                            } catch (e) {
                                // Ignore viseme errors
                            }
                        }, v.time)
                        visemeTimeouts.push(timeout)
                    })

                    // Play audio
                    audio.onended = () => {
                        console.log('[TalkingHead] Polly audio completed')
                        // Clean up
                        visemeTimeouts.forEach(t => clearTimeout(t))
                        URL.revokeObjectURL(audioUrl)
                        if (head.streamStop) {
                            head.streamStop()
                        }
                        // Reset to neutral
                        try {
                            if (head.setMorphTarget) {
                                head.setMorphTarget('viseme_sil', 1)
                            }
                        } catch { }
                        onSpeechEnd?.()
                    }

                    audio.onerror = (e) => {
                        console.error('[TalkingHead] Audio playback error:', e)
                        visemeTimeouts.forEach(t => clearTimeout(t))
                        URL.revokeObjectURL(audioUrl)
                        onSpeechEnd?.()
                    }

                    await audio.play()

                } catch (err) {
                    console.error('[TalkingHead] Polly error, falling back to browser TTS:', err)
                    // Fall back to browser TTS
                    await speakWithBrowserTTS(head, textToSpeak)
                }
            }

            // Fallback: Browser TTS (when Polly unavailable)
            const speakWithBrowserTTS = async (head: any, text: string) => {
                if (!window.speechSynthesis) {
                    console.error('Speech synthesis not supported')
                    onSpeechEnd?.()
                    return
                }

                const utterance = new SpeechSynthesisUtterance(text)
                utterance.lang = 'en-US'
                utterance.rate = 0.95

                const voices = speechSynthesis.getVoices()
                const preferredVoice = voices.find(v => v.lang.startsWith('en')) || voices[0]
                if (preferredVoice) utterance.voice = preferredVoice

                utterance.onend = () => {
                    console.log('[TalkingHead] Browser TTS completed')
                    onSpeechEnd?.()
                }

                utterance.onerror = () => {
                    onSpeechEnd?.()
                }

                speechSynthesis.cancel()
                speechSynthesis.speak(utterance)
            }

            speakWithPolly()
        }
    }, [textToSpeak, isLoaded, avatarId, onSpeechStart, onSpeechEnd])

    // Helper: Convert base64 to Blob
    function base64ToBlob(base64: string, mimeType: string): Blob {
        const byteCharacters = atob(base64)
        const byteNumbers = new Array(byteCharacters.length)

        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i)
        }

        const byteArray = new Uint8Array(byteNumbers)
        return new Blob([byteArray], { type: mimeType })
    }


    // Handle isSpeaking changes (for stopping)
    useEffect(() => {
        if (!isLoaded || !headRef.current) return

        if (!isSpeaking) {
            // Could stop speaking here if needed
        } else {
            headRef.current.lookAtCamera?.(1000)
        }
    }, [isSpeaking, isLoaded])

    // Show error state
    if (error) {
        return (
            <div className="w-full h-full bg-gradient-to-b from-slate-100 to-slate-200 rounded-2xl flex flex-col items-center justify-center p-8">
                <div className="text-6xl mb-4">😕</div>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Avatar Unavailable</h3>
                <p className="text-sm text-gray-500 text-center max-w-xs mb-4">
                    {error}
                </p>
                <button
                    onClick={() => {
                        setError(null)
                        initializingRef.current = false
                    }}
                    className="px-4 py-2 bg-indigo-500 text-white rounded-lg text-sm hover:bg-indigo-600 transition"
                >
                    Retry
                </button>
            </div>
        )
    }

    return (
        <div className="w-full h-full relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-50 to-slate-100">
            {/* TalkingHead container */}
            <div
                ref={containerRef}
                className="w-full h-full"
                style={{ minHeight: '400px' }}
            />

            {/* Loading overlay */}
            {!isLoaded && (
                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-slate-100 to-slate-200">
                    <div className="text-center">
                        <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                        <p className="text-sm font-medium text-gray-600">Loading 3D Avatar...</p>
                        <p className="text-xs text-gray-400 mt-1">
                            {loadingProgress > 0 ? `${loadingProgress}%` : 'Initializing...'}
                        </p>
                    </div>
                </div>
            )}

            {/* Name badge */}
            <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-sm px-4 py-1.5 rounded-full shadow-lg border border-white/50 z-10">
                <span className="text-sm font-semibold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                    {config.name}
                </span>
            </div>

            {/* Live indicator */}
            <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-gradient-to-r from-green-500 to-emerald-500 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg z-10">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                <span className="text-white text-xs font-semibold tracking-wide">LIVE</span>
            </div>

            {/* Speaking indicator */}
            {isSpeaking && isLoaded && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 backdrop-blur-sm px-5 py-2.5 rounded-full shadow-xl z-10">
                    <div className="flex gap-1">
                        <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-white text-sm font-medium">Speaking</span>
                </div>
            )}

            {/* Subtle vignette */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.05) 100%)',
                    zIndex: 5
                }}
            />
        </div>
    )
}
