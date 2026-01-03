'use client'

import { useState, useEffect, useRef } from 'react'
import { UserIcon } from '@heroicons/react/24/outline'

interface AvatarViewerProps {
    avatarId: 'male' | 'female'
    isSpeaking?: boolean
    onReady?: () => void
}

// Simple 2D avatar component that works without Three.js
// This is a reliable fallback that doesn't have compatibility issues
export default function AvatarViewer({ avatarId, isSpeaking = false, onReady }: AvatarViewerProps) {
    const [loaded, setLoaded] = useState(false)
    const mouthRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        // Simulate avatar loading
        const timer = setTimeout(() => {
            setLoaded(true)
            onReady?.()
        }, 500)
        return () => clearTimeout(timer)
    }, [onReady])

    // Animate mouth when speaking
    useEffect(() => {
        if (!isSpeaking || !mouthRef.current) return

        let frame = 0
        const animate = () => {
            if (mouthRef.current) {
                const scale = 1 + Math.sin(frame * 0.3) * 0.5
                mouthRef.current.style.transform = `scaleY(${scale})`
            }
            frame++
        }

        const interval = setInterval(animate, 50)
        return () => clearInterval(interval)
    }, [isSpeaking])

    const avatarConfig = {
        male: {
            name: 'Alex',
            skinColor: '#DEB887',
            hairColor: '#2C1810',
            shirtColor: '#3B82F6',
            eyeColor: '#4A4A4A'
        },
        female: {
            name: 'Sara',
            skinColor: '#D4A574',
            hairColor: '#8B4513',
            shirtColor: '#8B5CF6',
            eyeColor: '#4A4A4A'
        }
    }

    const config = avatarConfig[avatarId]

    if (!loaded) {
        return (
            <div className="w-full h-full bg-gradient-to-b from-gray-100 to-gray-200 rounded-2xl flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 rounded-full bg-gray-300 animate-pulse mx-auto mb-3" />
                    <p className="text-gray-500 text-sm">Loading Avatar...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="w-full h-full bg-gradient-to-b from-blue-50 via-gray-50 to-gray-200 rounded-2xl overflow-hidden relative">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-100/50 to-purple-100/50" />

            {/* Avatar Container */}
            <div className="relative w-full h-full flex items-center justify-center">
                <div className="relative">
                    {/* Body/Shoulders */}
                    <div
                        className="absolute -bottom-20 left-1/2 -translate-x-1/2 w-48 h-32 rounded-t-full"
                        style={{ backgroundColor: config.shirtColor }}
                    />

                    {/* Neck */}
                    <div
                        className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-12 h-10"
                        style={{ backgroundColor: config.skinColor }}
                    />

                    {/* Head */}
                    <div
                        className={`relative w-32 h-40 rounded-[50%] ${isSpeaking ? 'animate-[slight-bob_0.5s_ease-in-out_infinite]' : ''}`}
                        style={{ backgroundColor: config.skinColor }}
                    >
                        {/* Hair */}
                        {avatarId === 'female' ? (
                            <>
                                {/* Long hair */}
                                <div
                                    className="absolute -top-2 -left-4 w-40 h-24 rounded-t-full"
                                    style={{ backgroundColor: config.hairColor }}
                                />
                                <div
                                    className="absolute top-16 -left-6 w-8 h-32 rounded-b-full"
                                    style={{ backgroundColor: config.hairColor }}
                                />
                                <div
                                    className="absolute top-16 -right-6 w-8 h-32 rounded-b-full"
                                    style={{ backgroundColor: config.hairColor }}
                                />
                            </>
                        ) : (
                            /* Short hair */
                            <div
                                className="absolute -top-2 -left-1 w-34 h-20 rounded-t-full"
                                style={{ backgroundColor: config.hairColor, width: '130px' }}
                            />
                        )}

                        {/* Eyes */}
                        <div className="absolute top-14 left-6 w-5 h-6 bg-white rounded-full overflow-hidden">
                            <div
                                className={`w-3 h-3 rounded-full absolute bottom-1 left-1 ${isSpeaking ? 'animate-[look-around_3s_ease-in-out_infinite]' : ''}`}
                                style={{ backgroundColor: config.eyeColor }}
                            />
                        </div>
                        <div className="absolute top-14 right-6 w-5 h-6 bg-white rounded-full overflow-hidden">
                            <div
                                className={`w-3 h-3 rounded-full absolute bottom-1 left-1 ${isSpeaking ? 'animate-[look-around_3s_ease-in-out_infinite]' : ''}`}
                                style={{ backgroundColor: config.eyeColor }}
                            />
                        </div>

                        {/* Eyebrows */}
                        <div className="absolute top-11 left-5 w-6 h-1 bg-gray-600 rounded-full" />
                        <div className="absolute top-11 right-5 w-6 h-1 bg-gray-600 rounded-full" />

                        {/* Nose */}
                        <div
                            className="absolute top-20 left-1/2 -translate-x-1/2 w-3 h-4 rounded-b-full opacity-30"
                            style={{ backgroundColor: '#A0866D' }}
                        />

                        {/* Mouth */}
                        <div
                            ref={mouthRef}
                            className={`absolute top-28 left-1/2 -translate-x-1/2 rounded-full transition-all ${isSpeaking ? 'w-6 h-4 bg-gray-700' : 'w-8 h-2 bg-pink-400'
                                }`}
                        />

                        {/* Blush */}
                        <div className="absolute top-20 left-2 w-4 h-3 bg-pink-200 rounded-full opacity-50" />
                        <div className="absolute top-20 right-2 w-4 h-3 bg-pink-200 rounded-full opacity-50" />
                    </div>
                </div>
            </div>

            {/* Speaking indicator */}
            {isSpeaking && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1">
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
            )}

            {/* Avatar name badge */}
            <div className="absolute top-4 left-4 bg-white px-3 py-1 rounded-full shadow-sm">
                <span className="text-sm font-medium text-gray-700">{config.name}</span>
            </div>
        </div>
    )
}
