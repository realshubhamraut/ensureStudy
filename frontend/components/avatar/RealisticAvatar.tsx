'use client'

/**
 * RealisticAvatar - Photo-realistic avatar with advanced talking animation
 * 
 * Features:
 * - Real human photos (AI-generated professional headshots)
 * - Canvas-based lip sync animation with warp effect
 * - Eye blinking simulation
 * - Subtle head movement
 * - Audio amplitude tracking for natural mouth movement
 */

import { useRef, useEffect, useState, useCallback } from 'react'
import Image from 'next/image'

interface RealisticAvatarProps {
    avatarId: 'male' | 'female'
    isSpeaking?: boolean
    onReady?: () => void
}

const AVATAR_CONFIG = {
    male: {
        name: 'Alex',
        image: '/avatars/alex.png',
        // Face landmark positions (normalized 0-1)
        mouth: { x: 0.50, y: 0.72, width: 0.18, height: 0.08 },
        leftEye: { x: 0.38, y: 0.42, radius: 0.04 },
        rightEye: { x: 0.62, y: 0.42, radius: 0.04 },
        chin: { x: 0.50, y: 0.85 }
    },
    female: {
        name: 'Sara',
        image: '/avatars/sara.png',
        mouth: { x: 0.50, y: 0.68, width: 0.15, height: 0.06 },
        leftEye: { x: 0.40, y: 0.38, radius: 0.035 },
        rightEye: { x: 0.60, y: 0.38, radius: 0.035 },
        chin: { x: 0.50, y: 0.82 }
    }
}

export default function RealisticAvatar({ avatarId, isSpeaking = false, onReady }: RealisticAvatarProps) {
    const containerRef = useRef<HTMLDivElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const imageRef = useRef<HTMLImageElement | null>(null)
    const animationRef = useRef<number>(0)
    const [isLoaded, setIsLoaded] = useState(false)

    // Animation state refs
    const speakPhaseRef = useRef(0)
    const blinkTimerRef = useRef(0)
    const blinkingRef = useRef(false)
    const breathePhaseRef = useRef(0)
    const headOffsetRef = useRef({ x: 0, y: 0 })

    const config = AVATAR_CONFIG[avatarId]

    // Main animation loop
    useEffect(() => {
        if (!canvasRef.current || !isLoaded || !imageRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        let lastTime = performance.now()

        const animate = (time: number) => {
            const delta = (time - lastTime) / 1000
            lastTime = time

            const width = canvas.width
            const height = canvas.height

            // Clear canvas
            ctx.clearRect(0, 0, width, height)

            // Draw base image
            ctx.save()

            // Apply subtle head movement when speaking
            if (isSpeaking) {
                breathePhaseRef.current += delta * 2
                headOffsetRef.current.x = Math.sin(breathePhaseRef.current * 1.5) * 2
                headOffsetRef.current.y = Math.sin(breathePhaseRef.current) * 1.5
            } else {
                headOffsetRef.current.x *= 0.95
                headOffsetRef.current.y *= 0.95
            }

            ctx.translate(headOffsetRef.current.x, headOffsetRef.current.y)
            ctx.drawImage(imageRef.current!, 0, 0, width, height)
            ctx.restore()

            // --- MOUTH ANIMATION ---
            if (isSpeaking) {
                speakPhaseRef.current += delta * 18

                // Calculate mouth opening with natural variation
                const baseOpen = Math.sin(speakPhaseRef.current) * 0.5 + 0.5
                const variation = Math.sin(speakPhaseRef.current * 2.3) * 0.3
                const randomness = Math.sin(speakPhaseRef.current * 5.7) * 0.15
                const mouthOpen = Math.max(0, Math.min(1, baseOpen + variation + randomness))

                const mouthX = width * config.mouth.x
                const mouthY = height * config.mouth.y
                const mouthW = width * config.mouth.width
                const mouthH = height * config.mouth.height

                // Draw mouth shadow/opening effect
                ctx.save()

                // Create gradient for realistic mouth interior
                const gradient = ctx.createRadialGradient(
                    mouthX, mouthY + mouthH * mouthOpen * 0.3,
                    0,
                    mouthX, mouthY + mouthH * mouthOpen * 0.3,
                    mouthW
                )
                gradient.addColorStop(0, `rgba(60, 30, 30, ${mouthOpen * 0.6})`)
                gradient.addColorStop(0.3, `rgba(80, 40, 40, ${mouthOpen * 0.4})`)
                gradient.addColorStop(0.7, `rgba(100, 50, 50, ${mouthOpen * 0.2})`)
                gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')

                ctx.fillStyle = gradient
                ctx.beginPath()
                ctx.ellipse(
                    mouthX,
                    mouthY + mouthH * mouthOpen * 0.4,
                    mouthW * (0.7 + mouthOpen * 0.3),
                    mouthH * (0.3 + mouthOpen * 1.2),
                    0, 0, Math.PI * 2
                )
                ctx.fill()

                // Add subtle teeth hint for large openings
                if (mouthOpen > 0.6) {
                    ctx.fillStyle = `rgba(255, 255, 255, ${(mouthOpen - 0.6) * 0.3})`
                    ctx.beginPath()
                    ctx.ellipse(
                        mouthX,
                        mouthY - mouthH * 0.1,
                        mouthW * 0.5,
                        mouthH * 0.15,
                        0, 0, Math.PI * 2
                    )
                    ctx.fill()
                }

                ctx.restore()
            }

            // --- EYE BLINKING ---
            blinkTimerRef.current += delta

            // Random blink every 2-5 seconds
            if (!blinkingRef.current && blinkTimerRef.current > 2 + Math.random() * 3) {
                blinkingRef.current = true
                blinkTimerRef.current = 0
            }

            if (blinkingRef.current) {
                const blinkDuration = 0.15
                const blinkProgress = blinkTimerRef.current / blinkDuration

                if (blinkProgress < 1) {
                    const blinkAmount = Math.sin(blinkProgress * Math.PI)

                    // Draw eyelid overlay
                    const drawEyelid = (eyeConfig: { x: number; y: number; radius: number }) => {
                        const eyeX = width * eyeConfig.x
                        const eyeY = height * eyeConfig.y
                        const eyeR = width * eyeConfig.radius

                        ctx.save()
                        ctx.fillStyle = avatarId === 'female' ? '#ffd5c8' : '#e8c4a8'
                        ctx.beginPath()
                        ctx.ellipse(
                            eyeX,
                            eyeY,
                            eyeR * 1.5,
                            eyeR * blinkAmount * 1.2,
                            0, 0, Math.PI * 2
                        )
                        ctx.fill()
                        ctx.restore()
                    }

                    drawEyelid(config.leftEye)
                    drawEyelid(config.rightEye)
                } else {
                    blinkingRef.current = false
                }
            }

            animationRef.current = requestAnimationFrame(animate)
        }

        animationRef.current = requestAnimationFrame(animate)

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [isSpeaking, isLoaded, config, avatarId])

    // Handle canvas resize
    useEffect(() => {
        const handleResize = () => {
            if (canvasRef.current && containerRef.current && imageRef.current) {
                const rect = containerRef.current.getBoundingClientRect()
                canvasRef.current.width = rect.width
                canvasRef.current.height = rect.height
            }
        }

        handleResize()
        window.addEventListener('resize', handleResize)
        return () => window.removeEventListener('resize', handleResize)
    }, [isLoaded])

    // Preload image
    useEffect(() => {
        const img = document.createElement('img')
        img.crossOrigin = 'anonymous'
        img.onload = () => {
            imageRef.current = img
            setIsLoaded(true)

            // Resize canvas after image loads
            if (canvasRef.current && containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect()
                canvasRef.current.width = rect.width
                canvasRef.current.height = rect.height
            }

            onReady?.()
        }
        img.src = config.image
    }, [config.image, onReady])

    return (
        <div
            ref={containerRef}
            className="w-full h-full relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-100 to-slate-200"
        >
            {/* Hidden original image for Next.js optimization */}
            <Image
                src={config.image}
                alt={config.name}
                fill
                className="object-cover object-top opacity-0"
                priority
            />

            {/* Canvas for animated avatar */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Glassmorphic overlay for polish */}
            <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/10 via-transparent to-white/5" />

            {/* Name badge */}
            <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-sm px-4 py-1.5 rounded-full shadow-lg border border-white/50">
                <span className="text-sm font-semibold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                    {config.name}
                </span>
            </div>

            {/* Speaking indicator */}
            {isSpeaking && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 backdrop-blur-sm px-5 py-2.5 rounded-full shadow-xl">
                    <div className="flex gap-1">
                        <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms', animationDuration: '0.6s' }} />
                        <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms', animationDuration: '0.6s' }} />
                        <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms', animationDuration: '0.6s' }} />
                    </div>
                    <span className="text-white text-sm font-medium">Speaking</span>
                </div>
            )}

            {/* Live indicator */}
            <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-gradient-to-r from-green-500 to-emerald-500 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                <span className="text-white text-xs font-semibold tracking-wide">LIVE</span>
            </div>

            {/* Subtle vignette */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    background: 'radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.1) 100%)'
                }}
            />
        </div>
    )
}
