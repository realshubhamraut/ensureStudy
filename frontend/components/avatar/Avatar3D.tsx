'use client'

/**
 * Avatar3D - Procedural 3D avatar with lip sync
 * 
 * Uses Three.js to create a stylized 3D head with:
 * - Animated mouth for speech
 * - Blinking eyes
 * - Subtle head movements
 * - Professional appearance
 */

import { Suspense, useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, Html } from '@react-three/drei'
import * as THREE from 'three'

interface Avatar3DProps {
    avatarId: 'male' | 'female'
    isSpeaking?: boolean
    onReady?: () => void
}

// Stylized 3D Head Component
function StylizedHead({ isSpeaking, avatarId }: { isSpeaking: boolean; avatarId: 'male' | 'female' }) {
    const groupRef = useRef<THREE.Group>(null)
    const mouthRef = useRef<THREE.Mesh>(null)
    const leftEyeRef = useRef<THREE.Mesh>(null)
    const rightEyeRef = useRef<THREE.Mesh>(null)
    const leftPupilRef = useRef<THREE.Mesh>(null)
    const rightPupilRef = useRef<THREE.Mesh>(null)

    const blinkTimer = useRef(0)
    const breatheTimer = useRef(0)
    const speakTimer = useRef(0)
    const lookTimer = useRef(0)

    // Colors based on avatar
    const skinColor = avatarId === 'female' ? '#ffd5c8' : '#e8c4a8'
    const hairColor = avatarId === 'female' ? '#4a3728' : '#2a1810'
    const eyeColor = avatarId === 'female' ? '#4a90d9' : '#5d4037'
    const lipColor = avatarId === 'female' ? '#d48a8a' : '#c47a7a'

    useFrame((state, delta) => {
        if (!groupRef.current) return

        // Breathing animation
        breatheTimer.current += delta
        const breathe = Math.sin(breatheTimer.current * 0.8) * 0.01
        groupRef.current.position.y = breathe

        // Head sway
        const sway = Math.sin(breatheTimer.current * 0.3) * 0.02
        groupRef.current.rotation.z = sway

        // Speaking head nod
        if (isSpeaking) {
            speakTimer.current += delta
            groupRef.current.rotation.x = Math.sin(speakTimer.current * 3) * 0.03
            groupRef.current.rotation.y = Math.sin(speakTimer.current * 2) * 0.02
        } else {
            groupRef.current.rotation.x *= 0.95
            groupRef.current.rotation.y *= 0.95
        }

        // Blinking
        blinkTimer.current += delta
        if (blinkTimer.current > 3 + Math.random() * 2) {
            blinkTimer.current = 0
        }

        let blinkScale = 1
        if (blinkTimer.current < 0.15) {
            blinkScale = 1 - Math.sin((blinkTimer.current / 0.15) * Math.PI)
        }

        if (leftEyeRef.current) leftEyeRef.current.scale.y = blinkScale
        if (rightEyeRef.current) rightEyeRef.current.scale.y = blinkScale

        // Eye look around
        lookTimer.current += delta * 0.5
        const lookX = Math.sin(lookTimer.current) * 0.02
        const lookY = Math.sin(lookTimer.current * 0.7) * 0.01

        if (leftPupilRef.current) {
            leftPupilRef.current.position.x = lookX
            leftPupilRef.current.position.y = lookY
        }
        if (rightPupilRef.current) {
            rightPupilRef.current.position.x = lookX
            rightPupilRef.current.position.y = lookY
        }

        // Mouth animation when speaking
        if (mouthRef.current) {
            if (isSpeaking) {
                const mouthOpen = 0.5 + Math.sin(speakTimer.current * 12) * 0.3 + Math.random() * 0.2
                mouthRef.current.scale.y = mouthOpen
                mouthRef.current.scale.x = 1 + mouthOpen * 0.3
            } else {
                mouthRef.current.scale.y = THREE.MathUtils.lerp(mouthRef.current.scale.y, 0.3, delta * 5)
                mouthRef.current.scale.x = THREE.MathUtils.lerp(mouthRef.current.scale.x, 1, delta * 5)
            }
        }
    })

    return (
        <group ref={groupRef} position={[0, 0, 0]}>
            {/* Head */}
            <mesh position={[0, 0, 0]}>
                <sphereGeometry args={[1, 32, 32]} />
                <meshStandardMaterial color={skinColor} roughness={0.8} />
            </mesh>

            {/* Hair */}
            <mesh position={[0, 0.3, -0.1]}>
                <sphereGeometry args={[1.05, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.6]} />
                <meshStandardMaterial color={hairColor} roughness={0.9} />
            </mesh>

            {/* Left Eye White */}
            <mesh ref={leftEyeRef} position={[-0.35, 0.15, 0.85]}>
                <sphereGeometry args={[0.18, 16, 16]} />
                <meshStandardMaterial color="white" />
            </mesh>

            {/* Left Pupil */}
            <mesh ref={leftPupilRef} position={[-0.35, 0.15, 1.0]}>
                <sphereGeometry args={[0.08, 16, 16]} />
                <meshStandardMaterial color={eyeColor} />
            </mesh>

            {/* Right Eye White */}
            <mesh ref={rightEyeRef} position={[0.35, 0.15, 0.85]}>
                <sphereGeometry args={[0.18, 16, 16]} />
                <meshStandardMaterial color="white" />
            </mesh>

            {/* Right Pupil */}
            <mesh ref={rightPupilRef} position={[0.35, 0.15, 1.0]}>
                <sphereGeometry args={[0.08, 16, 16]} />
                <meshStandardMaterial color={eyeColor} />
            </mesh>

            {/* Nose */}
            <mesh position={[0, -0.05, 0.95]} rotation={[0.3, 0, 0]}>
                <coneGeometry args={[0.08, 0.2, 8]} />
                <meshStandardMaterial color={skinColor} roughness={0.8} />
            </mesh>

            {/* Mouth */}
            <mesh ref={mouthRef} position={[0, -0.4, 0.9]}>
                <capsuleGeometry args={[0.08, 0.25, 8, 16]} />
                <meshStandardMaterial color={lipColor} roughness={0.6} />
            </mesh>

            {/* Ears */}
            <mesh position={[-1.0, 0, 0]}>
                <sphereGeometry args={[0.15, 8, 8]} />
                <meshStandardMaterial color={skinColor} roughness={0.8} />
            </mesh>
            <mesh position={[1.0, 0, 0]}>
                <sphereGeometry args={[0.15, 8, 8]} />
                <meshStandardMaterial color={skinColor} roughness={0.8} />
            </mesh>

            {/* Neck */}
            <mesh position={[0, -1.1, 0]}>
                <cylinderGeometry args={[0.3, 0.35, 0.5, 16]} />
                <meshStandardMaterial color={skinColor} roughness={0.8} />
            </mesh>

            {/* Shoulders hint */}
            <mesh position={[0, -1.5, 0]}>
                <boxGeometry args={[1.8, 0.4, 0.6]} />
                <meshStandardMaterial color={avatarId === 'female' ? '#6366f1' : '#3b82f6'} roughness={0.6} />
            </mesh>
        </group>
    )
}

// Loading component
function LoadingAvatar() {
    return (
        <Html center>
            <div className="text-center">
                <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                <p className="text-sm text-gray-500">Loading 3D Avatar...</p>
            </div>
        </Html>
    )
}

// Main Avatar3D component
export default function Avatar3D({ avatarId, isSpeaking = false, onReady }: Avatar3DProps) {
    const [isReady, setIsReady] = useState(false)

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsReady(true)
            onReady?.()
        }, 500)
        return () => clearTimeout(timer)
    }, [onReady])

    return (
        <div className="w-full h-full bg-gradient-to-b from-indigo-100 via-blue-50 to-slate-100 rounded-2xl overflow-hidden relative">
            <Canvas
                camera={{ position: [0, 0, 4], fov: 35 }}
                dpr={[1, 2]}
                gl={{ antialias: true, alpha: true }}
            >
                <ambientLight intensity={0.6} />
                <directionalLight position={[5, 5, 5]} intensity={0.8} />
                <directionalLight position={[-3, 2, 4]} intensity={0.4} />

                <Suspense fallback={<LoadingAvatar />}>
                    <StylizedHead isSpeaking={isSpeaking} avatarId={avatarId} />
                </Suspense>

                <Environment preset="apartment" />

                <OrbitControls
                    enableZoom={false}
                    enablePan={false}
                    minPolarAngle={Math.PI / 2.5}
                    maxPolarAngle={Math.PI / 1.8}
                    minAzimuthAngle={-Math.PI / 6}
                    maxAzimuthAngle={Math.PI / 6}
                />
            </Canvas>

            {/* Avatar name badge */}
            <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full shadow-sm">
                <span className="text-sm font-medium text-gray-700">
                    {avatarId === 'male' ? 'Alex' : 'Sara'}
                </span>
            </div>

            {/* 3D badge */}
            <div className="absolute top-4 right-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs px-2 py-0.5 rounded-full font-medium">
                3D
            </div>

            {/* Speaking indicator */}
            {isSpeaking && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1 bg-blue-500/90 backdrop-blur-sm px-3 py-1.5 rounded-full">
                    <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
            )}
        </div>
    )
}
