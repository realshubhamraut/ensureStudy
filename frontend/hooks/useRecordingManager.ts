'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'

interface RecordingState {
    isRecording: boolean
    isPaused: boolean
    duration: number
    status: 'idle' | 'recording' | 'uploading' | 'processing' | 'complete' | 'error'
    error: string | null
    progress: number
}

interface UseRecordingManagerProps {
    meetingId: string
    roomId: string
    accessToken: string
    onRecordingComplete?: (recordingId: string) => void
}

/**
 * Hook to manage meeting recording
 * 
 * Approach: Capture all video elements on the meeting page and composite them
 * This works like Zoom - no screen share prompts, records what the host sees
 */
export function useRecordingManager({
    meetingId,
    roomId,
    accessToken,
    onRecordingComplete
}: UseRecordingManagerProps) {
    const [state, setState] = useState<RecordingState>({
        isRecording: false,
        isPaused: false,
        duration: 0,
        status: 'idle',
        error: null,
        progress: 0
    })

    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const recordedChunksRef = useRef<Blob[]>([])
    const chunkIndexRef = useRef(0)
    const startTimeRef = useRef<number>(0)
    const durationIntervalRef = useRef<NodeJS.Timeout | null>(null)
    const streamRef = useRef<MediaStream | null>(null)
    const pendingUploadsRef = useRef<Promise<boolean>[]>([])
    const uploadedChunksRef = useRef(0)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const animationFrameRef = useRef<number | null>(null)

    // Upload a single chunk
    const uploadChunk = useCallback(async (chunk: Blob, index: number, isFinal: boolean = false): Promise<boolean> => {
        console.log(`[Recording] Uploading chunk ${index}, size: ${chunk.size} bytes, final: ${isFinal}`)

        const formData = new FormData()
        formData.append('meeting_id', meetingId)
        formData.append('chunk_index', index.toString())
        formData.append('is_final', isFinal.toString())
        formData.append('chunk', chunk, `chunk_${index}.webm`)

        try {
            const apiUrl = `${getApiBaseUrl()}/api/recordings/upload-chunk`

            const res = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`
                },
                body: formData
            })

            if (!res.ok) {
                const errorText = await res.text()
                console.error(`[Recording] Chunk upload failed: ${res.status}`, errorText)
                return false
            }

            uploadedChunksRef.current++
            console.log(`[Recording] Chunk ${index} uploaded successfully (${uploadedChunksRef.current} total)`)
            return true
        } catch (error) {
            console.error('[Recording] Chunk upload error:', error)
            return false
        }
    }, [meetingId, accessToken])

    // Finalize recording
    const finalizeRecording = useCallback(async () => {
        const duration = Math.floor((Date.now() - startTimeRef.current) / 1000)

        console.log(`[Recording] Waiting for ${pendingUploadsRef.current.length} pending uploads...`)
        await Promise.all(pendingUploadsRef.current)
        console.log(`[Recording] All uploads complete. Uploaded chunks: ${uploadedChunksRef.current}`)

        console.log(`[Recording] Finalizing recording. Duration: ${duration}s, Total chunks: ${uploadedChunksRef.current}`)

        try {
            setState(prev => ({ ...prev, status: 'uploading', progress: 90 }))

            const res = await fetch(`${getApiBaseUrl()}/api/recordings/finalize`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    meeting_id: meetingId,
                    total_chunks: uploadedChunksRef.current,
                    duration,
                    title: `Recording - ${new Date().toLocaleString()}`
                })
            })

            if (!res.ok) {
                const errorText = await res.text()
                throw new Error(`Finalize failed: ${errorText}`)
            }

            const data = await res.json()
            console.log('[Recording] Recording finalized:', data)

            setState(prev => ({ ...prev, status: 'complete', progress: 100 }))

            if (onRecordingComplete && data.recording_id) {
                onRecordingComplete(data.recording_id)
            }

            // Reset after 3 seconds
            setTimeout(() => {
                setState({
                    isRecording: false,
                    isPaused: false,
                    duration: 0,
                    status: 'idle',
                    error: null,
                    progress: 0
                })
            }, 3000)

        } catch (error) {
            console.error('[Recording] Finalize error:', error)
            setState(prev => ({
                ...prev,
                status: 'error',
                error: error instanceof Error ? error.message : 'Failed to save recording'
            }))
        }
    }, [meetingId, accessToken, onRecordingComplete])

    /**
     * Start recording by capturing video elements and compositing them
     * This avoids the screen share prompt
     */
    const startRecording = useCallback(async () => {
        try {
            console.log('[Recording] Starting composite recording...')
            setState(prev => ({ ...prev, status: 'recording', error: null }))

            // Find all video elements in the meeting (LiveKit renders videos)
            const videoElements = document.querySelectorAll('video') as NodeListOf<HTMLVideoElement>
            console.log(`[Recording] Found ${videoElements.length} video elements`)

            // Create a canvas to composite all videos
            const canvas = document.createElement('canvas')
            canvas.width = 1920
            canvas.height = 1080
            canvasRef.current = canvas
            const ctx = canvas.getContext('2d')!

            // Get audio from all videos
            const audioContext = new AudioContext()
            const destination = audioContext.createMediaStreamDestination()

            // Connect audio from all video elements
            let hasAudio = false
            videoElements.forEach((video, i) => {
                if (video.srcObject instanceof MediaStream) {
                    const audioTracks = (video.srcObject as MediaStream).getAudioTracks()
                    if (audioTracks.length > 0) {
                        try {
                            const source = audioContext.createMediaStreamSource(video.srcObject as MediaStream)
                            source.connect(destination)
                            hasAudio = true
                            console.log(`[Recording] Connected audio from video ${i}`)
                        } catch (e) {
                            console.log(`[Recording] Could not connect audio from video ${i}:`, e)
                        }
                    }
                }
            })

            // Also capture user's microphone
            try {
                const micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
                const micSource = audioContext.createMediaStreamSource(micStream)
                micSource.connect(destination)
                hasAudio = true
                console.log('[Recording] Connected microphone audio')
            } catch (e) {
                console.log('[Recording] Could not get microphone:', e)
            }

            // Animation loop to draw all videos to canvas
            const drawFrame = () => {
                ctx.fillStyle = '#1f2937' // Dark gray background
                ctx.fillRect(0, 0, canvas.width, canvas.height)

                const videos = Array.from(document.querySelectorAll('video') as NodeListOf<HTMLVideoElement>)
                    .filter(v => v.videoWidth > 0 && v.videoHeight > 0)

                if (videos.length === 0) {
                    // No videos, just show a placeholder
                    ctx.fillStyle = '#6b7280'
                    ctx.font = '48px sans-serif'
                    ctx.textAlign = 'center'
                    ctx.fillText('Recording...', canvas.width / 2, canvas.height / 2)
                } else if (videos.length === 1) {
                    // Single video - full screen
                    const video = videos[0]
                    const aspectRatio = video.videoWidth / video.videoHeight
                    let drawWidth = canvas.width
                    let drawHeight = canvas.width / aspectRatio
                    if (drawHeight > canvas.height) {
                        drawHeight = canvas.height
                        drawWidth = canvas.height * aspectRatio
                    }
                    const x = (canvas.width - drawWidth) / 2
                    const y = (canvas.height - drawHeight) / 2
                    ctx.drawImage(video, x, y, drawWidth, drawHeight)
                } else {
                    // Multiple videos - grid layout
                    const cols = Math.ceil(Math.sqrt(videos.length))
                    const rows = Math.ceil(videos.length / cols)
                    const cellWidth = canvas.width / cols
                    const cellHeight = canvas.height / rows

                    videos.forEach((video, index) => {
                        const col = index % cols
                        const row = Math.floor(index / cols)
                        const x = col * cellWidth
                        const y = row * cellHeight

                        // Draw video maintaining aspect ratio
                        const aspectRatio = video.videoWidth / video.videoHeight
                        let drawWidth = cellWidth
                        let drawHeight = cellWidth / aspectRatio
                        if (drawHeight > cellHeight) {
                            drawHeight = cellHeight
                            drawWidth = cellHeight * aspectRatio
                        }
                        const offsetX = x + (cellWidth - drawWidth) / 2
                        const offsetY = y + (cellHeight - drawHeight) / 2

                        ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight)
                    })
                }

                // Recording indicator
                ctx.fillStyle = '#ef4444'
                ctx.beginPath()
                ctx.arc(canvas.width - 50, 50, 15, 0, Math.PI * 2)
                ctx.fill()
                ctx.fillStyle = 'white'
                ctx.font = '20px sans-serif'
                ctx.textAlign = 'right'
                ctx.fillText('REC', canvas.width - 75, 57)

                if (mediaRecorderRef.current?.state === 'recording') {
                    animationFrameRef.current = requestAnimationFrame(drawFrame)
                }
            }

            // Create combined stream: canvas video + audio
            const canvasStream = canvas.captureStream(30) // 30 FPS

            // Add audio tracks to the stream
            if (hasAudio) {
                destination.stream.getAudioTracks().forEach(track => {
                    canvasStream.addTrack(track)
                })
            }

            streamRef.current = canvasStream

            // Setup MediaRecorder
            const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
                ? 'video/webm;codecs=vp9,opus'
                : MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
                    ? 'video/webm;codecs=vp8,opus'
                    : 'video/webm'

            const mediaRecorder = new MediaRecorder(canvasStream, {
                mimeType,
                videoBitsPerSecond: 2500000 // 2.5 Mbps
            })

            mediaRecorderRef.current = mediaRecorder
            recordedChunksRef.current = []
            chunkIndexRef.current = 0
            uploadedChunksRef.current = 0
            pendingUploadsRef.current = []
            startTimeRef.current = Date.now()

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    recordedChunksRef.current.push(event.data)
                    const currentIndex = chunkIndexRef.current++
                    console.log(`[Recording] Got chunk ${currentIndex}, size: ${event.data.size}`)

                    // Upload chunk in background
                    const uploadPromise = uploadChunk(event.data, currentIndex)
                    pendingUploadsRef.current.push(uploadPromise)
                }
            }

            mediaRecorder.onstop = async () => {
                console.log('[Recording] MediaRecorder stopped')
                if (animationFrameRef.current) {
                    cancelAnimationFrame(animationFrameRef.current)
                }
                await finalizeRecording()
            }

            mediaRecorder.onerror = (event) => {
                console.error('[Recording] MediaRecorder error:', event)
                setState(prev => ({
                    ...prev,
                    status: 'error',
                    error: 'Recording failed'
                }))
            }

            // Start recording
            mediaRecorder.start(5000) // Chunk every 5 seconds
            drawFrame() // Start animation loop

            // Duration timer
            durationIntervalRef.current = setInterval(() => {
                setState(prev => ({
                    ...prev,
                    duration: Math.floor((Date.now() - startTimeRef.current) / 1000)
                }))
            }, 1000)

            setState(prev => ({ ...prev, isRecording: true, status: 'recording' }))
            console.log('[Recording] Started successfully - capturing all participants')

        } catch (error) {
            console.error('[Recording] Start error:', error)
            setState(prev => ({
                ...prev,
                status: 'error',
                error: error instanceof Error ? error.message : 'Failed to start recording'
            }))
        }
    }, [uploadChunk, finalizeRecording])

    // Stop recording
    const stopRecording = useCallback(() => {
        console.log('[Recording] Stopping...')

        if (durationIntervalRef.current) {
            clearInterval(durationIntervalRef.current)
            durationIntervalRef.current = null
        }

        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current)
        }

        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
        }

        setState(prev => ({ ...prev, isRecording: false, isPaused: false }))
    }, [])

    // Toggle pause
    const togglePause = useCallback(() => {
        if (!mediaRecorderRef.current) return

        if (state.isPaused) {
            mediaRecorderRef.current.resume()
            setState(prev => ({ ...prev, isPaused: false }))
        } else {
            mediaRecorderRef.current.pause()
            setState(prev => ({ ...prev, isPaused: true }))
        }
    }, [state.isPaused])

    // Format duration
    const formatDuration = (seconds: number): string => {
        const hrs = Math.floor(seconds / 3600)
        const mins = Math.floor((seconds % 3600) / 60)
        const secs = seconds % 60
        if (hrs > 0) {
            return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current)
            }
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current)
            }
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop())
            }
        }
    }, [])

    return {
        isRecording: state.isRecording,
        isPaused: state.isPaused,
        duration: state.duration,
        formattedDuration: formatDuration(state.duration),
        status: state.status,
        error: state.error,
        progress: state.progress,
        startRecording,
        stopRecording,
        togglePause
    }
}
