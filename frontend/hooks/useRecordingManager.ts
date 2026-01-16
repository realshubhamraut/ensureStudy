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
 * Hook to manage meeting recording with MediaRecorder
 * Records camera and microphone directly
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

    // Finalize recording - waits for all uploads to complete first
    const finalizeRecording = useCallback(async () => {
        const duration = Math.floor((Date.now() - startTimeRef.current) / 1000)

        // Wait for all pending uploads to complete
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
                    duration_seconds: duration
                })
            })

            if (!res.ok) {
                const errorText = await res.text()
                console.error(`[Recording] Finalize failed: ${res.status}`, errorText)
                throw new Error('Failed to finalize recording')
            }

            const data = await res.json()
            console.log('[Recording] Recording finalized successfully:', data.recording?.id)
            setState(prev => ({ ...prev, status: 'complete', progress: 100 }))

            if (onRecordingComplete && data.recording?.id) {
                onRecordingComplete(data.recording.id)
            }

            return data.recording
        } catch (error) {
            console.error('[Recording] Finalize error:', error)
            setState(prev => ({
                ...prev,
                status: 'error',
                error: 'Failed to finalize recording'
            }))
            return null
        }
    }, [meetingId, accessToken, onRecordingComplete])

    // Start recording - captures camera and microphone directly
    const startRecording = useCallback(async () => {
        console.log('[Recording] Starting recording for meeting:', meetingId)
        console.log('[Recording] Access token:', accessToken ? 'present' : 'missing')

        try {
            // Get camera and microphone
            console.log('[Recording] Requesting camera and microphone...')
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            })

            console.log('[Recording] Got media stream:', stream.getTracks().map(t => `${t.kind}: ${t.label}`))
            streamRef.current = stream

            // Create MediaRecorder
            const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
                ? 'video/webm;codecs=vp9,opus'
                : MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
                    ? 'video/webm;codecs=vp8,opus'
                    : 'video/webm'

            console.log('[Recording] Using MIME type:', mimeType)

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType,
                videoBitsPerSecond: 2500000 // 2.5 Mbps
            })

            mediaRecorderRef.current = mediaRecorder
            recordedChunksRef.current = []
            chunkIndexRef.current = 0
            uploadedChunksRef.current = 0
            pendingUploadsRef.current = []
            startTimeRef.current = Date.now()

            // Handle data available - upload chunks progressively
            mediaRecorder.ondataavailable = (event) => {
                console.log('[Recording] ondataavailable triggered, data size:', event.data?.size || 0)
                if (event.data && event.data.size > 0) {
                    recordedChunksRef.current.push(event.data)
                    const chunkIndex = chunkIndexRef.current++
                    console.log(`[Recording] Got chunk ${chunkIndex}, size: ${event.data.size}`)

                    // Track the upload promise
                    const uploadPromise = uploadChunk(event.data, chunkIndex)
                    pendingUploadsRef.current.push(uploadPromise)

                    setState(prev => ({
                        ...prev,
                        progress: Math.min(85, prev.progress + 5)
                    }))
                }
            }

            mediaRecorder.onstop = async () => {
                console.log('[Recording] MediaRecorder stopped, total chunks:', chunkIndexRef.current)

                // Finalize - this will wait for all pending uploads
                await finalizeRecording()

                // Clean up
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop())
                    streamRef.current = null
                }
            }

            mediaRecorder.onerror = (event) => {
                console.error('[Recording] MediaRecorder error:', event)
                setState(prev => ({
                    ...prev,
                    status: 'error',
                    error: 'Recording error occurred'
                }))
            }

            // Start recording with 5-second chunks
            console.log('[Recording] Starting MediaRecorder with 5s chunks...')
            mediaRecorder.start(5000)
            console.log('[Recording] MediaRecorder state:', mediaRecorder.state)

            // Start duration timer
            durationIntervalRef.current = setInterval(() => {
                setState(prev => ({
                    ...prev,
                    duration: Math.floor((Date.now() - startTimeRef.current) / 1000)
                }))
            }, 1000)

            setState({
                isRecording: true,
                isPaused: false,
                duration: 0,
                status: 'recording',
                error: null,
                progress: 0
            })

            console.log('[Recording] Recording started successfully!')

        } catch (error) {
            console.error('[Recording] Start recording error:', error)
            setState(prev => ({
                ...prev,
                status: 'error',
                error: error instanceof Error ? error.message : 'Failed to start recording'
            }))
        }
    }, [meetingId, accessToken, uploadChunk, finalizeRecording])

    // Stop recording
    const stopRecording = useCallback(() => {
        console.log('[Recording] Stopping recording...')
        console.log('[Recording] Current state:', state.isRecording)
        console.log('[Recording] MediaRecorder state:', mediaRecorderRef.current?.state)

        if (mediaRecorderRef.current && state.isRecording) {
            // Request remaining data before stopping
            if (mediaRecorderRef.current.state === 'recording') {
                console.log('[Recording] Requesting final data...')
                mediaRecorderRef.current.requestData()
            }

            mediaRecorderRef.current.stop()

            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current)
                durationIntervalRef.current = null
            }

            setState(prev => ({
                ...prev,
                isRecording: false,
                status: 'uploading'
            }))
        }
    }, [state.isRecording])

    // Pause/resume recording
    const togglePause = useCallback(() => {
        if (!mediaRecorderRef.current || !state.isRecording) return

        if (state.isPaused) {
            mediaRecorderRef.current.resume()
            setState(prev => ({ ...prev, isPaused: false }))
        } else {
            mediaRecorderRef.current.pause()
            setState(prev => ({ ...prev, isPaused: true }))
        }
    }, [state.isRecording, state.isPaused])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current)
            }
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop())
            }
        }
    }, [])

    // Format duration as mm:ss
    const formatDuration = (seconds: number): string => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }

    return {
        ...state,
        formattedDuration: formatDuration(state.duration),
        startRecording,
        stopRecording,
        togglePause
    }
}
