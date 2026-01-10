'use client'

import { useState, useRef, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import {
    XMarkIcon,
    PlayIcon,
    PauseIcon,
    SpeakerWaveIcon,
    SpeakerXMarkIcon,
    ArrowsPointingOutIcon,
    DocumentTextIcon,
    SparklesIcon,
    ClockIcon,
    ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline'

interface TranscriptSegment {
    start: number
    end: number
    text: string
}

interface Recording {
    id: string
    meeting_id: string
    storage_url: string
    duration_seconds: number
    status: string
    has_transcript: boolean
    summary_brief?: string
    meeting?: {
        id: string
        title: string
        host_name: string
        started_at: string
    }
}

interface EnhancedSessionPlayerProps {
    recording: Recording
    accessToken: string
    onClose: () => void
}

export function EnhancedSessionPlayer({ recording, accessToken, onClose }: EnhancedSessionPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [volume, setVolume] = useState(1)
    const [isMuted, setIsMuted] = useState(false)
    const [transcript, setTranscript] = useState<TranscriptSegment[]>([])
    const [summary, setSummary] = useState<string>('')
    const [loadingTranscript, setLoadingTranscript] = useState(true)
    const [activeSegmentIndex, setActiveSegmentIndex] = useState(-1)
    const transcriptRef = useRef<HTMLDivElement>(null)

    // Fetch transcript and summary
    useEffect(() => {
        const fetchData = async () => {
            setLoadingTranscript(true)
            try {
                // Fetch transcript
                const transcriptRes = await fetch(
                    `${getApiBaseUrl()}/api/recordings/${recording.id}/transcript`,
                    { headers: { Authorization: `Bearer ${accessToken}` } }
                )
                if (transcriptRes.ok) {
                    const data = await transcriptRes.json()
                    // Parse transcript into segments
                    if (data.segments) {
                        setTranscript(data.segments)
                    } else if (data.transcript) {
                        // Split text into segments by sentences
                        const segments = parseTranscriptText(data.transcript, recording.duration_seconds)
                        setTranscript(segments)
                    }
                }

                // Fetch summary
                const summaryRes = await fetch(
                    `${getApiBaseUrl()}/api/recordings/${recording.id}/summary`,
                    { headers: { Authorization: `Bearer ${accessToken}` } }
                )
                if (summaryRes.ok) {
                    const data = await summaryRes.json()
                    setSummary(data.summary || data.summary_brief || recording.summary_brief || '')
                }
            } catch (e) {
                console.error('Failed to fetch transcript/summary:', e)
            }
            setLoadingTranscript(false)
        }
        fetchData()
    }, [recording.id, accessToken])

    // Parse plain transcript text into timed segments
    const parseTranscriptText = (text: string, totalDuration: number): TranscriptSegment[] => {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0)
        const segmentDuration = totalDuration / Math.max(sentences.length, 1)

        return sentences.map((sentence, idx) => ({
            start: idx * segmentDuration,
            end: (idx + 1) * segmentDuration,
            text: sentence.trim() + '.'
        }))
    }

    // Update active segment based on current time
    useEffect(() => {
        const activeIdx = transcript.findIndex(
            seg => currentTime >= seg.start && currentTime < seg.end
        )
        if (activeIdx !== activeSegmentIndex && activeIdx >= 0) {
            setActiveSegmentIndex(activeIdx)
            // Scroll to active segment
            if (transcriptRef.current) {
                const activeElement = transcriptRef.current.children[activeIdx] as HTMLElement
                if (activeElement) {
                    activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }
            }
        }
    }, [currentTime, transcript])

    // Video event handlers
    const handleTimeUpdate = () => {
        if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime)
        }
    }

    const handleLoadedMetadata = () => {
        if (videoRef.current) {
            setDuration(videoRef.current.duration)
        }
    }

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause()
            } else {
                videoRef.current.play()
            }
            setIsPlaying(!isPlaying)
        }
    }

    const toggleMute = () => {
        if (videoRef.current) {
            videoRef.current.muted = !isMuted
            setIsMuted(!isMuted)
        }
    }

    const seekTo = (time: number) => {
        if (videoRef.current) {
            videoRef.current.currentTime = time
            setCurrentTime(time)
        }
    }

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const time = parseFloat(e.target.value)
        seekTo(time)
    }

    const formatTime = (seconds: number): string => {
        const h = Math.floor(seconds / 3600)
        const m = Math.floor((seconds % 3600) / 60)
        const s = Math.floor(seconds % 60)
        if (h > 0) {
            return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
        }
        return `${m}:${s.toString().padStart(2, '0')}`
    }

    const toggleFullscreen = () => {
        if (videoRef.current) {
            if (document.fullscreenElement) {
                document.exitFullscreen()
            } else {
                videoRef.current.requestFullscreen()
            }
        }
    }

    const streamUrl = `${getApiBaseUrl()}/api/recordings/${recording.id}/stream`

    return (
        <div className="fixed inset-0 bg-black/90 z-50 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-b from-black/80 to-transparent">
                <div>
                    <h2 className="text-xl font-semibold text-white">
                        {recording.meeting?.title || 'Recording'}
                    </h2>
                    <p className="text-sm text-gray-400">
                        {recording.meeting?.host_name && `Hosted by ${recording.meeting.host_name} • `}
                        {recording.meeting?.started_at && new Date(recording.meeting.started_at).toLocaleDateString()}
                    </p>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 hover:bg-white/10 rounded-full transition-colors"
                >
                    <XMarkIcon className="w-6 h-6 text-white" />
                </button>
            </div>

            {/* Summary Section */}
            {summary && (
                <div className="mx-6 mb-4 p-4 bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-xl border border-purple-500/30">
                    <div className="flex items-center gap-2 mb-2">
                        <SparklesIcon className="w-5 h-5 text-purple-400" />
                        <span className="text-sm font-medium text-purple-300">AI Summary</span>
                    </div>
                    <p className="text-white/90 text-sm leading-relaxed">{summary}</p>
                </div>
            )}

            {/* Main Content: Video + Transcript */}
            <div className="flex-1 flex px-6 pb-4 gap-4 min-h-0">
                {/* Video Player */}
                <div className="flex-1 flex flex-col min-w-0">
                    <div className="relative flex-1 bg-black rounded-xl overflow-hidden">
                        <video
                            ref={videoRef}
                            src={streamUrl}
                            className="w-full h-full object-contain"
                            onTimeUpdate={handleTimeUpdate}
                            onLoadedMetadata={handleLoadedMetadata}
                            onPlay={() => setIsPlaying(true)}
                            onPause={() => setIsPlaying(false)}
                            onClick={togglePlay}
                        />

                        {/* Play overlay when paused */}
                        {!isPlaying && (
                            <div
                                className="absolute inset-0 flex items-center justify-center bg-black/30 cursor-pointer"
                                onClick={togglePlay}
                            >
                                <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm">
                                    <PlayIcon className="w-10 h-10 text-white ml-1" />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Video Controls */}
                    <div className="mt-3 bg-gray-900/80 rounded-xl p-3">
                        {/* Progress bar */}
                        <div className="flex items-center gap-3 mb-2">
                            <span className="text-xs text-gray-400 w-12">{formatTime(currentTime)}</span>
                            <input
                                type="range"
                                min={0}
                                max={duration || 100}
                                value={currentTime}
                                onChange={handleSeek}
                                className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-primary-500 [&::-webkit-slider-thumb]:rounded-full"
                            />
                            <span className="text-xs text-gray-400 w-12">{formatTime(duration)}</span>
                        </div>

                        {/* Control buttons */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={togglePlay}
                                    className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                >
                                    {isPlaying ? (
                                        <PauseIcon className="w-6 h-6 text-white" />
                                    ) : (
                                        <PlayIcon className="w-6 h-6 text-white" />
                                    )}
                                </button>

                                <button
                                    onClick={toggleMute}
                                    className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                >
                                    {isMuted ? (
                                        <SpeakerXMarkIcon className="w-5 h-5 text-white" />
                                    ) : (
                                        <SpeakerWaveIcon className="w-5 h-5 text-white" />
                                    )}
                                </button>
                            </div>

                            <button
                                onClick={toggleFullscreen}
                                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                            >
                                <ArrowsPointingOutIcon className="w-5 h-5 text-white" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Transcript Panel */}
                <div className="w-96 flex flex-col bg-gray-900/80 rounded-xl overflow-hidden">
                    <div className="p-4 border-b border-gray-700/50 flex items-center gap-2">
                        <DocumentTextIcon className="w-5 h-5 text-primary-400" />
                        <span className="font-medium text-white">Transcript</span>
                        {transcript.length > 0 && (
                            <span className="text-xs text-gray-400 ml-auto">
                                {transcript.length} segments
                            </span>
                        )}
                    </div>

                    <div
                        ref={transcriptRef}
                        className="flex-1 overflow-y-auto p-4 space-y-2"
                    >
                        {loadingTranscript ? (
                            <div className="flex items-center justify-center py-8">
                                <div className="animate-spin rounded-full h-6 w-6 border-2 border-primary-500 border-t-transparent" />
                            </div>
                        ) : transcript.length === 0 ? (
                            <div className="text-center py-8 text-gray-500">
                                <DocumentTextIcon className="w-10 h-10 mx-auto mb-2 opacity-50" />
                                <p>No transcript available</p>
                            </div>
                        ) : (
                            transcript.map((segment, idx) => (
                                <div
                                    key={idx}
                                    onClick={() => seekTo(segment.start)}
                                    className={`p-3 rounded-lg cursor-pointer transition-all ${idx === activeSegmentIndex
                                            ? 'bg-primary-600/30 border border-primary-500/50'
                                            : 'bg-gray-800/50 hover:bg-gray-700/50'
                                        }`}
                                >
                                    <div className="flex items-center gap-2 mb-1">
                                        <ClockIcon className="w-3.5 h-3.5 text-gray-400" />
                                        <span className="text-xs text-primary-400 font-mono">
                                            {formatTime(segment.start)}
                                        </span>
                                    </div>
                                    <p className={`text-sm ${idx === activeSegmentIndex ? 'text-white' : 'text-gray-300'
                                        }`}>
                                        {segment.text}
                                    </p>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
