'use client'

import { useState, useRef, useEffect } from 'react'
import {
    PlayIcon,
    PauseIcon,
    SpeakerWaveIcon,
    SpeakerXMarkIcon,
    ArrowsPointingOutIcon,
    ArrowsPointingInIcon,
    ForwardIcon,
    BackwardIcon,
    XMarkIcon
} from '@heroicons/react/24/solid'
import clsx from 'clsx'

interface TranscriptSegment {
    id: number
    start: number
    end: number
    speaker_id: number
    speaker_name?: string
    text: string
}

interface MeetingPlayerProps {
    videoUrl: string
    title: string
    duration: number // seconds
    transcript?: TranscriptSegment[]
    onClose?: () => void
    onSeekToTimestamp?: (timestamp: number) => void
}

/**
 * Meeting recording player with transcript sidebar
 * Supports seeking by clicking transcript segments
 */
export function MeetingPlayer({
    videoUrl,
    title,
    duration,
    transcript = [],
    onClose
}: MeetingPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [volume, setVolume] = useState(1)
    const [isMuted, setIsMuted] = useState(false)
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [playbackRate, setPlaybackRate] = useState(1)
    const [activeSegmentId, setActiveSegmentId] = useState<number | null>(null)
    const [showTranscript, setShowTranscript] = useState(true)
    const transcriptRef = useRef<HTMLDivElement>(null)

    // Format time as mm:ss or hh:mm:ss
    const formatTime = (seconds: number): string => {
        const h = Math.floor(seconds / 3600)
        const m = Math.floor((seconds % 3600) / 60)
        const s = Math.floor(seconds % 60)

        if (h > 0) {
            return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
        }
        return `${m}:${s.toString().padStart(2, '0')}`
    }

    // Update current time and active segment
    useEffect(() => {
        const video = videoRef.current
        if (!video) return

        const handleTimeUpdate = () => {
            setCurrentTime(video.currentTime)

            // Find active transcript segment
            const active = transcript.find(
                seg => video.currentTime >= seg.start && video.currentTime < seg.end
            )
            if (active) {
                setActiveSegmentId(active.id)
            }
        }

        video.addEventListener('timeupdate', handleTimeUpdate)
        return () => video.removeEventListener('timeupdate', handleTimeUpdate)
    }, [transcript])

    // Auto-scroll transcript
    useEffect(() => {
        if (activeSegmentId === null || !transcriptRef.current) return

        const activeElement = transcriptRef.current.querySelector(
            `[data-segment-id="${activeSegmentId}"]`
        )
        if (activeElement) {
            activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
    }, [activeSegmentId])

    // Play/pause toggle
    const togglePlay = () => {
        const video = videoRef.current
        if (!video) return

        if (video.paused) {
            video.play()
            setIsPlaying(true)
        } else {
            video.pause()
            setIsPlaying(false)
        }
    }

    // Seek to timestamp
    const seekTo = (time: number) => {
        const video = videoRef.current
        if (!video) return
        video.currentTime = time
    }

    // Seek by clicking progress bar
    const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const percent = (e.clientX - rect.left) / rect.width
        seekTo(percent * duration)
    }

    // Volume change
    const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value)
        setVolume(value)
        if (videoRef.current) {
            videoRef.current.volume = value
        }
        setIsMuted(value === 0)
    }

    // Toggle mute
    const toggleMute = () => {
        if (videoRef.current) {
            videoRef.current.muted = !isMuted
            setIsMuted(!isMuted)
        }
    }

    // Playback rate change
    const handlePlaybackRateChange = (rate: number) => {
        setPlaybackRate(rate)
        if (videoRef.current) {
            videoRef.current.playbackRate = rate
        }
    }

    // Skip forward/backward
    const skip = (seconds: number) => {
        const video = videoRef.current
        if (!video) return
        video.currentTime = Math.max(0, Math.min(duration, video.currentTime + seconds))
    }

    // Toggle fullscreen
    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen()
            setIsFullscreen(true)
        } else {
            document.exitFullscreen()
            setIsFullscreen(false)
        }
    }

    // Get speaker color
    const getSpeakerColor = (speakerId: number) => {
        const colors = [
            'bg-blue-500', 'bg-green-500', 'bg-purple-500',
            'bg-orange-500', 'bg-pink-500', 'bg-teal-500'
        ]
        return colors[speakerId % colors.length]
    }

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0

    return (
        <div className="fixed inset-0 bg-black z-50 flex">
            {/* Video area */}
            <div className={clsx(
                'flex flex-col',
                showTranscript && transcript.length > 0 ? 'flex-1' : 'w-full'
            )}>
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800">
                    <h2 className="text-white font-medium truncate">{title}</h2>
                    <div className="flex items-center gap-2">
                        {transcript.length > 0 && (
                            <button
                                onClick={() => setShowTranscript(!showTranscript)}
                                className="p-2 text-gray-400 hover:text-white rounded"
                            >
                                {showTranscript ? 'Hide' : 'Show'} Transcript
                            </button>
                        )}
                        {onClose && (
                            <button
                                onClick={onClose}
                                className="p-2 text-gray-400 hover:text-white rounded"
                            >
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        )}
                    </div>
                </div>

                {/* Video container */}
                <div className="flex-1 relative bg-black flex items-center justify-center">
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        className="max-h-full max-w-full"
                        onClick={togglePlay}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                        onEnded={() => setIsPlaying(false)}
                    />

                    {/* Play overlay */}
                    {!isPlaying && (
                        <button
                            onClick={togglePlay}
                            className="absolute inset-0 flex items-center justify-center bg-black/30"
                        >
                            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center">
                                <PlayIcon className="w-8 h-8 text-white ml-1" />
                            </div>
                        </button>
                    )}
                </div>

                {/* Controls */}
                <div className="bg-gray-900 px-4 py-3">
                    {/* Progress bar */}
                    <div
                        className="h-1 bg-gray-700 rounded cursor-pointer mb-3"
                        onClick={handleProgressClick}
                    >
                        <div
                            className="h-full bg-primary-500 rounded"
                            style={{ width: `${progress}%` }}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        {/* Left controls */}
                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => skip(-10)}
                                className="p-2 text-gray-400 hover:text-white rounded"
                            >
                                <BackwardIcon className="w-5 h-5" />
                            </button>

                            <button
                                onClick={togglePlay}
                                className="p-2 bg-white text-black rounded-full"
                            >
                                {isPlaying ? (
                                    <PauseIcon className="w-6 h-6" />
                                ) : (
                                    <PlayIcon className="w-6 h-6 ml-0.5" />
                                )}
                            </button>

                            <button
                                onClick={() => skip(10)}
                                className="p-2 text-gray-400 hover:text-white rounded"
                            >
                                <ForwardIcon className="w-5 h-5" />
                            </button>

                            {/* Time display */}
                            <span className="text-gray-400 text-sm font-mono">
                                {formatTime(currentTime)} / {formatTime(duration)}
                            </span>
                        </div>

                        {/* Right controls */}
                        <div className="flex items-center gap-3">
                            {/* Volume */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={toggleMute}
                                    className="p-2 text-gray-400 hover:text-white rounded"
                                >
                                    {isMuted || volume === 0 ? (
                                        <SpeakerXMarkIcon className="w-5 h-5" />
                                    ) : (
                                        <SpeakerWaveIcon className="w-5 h-5" />
                                    )}
                                </button>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    value={isMuted ? 0 : volume}
                                    onChange={handleVolumeChange}
                                    className="w-20 accent-primary-500"
                                />
                            </div>

                            {/* Playback speed */}
                            <select
                                value={playbackRate}
                                onChange={(e) => handlePlaybackRateChange(parseFloat(e.target.value))}
                                className="bg-gray-800 text-white text-sm rounded px-2 py-1"
                            >
                                <option value={0.5}>0.5x</option>
                                <option value={0.75}>0.75x</option>
                                <option value={1}>1x</option>
                                <option value={1.25}>1.25x</option>
                                <option value={1.5}>1.5x</option>
                                <option value={2}>2x</option>
                            </select>

                            {/* Fullscreen */}
                            <button
                                onClick={toggleFullscreen}
                                className="p-2 text-gray-400 hover:text-white rounded"
                            >
                                {isFullscreen ? (
                                    <ArrowsPointingInIcon className="w-5 h-5" />
                                ) : (
                                    <ArrowsPointingOutIcon className="w-5 h-5" />
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Transcript sidebar */}
            {showTranscript && transcript.length > 0 && (
                <div className="w-96 bg-gray-900 border-l border-gray-800 flex flex-col">
                    <div className="px-4 py-3 border-b border-gray-800">
                        <h3 className="text-white font-medium">Transcript</h3>
                    </div>
                    <div
                        ref={transcriptRef}
                        className="flex-1 overflow-y-auto p-4 space-y-3"
                    >
                        {transcript.map(segment => (
                            <button
                                key={segment.id}
                                data-segment-id={segment.id}
                                onClick={() => seekTo(segment.start)}
                                className={clsx(
                                    'w-full text-left p-3 rounded-lg transition-colors',
                                    activeSegmentId === segment.id
                                        ? 'bg-primary-600/20 border border-primary-500'
                                        : 'bg-gray-800 hover:bg-gray-700'
                                )}
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span className={clsx(
                                        'w-2 h-2 rounded-full',
                                        getSpeakerColor(segment.speaker_id)
                                    )} />
                                    <span className="text-sm font-medium text-gray-300">
                                        {segment.speaker_name || `Speaker ${segment.speaker_id + 1}`}
                                    </span>
                                    <span className="text-xs text-gray-500 font-mono ml-auto">
                                        {formatTime(segment.start)}
                                    </span>
                                </div>
                                <p className="text-gray-400 text-sm">{segment.text}</p>
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
