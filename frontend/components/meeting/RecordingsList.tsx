'use client'

import { useState, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import { EnhancedSessionPlayer } from './EnhancedSessionPlayer'
import {
    VideoCameraIcon,
    PlayIcon,
    DocumentTextIcon,
    SparklesIcon
} from '@heroicons/react/24/solid'
import clsx from 'clsx'

interface Recording {
    id: string
    meeting_id: string
    storage_url: string
    file_size: number
    duration_seconds: number
    status: string
    has_transcript: boolean
    summary_brief?: string
    created_at: string
    meeting?: {
        id: string
        title: string
        host_name: string
        started_at: string
        ended_at: string
    }
}

interface RecordingsListProps {
    classroomId: string
    accessToken: string
}

export function RecordingsList({ classroomId, accessToken }: RecordingsListProps) {
    const [recordings, setRecordings] = useState<Recording[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null)

    useEffect(() => {
        const fetchRecordings = async () => {
            try {
                const res = await fetch(
                    `${getApiBaseUrl()}/api/recordings/classroom/${classroomId}`,
                    { headers: { Authorization: `Bearer ${accessToken}` } }
                )
                if (res.ok) {
                    const data = await res.json()
                    setRecordings(data.recordings || [])
                } else {
                    setError('Failed to load recordings')
                }
            } catch (e) {
                console.error('Failed to fetch recordings:', e)
                setError('Failed to load recordings')
            } finally {
                setLoading(false)
            }
        }
        if (accessToken) fetchRecordings()
    }, [classroomId, accessToken])

    const formatDuration = (seconds: number): string => {
        if (!seconds) return '0:00'
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-primary-500 border-t-transparent" />
            </div>
        )
    }

    if (error) {
        return <div className="text-center py-8 text-red-500 text-sm">{error}</div>
    }

    if (recordings.length === 0) {
        return (
            <div className="text-center py-8">
                <VideoCameraIcon className="w-10 h-10 mx-auto text-gray-300 mb-2" />
                <p className="text-gray-500 text-sm">No recordings yet</p>
            </div>
        )
    }

    return (
        <>
            <div className="space-y-3">
                {recordings.map(recording => {
                    const apiBase = getApiBaseUrl()
                    const streamUrl = `${apiBase}/api/recordings/${recording.id}/stream`
                    console.log('[Recording] API Base:', apiBase, 'Stream URL:', streamUrl)

                    return (
                        <div
                            key={recording.id}
                            className={clsx(
                                'bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-md transition-all',
                                recording.status !== 'ready' && 'opacity-60'
                            )}
                        >
                            {/* Compact Single Row */}
                            <div className="flex">
                                {/* Video Thumbnail with Play Button */}
                                <div
                                    className="relative w-44 h-24 shrink-0 bg-black cursor-pointer group"
                                    onClick={() => recording.status === 'ready' && setSelectedRecording(recording)}
                                >
                                    {/* Video Preview */}
                                    <video
                                        src={streamUrl}
                                        className="w-full h-full object-cover"
                                        muted
                                        preload="metadata"
                                        onLoadedMetadata={(e) => {
                                            // Seek to 1 second to get a better thumbnail frame
                                            (e.target as HTMLVideoElement).currentTime = 1
                                        }}
                                    />

                                    {/* Play Button Overlay */}
                                    <div className="absolute inset-0 bg-black/40 flex items-center justify-center group-hover:bg-black/50 transition-colors">
                                        <div className="w-10 h-10 bg-white/90 rounded-full flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                                            <PlayIcon className="w-5 h-5 text-gray-800 ml-0.5" />
                                        </div>
                                    </div>

                                    {/* Duration Badge */}
                                    <div className="absolute bottom-1.5 right-1.5 bg-black/80 text-white text-[10px] font-medium px-1.5 py-0.5 rounded">
                                        {formatDuration(recording.duration_seconds)}
                                    </div>
                                </div>

                                {/* Content */}
                                <div className="flex-1 flex min-w-0">
                                    {/* Left: Title + Transcript */}
                                    <div className="flex-1 p-3 border-r border-gray-100 min-w-0">
                                        <h3 className="font-medium text-gray-900 truncate text-sm">
                                            {recording.meeting?.title || 'Recording'}
                                        </h3>
                                        <p className="text-xs text-green-600 mt-0.5">Recording Available</p>

                                        <button
                                            onClick={() => recording.status === 'ready' && setSelectedRecording(recording)}
                                            className="mt-2 flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
                                        >
                                            <DocumentTextIcon className="w-3.5 h-3.5" />
                                            <span>Full Transcript</span>
                                        </button>
                                    </div>

                                    {/* Right: Summary + AI Button */}
                                    <div className="flex-1 p-3 min-w-0">
                                        <div className="flex items-center gap-1 text-xs text-gray-500 mb-1">
                                            <SparklesIcon className="w-3 h-3 text-purple-500" />
                                            <span>Summary</span>
                                        </div>
                                        <p className="text-xs text-gray-600 line-clamp-2 mb-2">
                                            {recording.summary_brief || 'No summary available...'}
                                        </p>
                                        <button
                                            onClick={() => recording.status === 'ready' && setSelectedRecording(recording)}
                                            className="w-full flex items-center justify-center gap-1 px-2 py-1.5 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-md text-xs font-medium transition-all"
                                        >
                                            <SparklesIcon className="w-3 h-3" />
                                            Ask Questions (AI)
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>

            {selectedRecording && (
                <EnhancedSessionPlayer
                    recording={selectedRecording}
                    accessToken={accessToken}
                    onClose={() => setSelectedRecording(null)}
                />
            )}
        </>
    )
}
