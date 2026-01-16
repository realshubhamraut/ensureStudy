'use client'

import { useState, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import { MeetingPlayer } from './MeetingPlayer'
import {
    VideoCameraIcon,
    PlayIcon,
    ClockIcon,
    UserIcon,
    CalendarIcon
} from '@heroicons/react/24/outline'
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

/**
 * List of recordings for a classroom
 * Shows all past meeting recordings with VOD playback
 */
export function RecordingsList({ classroomId, accessToken }: RecordingsListProps) {
    const [recordings, setRecordings] = useState<Recording[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null)

    // Fetch recordings
    useEffect(() => {
        async function fetchRecordings() {
            try {
                setLoading(true)
                const res = await fetch(
                    `${getApiBaseUrl()}/api/recordings/classroom/${classroomId}`,
                    {
                        headers: { 'Authorization': `Bearer ${accessToken}` }
                    }
                )

                if (!res.ok) {
                    throw new Error('Failed to fetch recordings')
                }

                const data = await res.json()
                setRecordings(data.recordings || [])
            } catch (err) {
                console.error('Error fetching recordings:', err)
                setError(err instanceof Error ? err.message : 'Failed to load recordings')
            } finally {
                setLoading(false)
            }
        }

        if (classroomId && accessToken) {
            fetchRecordings()
        }
    }, [classroomId, accessToken])

    // Format duration
    const formatDuration = (seconds: number): string => {
        const h = Math.floor(seconds / 3600)
        const m = Math.floor((seconds % 3600) / 60)

        if (h > 0) {
            return `${h}h ${m}m`
        }
        return `${m} min`
    }

    // Format file size
    const formatFileSize = (bytes: number): string => {
        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`
        }
        if (bytes < 1024 * 1024 * 1024) {
            return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
        }
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
    }

    // Format date
    const formatDate = (dateStr: string): string => {
        const date = new Date(dateStr.endsWith('Z') ? dateStr : dateStr + 'Z')
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        })
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary-500 border-t-transparent" />
            </div>
        )
    }

    if (error) {
        return (
            <div className="text-center py-12 text-red-500">
                <p>{error}</p>
            </div>
        )
    }

    if (recordings.length === 0) {
        return (
            <div className="text-center py-12">
                <VideoCameraIcon className="w-16 h-16 mx-auto text-gray-300 mb-4" />
                <h3 className="text-lg font-medium text-gray-700 mb-2">No recordings yet</h3>
                <p className="text-gray-500">Past meeting recordings will appear here</p>
            </div>
        )
    }

    return (
        <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recordings.map(recording => (
                    <div
                        key={recording.id}
                        className={clsx(
                            'card overflow-hidden cursor-pointer hover:shadow-lg transition-shadow',
                            recording.status !== 'ready' && 'opacity-70'
                        )}
                        onClick={() => recording.status === 'ready' && setSelectedRecording(recording)}
                    >
                        {/* Thumbnail placeholder */}
                        <div className="aspect-video bg-gradient-to-br from-gray-700 to-gray-900 relative flex items-center justify-center">
                            <VideoCameraIcon className="w-12 h-12 text-gray-500" />

                            {/* Duration badge */}
                            <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                                {formatDuration(recording.duration_seconds)}
                            </div>

                            {/* Play button overlay */}
                            {recording.status === 'ready' && (
                                <div className="absolute inset-0 bg-black/30 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
                                    <div className="w-14 h-14 bg-white/20 rounded-full flex items-center justify-center">
                                        <PlayIcon className="w-7 h-7 text-white ml-1" />
                                    </div>
                                </div>
                            )}

                            {/* Processing status */}
                            {recording.status === 'processing' && (
                                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                                    <div className="text-center text-white">
                                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent mx-auto mb-2" />
                                        <span className="text-sm">Processing...</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Info */}
                        <div className="p-4">
                            {/* Title row with date on the right */}
                            <div className="flex items-start justify-between gap-2 mb-2">
                                <h3 className="font-medium text-gray-900 line-clamp-2 flex-1">
                                    {recording.meeting?.title || 'Meeting Recording'}
                                </h3>
                                <div className="flex items-center gap-1 text-xs text-gray-400 shrink-0">
                                    <CalendarIcon className="w-3.5 h-3.5" />
                                    <span>{formatDate(recording.created_at)}</span>
                                </div>
                            </div>

                            <div className="space-y-1 text-sm text-gray-500">
                                {recording.meeting?.host_name && (
                                    <div className="flex items-center gap-2">
                                        <UserIcon className="w-4 h-4" />
                                        <span>{recording.meeting.host_name}</span>
                                    </div>
                                )}
                                <div className="flex items-center gap-2">
                                    <ClockIcon className="w-4 h-4" />
                                    <span>{formatDuration(recording.duration_seconds)} • {formatFileSize(recording.file_size)}</span>
                                </div>
                            </div>

                            {/* Transcript badge */}
                            {recording.has_transcript && (
                                <div className="mt-3">
                                    <span className="inline-flex items-center px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                                        Transcript Available
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {/* Video player modal */}
            {selectedRecording && (
                <MeetingPlayer
                    videoUrl={selectedRecording.storage_url}
                    title={selectedRecording.meeting?.title || 'Meeting Recording'}
                    duration={selectedRecording.duration_seconds}
                    transcript={[]} // Will be populated when transcription is ready
                    onClose={() => setSelectedRecording(null)}
                />
            )}
        </>
    )
}
