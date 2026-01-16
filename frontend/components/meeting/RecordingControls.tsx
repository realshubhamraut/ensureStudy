'use client'

import { useRecordingManager } from '@/hooks/useRecordingManager'
import {
    StopCircleIcon,
    PauseCircleIcon,
    PlayCircleIcon,
    SignalIcon,
    ExclamationCircleIcon,
    CheckCircleIcon
} from '@heroicons/react/24/solid'
import clsx from 'clsx'

interface RecordingControlsProps {
    meetingId: string
    roomId: string
    accessToken: string
    isHost: boolean
    onRecordingComplete?: (recordingId: string) => void
}

/**
 * Recording controls component for meeting room
 * Shows record button for host, recording status for all
 */
export function RecordingControls({
    meetingId,
    roomId,
    accessToken,
    isHost,
    onRecordingComplete
}: RecordingControlsProps) {
    const {
        isRecording,
        isPaused,
        status,
        error,
        formattedDuration,
        startRecording,
        stopRecording,
        togglePause
    } = useRecordingManager({
        meetingId,
        roomId,
        accessToken,
        onRecordingComplete
    })

    // Always show recording controls in meeting (for now - host check can be strict if needed)
    // Show recording indicator if actively recording, otherwise show controls
    if (!isHost && !isRecording && status !== 'uploading' && status !== 'processing') {
        // Still show a record button for non-hosts to at least screen record their own view
        return (
            <button
                onClick={startRecording}
                className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors"
                title="Record your screen"
            >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <circle cx="10" cy="10" r="5" />
                </svg>
                <span>Record</span>
            </button>
        )
    }

    if (!isHost && isRecording) {
        return (
            <div className="flex items-center gap-2 px-3 py-2 bg-red-600/90 text-white rounded-lg text-sm">
                <SignalIcon className="w-4 h-4 animate-pulse" />
                <span>Recording</span>
                <span className="font-mono">{formattedDuration}</span>
                <button
                    onClick={stopRecording}
                    className="ml-2 p-1 bg-red-700 hover:bg-red-800 rounded"
                    title="Stop recording"
                >
                    <StopCircleIcon className="w-4 h-4" />
                </button>
            </div>
        )
    }

    // Host controls
    return (
        <div className="flex items-center gap-2">
            {/* Recording status badge */}
            {status === 'recording' && (
                <div className="flex items-center gap-2 px-3 py-2 bg-red-600 text-white rounded-lg text-sm">
                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    <span className="font-mono">{formattedDuration}</span>
                </div>
            )}

            {status === 'uploading' && (
                <div className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg text-sm">
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span>Uploading...</span>
                </div>
            )}

            {status === 'complete' && (
                <div className="flex items-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg text-sm">
                    <CheckCircleIcon className="w-4 h-4" />
                    <span>Saved</span>
                </div>
            )}

            {status === 'error' && (
                <div className="flex items-center gap-2 px-3 py-2 bg-red-600 text-white rounded-lg text-sm">
                    <ExclamationCircleIcon className="w-4 h-4" />
                    <span>{error || 'Error'}</span>
                </div>
            )}

            {/* Control buttons */}
            {!isRecording && status !== 'uploading' && status !== 'processing' && (
                <button
                    onClick={startRecording}
                    className={clsx(
                        'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
                        'bg-red-600 hover:bg-red-700 text-white'
                    )}
                    title="Start Recording"
                >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <circle cx="10" cy="10" r="6" />
                    </svg>
                    Record
                </button>
            )}

            {isRecording && (
                <>
                    {/* Pause/Resume button */}
                    <button
                        onClick={togglePause}
                        className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                        title={isPaused ? 'Resume' : 'Pause'}
                    >
                        {isPaused ? (
                            <PlayCircleIcon className="w-6 h-6" />
                        ) : (
                            <PauseCircleIcon className="w-6 h-6" />
                        )}
                    </button>

                    {/* Stop button */}
                    <button
                        onClick={stopRecording}
                        className="p-2 rounded-lg bg-red-600 hover:bg-red-700 text-white transition-colors"
                        title="Stop Recording"
                    >
                        <StopCircleIcon className="w-6 h-6" />
                    </button>
                </>
            )}
        </div>
    )
}
