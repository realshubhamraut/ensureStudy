'use client'

import { getApiBaseUrl } from '@/utils/api'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import {
    LiveKitRoom,
    VideoConference,
    RoomAudioRenderer,
    ControlBar,
    useTracks,
    GridLayout,
    ParticipantTile,
    useRoomContext,
    useParticipants,
    Chat,
} from '@livekit/components-react'
import '@livekit/components-styles'
import { Track, Room, RoomEvent } from 'livekit-client'
import {
    ArrowLeftIcon,
    VideoCameraIcon,
    MicrophoneIcon,
    ComputerDesktopIcon,
    ChatBubbleLeftRightIcon,
    UsersIcon,
    PhoneXMarkIcon,
    Cog6ToothIcon,
    ClipboardDocumentIcon,
} from '@heroicons/react/24/outline'
import { RecordingControls } from '@/components/meeting/RecordingControls'

// LiveKit server configuration
const LIVEKIT_URL = process.env.NEXT_PUBLIC_LIVEKIT_URL || 'wss://your-livekit-server.livekit.cloud'

interface MeetingInfo {
    id: string
    title: string
    room_id: string
    classroom_id: string
    host_id: string
    status: string
}

export default function MeetingRoomPage() {
    const params = useParams()
    const router = useRouter()
    const searchParams = useSearchParams()
    const meetingId = params.id as string

    const [meeting, setMeeting] = useState<MeetingInfo | null>(null)
    const [token, setToken] = useState<string>('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [showChat, setShowChat] = useState(false)
    const [showParticipants, setShowParticipants] = useState(false)
    const [participantName, setParticipantName] = useState('Guest')
    const [isHost, setIsHost] = useState(false)
    const [accessToken, setAccessToken] = useState('')

    // Set participant name on client side only
    useEffect(() => {
        const name = searchParams.get('name') ||
            (typeof window !== 'undefined' ? localStorage.getItem('userName') : null) ||
            'Guest'
        setParticipantName(name)
    }, [searchParams])

    // Fetch meeting info and get token
    useEffect(() => {
        // Only run on client side
        if (typeof window === 'undefined') return

        const initMeeting = async () => {
            try {
                const accessToken = localStorage.getItem('accessToken')

                // Fetch meeting details
                const meetingRes = await fetch(`${getApiBaseUrl()}/api/meeting/${meetingId}`, {
                    headers: {
                        'Authorization': `Bearer ${accessToken}`
                    }
                })

                if (!meetingRes.ok) {
                    setError('Meeting not found')
                    return
                }

                const meetingData = await meetingRes.json()
                setMeeting(meetingData.meeting)

                // Check if current user is host
                const userId = localStorage.getItem('userId')
                const userRole = localStorage.getItem('userRole')
                const hostId = meetingData.meeting.host_id

                // Debug logging
                console.log('[Meeting] User ID from localStorage:', userId)
                console.log('[Meeting] Host ID from meeting:', hostId)
                console.log('[Meeting] User role:', userRole)

                // User is host if: their ID matches host_id OR they are a teacher
                // TEMP: Force true until localStorage is properly set after login
                const isUserHost = true // userId === hostId || userRole === 'teacher'
                console.log('[Meeting] Is host:', isUserHost)

                setIsHost(isUserHost)
                setAccessToken(accessToken || '')

                // Join the meeting
                await fetch(`${getApiBaseUrl()}/api/meeting/${meetingId}/join`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${accessToken}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        device_type: 'desktop'
                    })
                })

                // Get LiveKit token
                const tokenRes = await fetch(`${getApiBaseUrl()}/api/meeting/${meetingId}/token`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${accessToken}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        participant_name: participantName
                    })
                })

                if (tokenRes.ok) {
                    const tokenData = await tokenRes.json()
                    setToken(tokenData.token)
                } else {
                    // For demo/development, generate a mock token
                    console.warn('Could not get LiveKit token from server, using demo mode')
                    setToken('demo-token')
                }

            } catch (err) {
                console.error('Failed to initialize meeting:', err)
                setError('Failed to join meeting')
            } finally {
                setLoading(false)
            }
        }

        initMeeting()
    }, [meetingId, participantName])

    // Handle leaving the meeting
    const handleLeave = useCallback(async () => {
        try {
            await fetch(`${getApiBaseUrl()}/api/meeting/${meetingId}/leave`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
        } catch (err) {
            console.error('Error leaving meeting:', err)
        }

        // Navigate back to classroom based on user role
        const userRole = localStorage.getItem('userRole')
        if (meeting?.classroom_id) {
            if (userRole === 'teacher') {
                router.push(`/teacher/classroom/${meeting.classroom_id}?tab=meet`)
            } else {
                // Students and others go to student classroom view
                router.push(`/classrooms/${meeting.classroom_id}`)
            }
        } else {
            // Fallback to role-based dashboard
            if (userRole === 'teacher') {
                router.push('/teacher/dashboard')
            } else {
                router.push('/dashboard')
            }
        }
    }, [meetingId, meeting, router])

    // Copy meeting link
    const copyMeetingLink = () => {
        const link = `${window.location.origin}/meet/${meetingId}`
        navigator.clipboard.writeText(link)
        alert('Meeting link copied!')
    }

    if (loading) {
        return (
            <div className="h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-white text-lg">Joining meeting...</p>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                        <PhoneXMarkIcon className="w-8 h-8 text-red-500" />
                    </div>
                    <h1 className="text-white text-2xl font-bold mb-2">Unable to Join</h1>
                    <p className="text-gray-400 mb-6">{error}</p>
                    <button
                        onClick={() => router.back()}
                        className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                    >
                        Go Back
                    </button>
                </div>
            </div>
        )
    }

    // Demo mode when no LiveKit server is configured
    if (token === 'demo-token' || !LIVEKIT_URL.includes('livekit')) {
        return (
            <div className="h-screen bg-gray-900 flex flex-col">
                {/* Header */}
                <div className="h-16 bg-gray-800 flex items-center justify-between px-6 border-b border-gray-700">
                    <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
                            <VideoCameraIcon className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="text-white font-semibold">{meeting?.title || 'Meeting'}</h1>
                            <p className="text-gray-400 text-sm">Demo Mode - Configure LiveKit for full experience</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        {/* Recording Controls - Host only can record */}
                        <RecordingControls
                            meetingId={meetingId}
                            roomId={meeting?.room_id || meetingId}
                            accessToken={accessToken}
                            isHost={isHost}
                            onRecordingComplete={(recordingId) => {
                                console.log('Recording saved:', recordingId)
                            }}
                        />
                        <button
                            onClick={copyMeetingLink}
                            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg"
                            title="Copy link"
                        >
                            <ClipboardDocumentIcon className="w-5 h-5" />
                        </button>
                        <button
                            onClick={handleLeave}
                            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
                        >
                            <PhoneXMarkIcon className="w-5 h-5" />
                            Leave
                        </button>
                    </div>
                </div>

                {/* Main Content - Video Grid */}
                <div className="flex-1 flex relative">
                    <div className="flex-1 p-6">
                        <div className="h-full bg-gray-800 rounded-2xl p-6 flex flex-col items-center justify-center">
                            {/* Demo self view */}
                            <div className="w-full max-w-4xl aspect-video bg-gray-700 rounded-xl mb-6 flex items-center justify-center relative overflow-hidden">
                                <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-blue-600/20"></div>
                                <div className="text-center z-10">
                                    <div className="w-24 h-24 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                                        <span className="text-white text-3xl font-bold">
                                            {participantName.charAt(0).toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-white text-xl font-medium">{participantName}</p>
                                    <p className="text-gray-400 mt-2">You (Host)</p>
                                </div>
                            </div>


                            {/* Demo controls */}
                            <div className="flex items-center gap-4">
                                <button className="w-14 h-14 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center text-white">
                                    <MicrophoneIcon className="w-6 h-6" />
                                </button>
                                <button className="w-14 h-14 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center text-white">
                                    <VideoCameraIcon className="w-6 h-6" />
                                </button>
                                <button className="w-14 h-14 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center text-white">
                                    <ComputerDesktopIcon className="w-6 h-6" />
                                </button>
                                <button
                                    onClick={() => setShowChat(!showChat)}
                                    className={`w-14 h-14 ${showChat ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'} rounded-full flex items-center justify-center text-white`}
                                >
                                    <ChatBubbleLeftRightIcon className="w-6 h-6" />
                                </button>
                                <button
                                    onClick={() => setShowParticipants(!showParticipants)}
                                    className={`w-14 h-14 ${showParticipants ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'} rounded-full flex items-center justify-center text-white`}
                                >
                                    <UsersIcon className="w-6 h-6" />
                                </button>
                                <button
                                    onClick={handleLeave}
                                    className="w-14 h-14 bg-red-600 hover:bg-red-700 rounded-full flex items-center justify-center text-white"
                                >
                                    <PhoneXMarkIcon className="w-6 h-6" />
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Chat/Participants Sidebar */}
                    {(showChat || showParticipants) && (
                        <div className="w-80 bg-gray-800 border-l border-gray-700 p-4">
                            {showChat && (
                                <div className="h-full flex flex-col">
                                    <h3 className="text-white font-semibold mb-4">Chat</h3>
                                    <div className="flex-1 bg-gray-700 rounded-lg p-4 mb-4 overflow-y-auto">
                                        <p className="text-gray-400 text-sm text-center">No messages yet</p>
                                    </div>
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            placeholder="Type a message..."
                                            className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                        />
                                        <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                                            Send
                                        </button>
                                    </div>
                                </div>
                            )}
                            {showParticipants && !showChat && (
                                <div>
                                    <h3 className="text-white font-semibold mb-4">Participants (1)</h3>
                                    <div className="space-y-2">
                                        <div className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg">
                                            <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center">
                                                <span className="text-white font-medium">
                                                    {participantName.charAt(0).toUpperCase()}
                                                </span>
                                            </div>
                                            <div>
                                                <p className="text-white font-medium">{participantName}</p>
                                                <p className="text-gray-400 text-sm">Host</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Setup Instructions Toast */}
                <div className="fixed bottom-6 left-6 max-w-md bg-yellow-500/20 border border-yellow-500/50 rounded-xl p-4">
                    <h4 className="text-yellow-400 font-semibold mb-2">🔧 LiveKit Setup Required</h4>
                    <p className="text-yellow-200 text-sm">
                        To enable real video calls, set up a LiveKit server and add <code className="bg-yellow-500/30 px-1 rounded">NEXT_PUBLIC_LIVEKIT_URL</code> to your .env file.
                    </p>
                </div>
            </div>
        )
    }

    // Full LiveKit integration when connected
    return (
        <div className="h-screen bg-gray-900">
            <LiveKitRoom
                serverUrl={LIVEKIT_URL}
                token={token}
                connectOptions={{
                    autoSubscribe: true,
                }}
                video={true}
                audio={true}
                onDisconnected={handleLeave}
                data-lk-theme="default"
                style={{ height: '100%' }}
            >
                {/* Custom Header */}
                <div className="h-16 bg-gray-800 flex items-center justify-between px-6 border-b border-gray-700 absolute top-0 left-0 right-0 z-10">
                    <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
                            <VideoCameraIcon className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="text-white font-semibold">{meeting?.title || 'Meeting'}</h1>
                            <MeetingTimer />
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        {/* Recording Controls - Host only can record */}
                        <RecordingControls
                            meetingId={meetingId}
                            roomId={meeting?.room_id || meetingId}
                            accessToken={accessToken}
                            isHost={isHost}
                            onRecordingComplete={(recordingId) => {
                                console.log('Recording saved:', recordingId)
                            }}
                        />
                        <button
                            onClick={copyMeetingLink}
                            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg"
                            title="Copy link"
                        >
                            <ClipboardDocumentIcon className="w-5 h-5" />
                        </button>
                        <ParticipantCount />
                    </div>
                </div>

                {/* Video Conference */}
                <div className="pt-16 h-full relative">
                    <VideoConference />
                </div>

                <RoomAudioRenderer />
            </LiveKitRoom>
        </div>
    )
}

// Meeting timer component
function MeetingTimer() {
    const [elapsed, setElapsed] = useState(0)

    useEffect(() => {
        const interval = setInterval(() => {
            setElapsed(e => e + 1)
        }, 1000)
        return () => clearInterval(interval)
    }, [])

    const minutes = Math.floor(elapsed / 60)
    const seconds = elapsed % 60

    return (
        <p className="text-gray-400 text-sm">
            {String(minutes).padStart(2, '0')}:{String(seconds).padStart(2, '0')}
        </p>
    )
}

// Participant count component
function ParticipantCount() {
    const participants = useParticipants()
    return (
        <div className="flex items-center gap-2 bg-gray-700 px-3 py-1.5 rounded-lg">
            <UsersIcon className="w-4 h-4 text-gray-400" />
            <span className="text-white text-sm">{participants.length}</span>
        </div>
    )
}
