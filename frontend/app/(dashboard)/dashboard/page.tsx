'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import {
    ChatBubbleLeftRightIcon,
    BookOpenIcon,
    ClipboardDocumentListIcon,
    TrophyIcon,
    FireIcon,
    AcademicCapIcon,
    ArrowTrendingUpIcon,
    BellIcon,
    MegaphoneIcon,
    DocumentTextIcon,
    VideoCameraIcon,
    CalendarIcon,
    CheckCircleIcon,
    UserCircleIcon,
    ArrowPathIcon,
    CameraIcon,
    XMarkIcon
} from '@heroicons/react/24/outline'

interface Notification {
    id: string
    type: 'stream' | 'material' | 'meet' | 'assignment' | 'result' | 'message' | 'assessment'
    title: string
    message: string
    source_id?: string
    source_type?: string
    action_url?: string
    is_read: boolean
    created_at: string
}

// Format relative time
function formatRelativeTime(dateString: string): string {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
}

// Helper to get API base URL
function getApiBaseUrl() {
    if (typeof window !== 'undefined') {
        return process.env.NEXT_PUBLIC_API_URL || `http://${window.location.hostname}:5000`
    }
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
}

export default function DashboardPage() {
    const { data: session } = useSession()
    const [notifications, setNotifications] = useState<Notification[]>([])
    const [loadingNotifications, setLoadingNotifications] = useState(true)

    const apiUrl = getApiBaseUrl()

    useEffect(() => {
        if (!session?.accessToken) return

        async function fetchNotifications() {
            try {
                setLoadingNotifications(true)
                const res = await fetch(`${apiUrl}/api/notifications/recent?limit=5`, {
                    headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                })

                if (res.ok) {
                    const data = await res.json()
                    setNotifications(data.notifications || [])
                }
            } catch (err) {
                console.error('Error fetching notifications:', err)
            } finally {
                setLoadingNotifications(false)
            }
        }

        fetchNotifications()
    }, [session?.accessToken, apiUrl])

    const getNotificationIcon = (type: Notification['type']) => {
        switch (type) {
            case 'stream':
                return <MegaphoneIcon className="w-4 h-4" />
            case 'material':
                return <DocumentTextIcon className="w-4 h-4" />
            case 'meet':
                return <VideoCameraIcon className="w-4 h-4" />
            case 'assignment':
                return <ClipboardDocumentListIcon className="w-4 h-4" />
            case 'assessment':
                return <ClipboardDocumentListIcon className="w-4 h-4" />
            case 'result':
                return <CheckCircleIcon className="w-4 h-4" />
            case 'message':
                return <ChatBubbleLeftRightIcon className="w-4 h-4" />
            default:
                return <BellIcon className="w-4 h-4" />
        }
    }

    const getNotificationColor = (type: Notification['type']) => {
        switch (type) {
            case 'stream':
                return 'bg-blue-100 text-blue-600'
            case 'material':
                return 'bg-purple-100 text-purple-600'
            case 'meet':
                return 'bg-green-100 text-green-600'
            case 'assignment':
                return 'bg-orange-100 text-orange-600'
            case 'assessment':
                return 'bg-yellow-100 text-yellow-600'
            case 'result':
                return 'bg-emerald-100 text-emerald-600'
            case 'message':
                return 'bg-pink-100 text-pink-600'
            default:
                return 'bg-gray-100 text-gray-600'
        }
    }

    return (
        <div className="space-y-8">
            {/* Welcome Section + Notifications Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Welcome Section - Takes 2 columns */}
                <div className="lg:col-span-2 relative bg-gradient-to-r from-primary-400 to-secondary-400 rounded-2xl p-8 text-white">
                    {/* Grain texture overlay */}
                    <div
                        className="absolute inset-0 opacity-50 pointer-events-none rounded-2xl overflow-hidden"
                        style={{
                            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
                        }}
                    />
                    <div className="relative z-10 flex items-center justify-between">
                        {/* Left side - Text content */}
                        <div className="flex-1">
                            <h1 className="text-3xl font-bold mb-2 drop-shadow-lg" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.2)' }}>
                                Welcome back, {session?.user?.username || 'Student'}!{' '}
                                <span className="inline-block animate-wave origin-bottom-right">👋</span>
                            </h1>
                            <p className="text-white text-lg drop-shadow-lg" style={{ textShadow: '0 2px 6px rgba(0,0,0,0.3)' }}>
                                Ready to continue your learning journey? Let&apos;s make today productive.
                            </p>
                            <div className="flex gap-4 mt-6">
                                <Link href="/chat" className="px-6 py-3 bg-white text-primary-600 rounded-lg font-medium hover:bg-gray-100 transition-colors shadow-md">
                                    Start Learning
                                </Link>
                                <Link href="/assessments" className="px-6 py-3 bg-gray-800/20 text-white rounded-lg font-medium hover:bg-gray-800/35 transition-colors backdrop-blur-sm border-2 border-white/50 shadow-md">
                                    Take Assessment
                                </Link>
                            </div>
                        </div>

                        {/* Right side - Avatar */}
                        <div className="hidden md:flex items-center justify-center ml-8">
                            <AvatarSelector session={session} />
                        </div>
                    </div>
                </div>

                {/* What's New - Scrollable Notifications Panel */}
                <div className="card flex flex-col h-full min-h-[200px]">
                    <div className="flex items-center justify-between mb-3 flex-shrink-0">
                        <div className="flex items-center gap-2">
                            <BellIcon className="w-5 h-5 text-primary-600" />
                            <h2 className="text-lg font-bold text-gray-900">What&apos;s New</h2>
                        </div>
                        <Link href="/notifications" className="text-sm text-primary-600 hover:text-primary-700 font-medium">
                            View All
                        </Link>
                    </div>

                    {/* Scrollable Notification List */}
                    <div className="flex-1 overflow-y-auto -mx-4 px-4" style={{ maxHeight: '140px' }}>
                        {loadingNotifications ? (
                            <div className="flex items-center justify-center h-full py-8">
                                <ArrowPathIcon className="w-6 h-6 animate-spin text-primary-600" />
                            </div>
                        ) : notifications.length > 0 ? (
                            <div className="space-y-2">
                                {notifications.map((notif) => (
                                    <Link
                                        key={notif.id}
                                        href={notif.action_url || '/notifications'}
                                        className={`block p-3 rounded-lg border transition-colors hover:bg-gray-50 ${notif.is_read ? 'bg-white border-gray-100' : 'bg-blue-50/50 border-blue-100'
                                            }`}
                                    >
                                        <div className="flex items-start gap-3">
                                            <div className={`p-1.5 rounded-lg flex-shrink-0 ${getNotificationColor(notif.type)}`}>
                                                {getNotificationIcon(notif.type)}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <p className={`text-sm font-medium text-gray-900 truncate ${!notif.is_read ? 'font-semibold' : ''}`}>
                                                    {notif.title}
                                                </p>
                                                <p className="text-xs text-gray-500 truncate mt-0.5">
                                                    {notif.message}
                                                </p>
                                                <div className="flex items-center gap-2 mt-1">
                                                    {notif.source_type && (
                                                        <span className="text-xs text-gray-400 capitalize">{notif.source_type}</span>
                                                    )}
                                                    <span className="text-xs text-gray-400">• {formatRelativeTime(notif.created_at)}</span>
                                                </div>
                                            </div>
                                            {!notif.is_read && (
                                                <div className="w-2 h-2 bg-primary-500 rounded-full flex-shrink-0 mt-1.5" />
                                            )}
                                        </div>
                                    </Link>
                                ))}
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center text-center py-6 h-full">
                                <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
                                    <BellIcon className="w-6 h-6 text-gray-400" />
                                </div>
                                <p className="text-gray-500 font-medium text-sm">No notifications yet</p>
                                <p className="text-gray-400 text-xs mt-1">Check back later for updates</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                    icon={<FireIcon className="w-6 h-6" />}
                    label="Study Streak"
                    value="7 days"
                    change="+2 from last week"
                    color="orange"
                />
                <StatCard
                    icon={<AcademicCapIcon className="w-6 h-6" />}
                    label="Topics Mastered"
                    value="12"
                    change="3 this week"
                    color="green"
                />
                <StatCard
                    icon={<ClipboardDocumentListIcon className="w-6 h-6" />}
                    label="Assessments"
                    value="24"
                    change="Avg: 82%"
                    color="blue"
                />
                <StatCard
                    icon={<TrophyIcon className="w-6 h-6" />}
                    label="Global Rank"
                    value="#42"
                    change="↑ 5 positions"
                    color="purple"
                />
            </div>

            {/* Quick Actions */}
            <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <QuickActionCard
                        href="/chat"
                        icon={<ChatBubbleLeftRightIcon className="w-8 h-8" />}
                        title="Ask AI Tutor"
                        description="Get instant help with any topic"
                        gradient="from-blue-500 to-cyan-500"
                    />
                    <QuickActionCard
                        href="/classrooms"
                        icon={<BookOpenIcon className="w-8 h-8" />}
                        title="My Classrooms"
                        description="View study materials & notes"
                        gradient="from-purple-500 to-pink-500"
                    />
                    <QuickActionCard
                        href="/assessments"
                        icon={<ClipboardDocumentListIcon className="w-8 h-8" />}
                        title="Practice Quiz"
                        description="Test your knowledge"
                        gradient="from-green-500 to-emerald-500"
                    />
                </div>
            </div>

            {/* Weak Topics Alert */}
            <div className="card border-l-4 border-warning-500">
                <div className="flex items-start gap-4">
                    <div className="p-3 bg-warning-100 rounded-lg">
                        <ArrowTrendingUpIcon className="w-6 h-6 text-warning-600" />
                    </div>
                    <div>
                        <h3 className="font-bold text-gray-900">Topics Needing Attention</h3>
                        <p className="text-gray-600 mt-1">
                            Based on your recent assessments, you might want to review these topics:
                        </p>
                        <div className="flex flex-wrap gap-2 mt-3">
                            <span className="px-3 py-1 bg-warning-100 text-warning-700 rounded-full text-sm font-medium">
                                Photosynthesis
                            </span>
                            <span className="px-3 py-1 bg-warning-100 text-warning-700 rounded-full text-sm font-medium">
                                Quadratic Equations
                            </span>
                            <span className="px-3 py-1 bg-warning-100 text-warning-700 rounded-full text-sm font-medium">
                                World War II
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Activity */}
            <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Activity</h2>
                <div className="card divide-y divide-gray-100">
                    <ActivityItem
                        icon={<ChatBubbleLeftRightIcon className="w-5 h-5" />}
                        title="Asked about mitochondria function"
                        time="2 hours ago"
                        color="blue"
                    />
                    <ActivityItem
                        icon={<ClipboardDocumentListIcon className="w-5 h-5" />}
                        title="Completed Biology Quiz - 85%"
                        time="Yesterday"
                        color="green"
                    />
                    <ActivityItem
                        icon={<BookOpenIcon className="w-5 h-5" />}
                        title="Reviewed Calculus notes"
                        time="2 days ago"
                        color="purple"
                    />
                </div>
            </div>
        </div>
    )
}

function StatCard({
    icon,
    label,
    value,
    change,
    color
}: {
    icon: React.ReactNode
    label: string
    value: string
    change: string
    color: 'orange' | 'green' | 'blue' | 'purple'
}) {
    const colors = {
        orange: 'bg-orange-100 text-orange-600',
        green: 'bg-green-100 text-green-600',
        blue: 'bg-blue-100 text-blue-600',
        purple: 'bg-purple-100 text-purple-600',
    }

    return (
        <div className="card">
            <div className={`inline-flex p-3 rounded-lg ${colors[color]} mb-4`}>
                {icon}
            </div>
            <p className="text-gray-500 text-sm">{label}</p>
            <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
            <p className="text-sm text-gray-500 mt-1">{change}</p>
        </div>
    )
}

function QuickActionCard({
    href,
    icon,
    title,
    description,
    gradient
}: {
    href: string
    icon: React.ReactNode
    title: string
    description: string
    gradient: string
}) {
    return (
        <Link href={href} className="card-hover group flex items-start gap-4">
            <div className={`p-4 rounded-xl bg-gradient-to-br ${gradient} text-white group-hover:scale-110 transition-transform`}>
                {icon}
            </div>
            <div>
                <h3 className="font-bold text-gray-900">{title}</h3>
                <p className="text-gray-600 text-sm">{description}</p>
            </div>
        </Link>
    )
}

function ActivityItem({
    icon,
    title,
    time,
    color
}: {
    icon: React.ReactNode
    title: string
    time: string
    color: 'blue' | 'green' | 'purple'
}) {
    const colors = {
        blue: 'bg-blue-100 text-blue-600',
        green: 'bg-green-100 text-green-600',
        purple: 'bg-purple-100 text-purple-600',
    }

    return (
        <div className="flex items-center gap-4 py-4">
            <div className={`p-2 rounded-lg ${colors[color]}`}>
                {icon}
            </div>
            <div className="flex-1">
                <p className="font-medium text-gray-900">{title}</p>
            </div>
            <p className="text-sm text-gray-500">{time}</p>
        </div>
    )
}

// Preset avatars - Image paths from public/avatars folder
const MALE_AVATARS = [
    '/avatars/male/Untitled.png',
    '/avatars/male/Untitled 2.png',
    '/avatars/male/Untitled 3.png',
    '/avatars/male/Untitled 4.png',
    '/avatars/male/Untitled 5.png',
    '/avatars/male/Untitled 6.png',
    '/avatars/male/Untitled 7.png',
    '/avatars/male/Untitled 8.png',
    '/avatars/male/Untitled 9.png',
    '/avatars/male/Untitled 10.png',
    '/avatars/male/Untitledadfa.png',
    '/avatars/male/male.png',
    '/avatars/male/12.png',
    '/avatars/male/12412.png',
    '/avatars/male/12r4124e.png',
    '/avatars/male/1r1.png',
    '/avatars/male/3241.png',
    '/avatars/male/3r14.png',
    '/avatars/male/afq.png',
    '/avatars/male/q124.png',
    '/avatars/male/qrqwsr.png',
]

const FEMALE_AVATARS = [
    '/avatars/female/Untitled.png',
    '/avatars/female/Untitled 2.png',
    '/avatars/female/Untitled 3.png',
    '/avatars/female/Untitled 4.png',
    '/avatars/female/Untitled 5.png',
    '/avatars/female/Untitled 11.png',
    '/avatars/female/Untitled 11sde.png',
    '/avatars/female/Untitled copy.png',
    '/avatars/female/df.png',
    '/avatars/female/dfda.png',
    '/avatars/female/123.png',
    '/avatars/female/1231231.png',
    '/avatars/female/11211.png',
    '/avatars/female/431.png',
    '/avatars/female/aadfa.png',
    '/avatars/female/adfadfa.png',
    '/avatars/female/adfae.png',
    '/avatars/female/adfgadfa.png',
    '/avatars/female/adgadgaf.png',
    '/avatars/female/adgff.png',
    '/avatars/female/agaabf.png',
    '/avatars/female/agad.png',
    '/avatars/female/agadaf.png',
    '/avatars/female/agadfagfa.png',
    '/avatars/female/agfaa.png',
    '/avatars/female/asdagd.png',
    '/avatars/female/asdfadfa.png',
    '/avatars/female/asfdgadf.png',
    '/avatars/female/fead.png',
    '/avatars/female/fews.png',
    '/avatars/female/gtrew.png',
    '/avatars/female/kljg.png',
    '/avatars/female/lkadf.png',
    '/avatars/female/sdf.png',
    '/avatars/female/sdfa.png',
    '/avatars/female/sdfadfadfg.png',
    '/avatars/female/tsdf1.png',
    '/avatars/female/wefasf.png',
    '/avatars/female/wqerafd.png',
]

function AvatarSelector({ session }: { session: any }) {
    const [showPicker, setShowPicker] = useState(false)
    const [activeTab, setActiveTab] = useState<'male' | 'female' | 'upload'>('male')
    const [selectedAvatar, setSelectedAvatar] = useState<string>(MALE_AVATARS[0])
    const [customImage, setCustomImage] = useState<string | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const pickerRef = useRef<HTMLDivElement>(null)

    // Load saved avatar on mount
    useEffect(() => {
        const savedAvatar = localStorage.getItem('userAvatar')
        const savedCustomImage = localStorage.getItem('userCustomImage')
        if (savedCustomImage) {
            setCustomImage(savedCustomImage)
        } else if (savedAvatar) {
            setSelectedAvatar(savedAvatar)
        }
    }, [])

    // Close picker when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
                setShowPicker(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const handleAvatarSelect = (avatar: string) => {
        setSelectedAvatar(avatar)
        setCustomImage(null)
        localStorage.setItem('userAvatar', avatar)
        localStorage.removeItem('userCustomImage')
        setShowPicker(false)
    }

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        if (file) {
            const reader = new FileReader()
            reader.onloadend = () => {
                const base64 = reader.result as string
                setCustomImage(base64)
                localStorage.setItem('userCustomImage', base64)
                localStorage.removeItem('userAvatar')
                setShowPicker(false)
            }
            reader.readAsDataURL(file)
        }
    }

    return (
        <div className="relative group" ref={pickerRef}>
            {/* Avatar Display */}
            <button
                onClick={() => setShowPicker(!showPicker)}
                className="w-24 h-24 rounded-full bg-white/20 backdrop-blur-sm border-4 border-white/50 flex items-center justify-center hover:scale-105 hover:border-white/80 transition-all cursor-pointer shadow-xl overflow-hidden"
                title="Click to change avatar"
            >
                {customImage ? (
                    <img src={customImage} alt="Avatar" className="w-full h-full object-cover" />
                ) : (
                    <img src={selectedAvatar} alt="Avatar" className="w-full h-full object-cover" />
                )}
            </button>

            {/* Camera badge - appears on hover */}
            <div className="absolute bottom-0 right-0 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center border-2 border-primary-400 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer">
                <CameraIcon className="w-4 h-4 text-primary-600" />
            </div>

            {/* Avatar Picker Popup */}
            {showPicker && (
                <div className="absolute right-0 top-28 w-80 bg-white rounded-xl shadow-2xl border border-gray-200 z-[9999] overflow-hidden">
                    {/* Header */}
                    <div className="flex items-center justify-between p-3 border-b border-gray-100">
                        <h3 className="font-semibold text-gray-900">Choose Avatar</h3>
                        <button onClick={() => setShowPicker(false)} className="p-1 hover:bg-gray-100 rounded">
                            <XMarkIcon className="w-5 h-5 text-gray-500" />
                        </button>
                    </div>

                    {/* Tabs */}
                    <div className="flex border-b border-gray-100">
                        <button
                            onClick={() => setActiveTab('male')}
                            className={`flex-1 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-1.5 ${activeTab === 'male'
                                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
                                : 'text-gray-500 hover:text-gray-700'
                                }`}
                        >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            Male
                        </button>
                        <button
                            onClick={() => setActiveTab('female')}
                            className={`flex-1 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-1.5 ${activeTab === 'female'
                                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
                                : 'text-gray-500 hover:text-gray-700'
                                }`}
                        >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            Female
                        </button>
                        <button
                            onClick={() => setActiveTab('upload')}
                            className={`flex-1 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-1.5 ${activeTab === 'upload'
                                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
                                : 'text-gray-500 hover:text-gray-700'
                                }`}
                        >
                            <CameraIcon className="w-4 h-4" />
                            Upload
                        </button>
                    </div>

                    {/* Avatar Grid */}
                    <div className="p-3 max-h-64 overflow-y-auto">
                        {activeTab === 'male' && (
                            <div className="grid grid-cols-5 gap-2">
                                {MALE_AVATARS.map((avatar, idx) => (
                                    <button
                                        key={idx}
                                        onClick={() => handleAvatarSelect(avatar)}
                                        className={`w-12 h-12 rounded-lg overflow-hidden hover:ring-2 hover:ring-primary-300 transition-all ${selectedAvatar === avatar && !customImage ? 'ring-2 ring-primary-500 bg-primary-100' : 'bg-gray-50'
                                            }`}
                                    >
                                        <img src={avatar} alt={`Male avatar ${idx + 1}`} className="w-full h-full object-cover" />
                                    </button>
                                ))}
                            </div>
                        )}

                        {activeTab === 'female' && (
                            <div className="grid grid-cols-5 gap-2">
                                {FEMALE_AVATARS.map((avatar, idx) => (
                                    <button
                                        key={idx}
                                        onClick={() => handleAvatarSelect(avatar)}
                                        className={`w-12 h-12 rounded-lg overflow-hidden hover:ring-2 hover:ring-primary-300 transition-all ${selectedAvatar === avatar && !customImage ? 'ring-2 ring-primary-500 bg-primary-100' : 'bg-gray-50'
                                            }`}
                                    >
                                        <img src={avatar} alt={`Female avatar ${idx + 1}`} className="w-full h-full object-cover" />
                                    </button>
                                ))}
                            </div>
                        )}

                        {activeTab === 'upload' && (
                            <div className="space-y-4">
                                {/* Notion Faces Link */}
                                <div className="text-center p-4 bg-gradient-to-r from-primary-50 to-secondary-50 rounded-lg border border-primary-100">
                                    <p className="text-sm text-gray-700 mb-2">Create a custom illustrated avatar</p>
                                    <a
                                        href="https://faces.notion.com/"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-2 px-4 py-2 bg-black text-white rounded-lg font-medium hover:bg-gray-800 transition-colors text-sm"
                                    >
                                        <span>✨</span>
                                        Open Notion Faces
                                        <span className="text-xs">↗</span>
                                    </a>
                                    <p className="text-xs text-gray-500 mt-2">Design your face, save it, then upload here</p>
                                </div>

                                {/* Divider */}
                                <div className="flex items-center gap-2">
                                    <div className="flex-1 h-px bg-gray-200"></div>
                                    <span className="text-xs text-gray-400">or</span>
                                    <div className="flex-1 h-px bg-gray-200"></div>
                                </div>

                                {/* Photo Upload */}
                                <div className="text-center">
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="image/*"
                                        onChange={handleFileUpload}
                                        className="hidden"
                                    />
                                    <button
                                        onClick={() => fileInputRef.current?.click()}
                                        className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors text-sm"
                                    >
                                        📷 Upload Photo
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
