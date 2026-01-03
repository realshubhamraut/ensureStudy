'use client'

import { getApiBaseUrl } from '@/utils/api'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
    BellIcon,
    MegaphoneIcon,
    DocumentTextIcon,
    ClipboardDocumentListIcon,
    ChartBarIcon,
    ChatBubbleOvalLeftIcon,
    CalendarIcon,
    TrophyIcon,
    CheckIcon,
    TrashIcon,
    ArrowPathIcon,
    FunnelIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Notification {
    id: string
    type: string
    title: string
    message: string
    source_type: string
    action_url?: string
    is_read: boolean
    created_at: string
}

export default function NotificationsPage() {
    const { data: session } = useSession()
    const [notifications, setNotifications] = useState<Notification[]>([])
    const [unreadCount, setUnreadCount] = useState(0)
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState<string>('all')
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || `${getApiBaseUrl()}'

    // Fetch notifications
    useEffect(() => {
        if (!session?.accessToken) return

        async function fetchNotifications() {
            setLoading(true)
            try {
                let url = `${apiUrl}/api/notifications?page=${page}&per_page=20`
                if (filter === 'unread') {
                    url += '&unread_only=true'
                } else if (filter !== 'all') {
                    url += `&type=${filter}`
                }

                const res = await fetch(url, {
                    headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                })
                if (res.ok) {
                    const data = await res.json()
                    setNotifications(data.notifications || [])
                    setUnreadCount(data.unread_count || 0)
                    setTotalPages(data.pages || 1)
                }
            } catch (err) {
                console.error('Error fetching notifications:', err)
            } finally {
                setLoading(false)
            }
        }

        fetchNotifications()
    }, [session?.accessToken, apiUrl, page, filter])

    async function markAsRead(id: string) {
        try {
            const res = await fetch(`${apiUrl}/api/notifications/${id}/read`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${session?.accessToken}` }
            })
            if (res.ok) {
                setNotifications(prev => prev.map(n =>
                    n.id === id ? { ...n, is_read: true } : n
                ))
                setUnreadCount(prev => Math.max(0, prev - 1))
            }
        } catch (err) {
            console.error('Error marking as read:', err)
        }
    }

    async function markAllAsRead() {
        try {
            const res = await fetch(`${apiUrl}/api/notifications/read-all`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${session?.accessToken}` }
            })
            if (res.ok) {
                setNotifications(prev => prev.map(n => ({ ...n, is_read: true })))
                setUnreadCount(0)
            }
        } catch (err) {
            console.error('Error marking all as read:', err)
        }
    }

    async function deleteNotification(id: string) {
        try {
            const res = await fetch(`${apiUrl}/api/notifications/${id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${session?.accessToken}` }
            })
            if (res.ok) {
                setNotifications(prev => prev.filter(n => n.id !== id))
            }
        } catch (err) {
            console.error('Error deleting notification:', err)
        }
    }

    function getNotificationIcon(type: string) {
        switch (type) {
            case 'stream': return <MegaphoneIcon className="w-5 h-5" />
            case 'material': return <DocumentTextIcon className="w-5 h-5" />
            case 'assignment': return <ClipboardDocumentListIcon className="w-5 h-5" />
            case 'assessment': return <ChartBarIcon className="w-5 h-5" />
            case 'message': return <ChatBubbleOvalLeftIcon className="w-5 h-5" />
            case 'meet': return <CalendarIcon className="w-5 h-5" />
            case 'result': return <TrophyIcon className="w-5 h-5" />
            default: return <BellIcon className="w-5 h-5" />
        }
    }

    function getNotificationColor(type: string) {
        switch (type) {
            case 'stream': return 'bg-blue-100 text-blue-600'
            case 'material': return 'bg-purple-100 text-purple-600'
            case 'assignment': return 'bg-orange-100 text-orange-600'
            case 'assessment': return 'bg-red-100 text-red-600'
            case 'message': return 'bg-green-100 text-green-600'
            case 'meet': return 'bg-cyan-100 text-cyan-600'
            case 'result': return 'bg-yellow-100 text-yellow-600'
            default: return 'bg-gray-100 text-gray-600'
        }
    }

    function formatTime(dateStr: string) {
        const date = new Date(dateStr)
        const now = new Date()
        const diffMs = now.getTime() - date.getTime()
        const diffMins = Math.floor(diffMs / 60000)
        const diffHours = Math.floor(diffMs / 3600000)
        const diffDays = Math.floor(diffMs / 86400000)

        if (diffMins < 1) return 'Just now'
        if (diffMins < 60) return `${diffMins} min ago`
        if (diffHours < 24) return `${diffHours} hours ago`
        if (diffDays < 7) return `${diffDays} days ago`
        return date.toLocaleDateString()
    }

    const filters = [
        { id: 'all', label: 'All' },
        { id: 'unread', label: 'Unread' },
        { id: 'stream', label: 'Announcements' },
        { id: 'assignment', label: 'Assignments' },
        { id: 'material', label: 'Materials' },
        { id: 'message', label: 'Messages' },
        { id: 'result', label: 'Results' },
    ]

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-primary-100 rounded-xl">
                        <BellIcon className="w-6 h-6 text-primary-600" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Notifications</h1>
                        <p className="text-gray-500">
                            {unreadCount > 0 ? `${unreadCount} unread` : 'All caught up!'}
                        </p>
                    </div>
                </div>
                {unreadCount > 0 && (
                    <button
                        onClick={markAllAsRead}
                        className="px-4 py-2 text-primary-600 hover:bg-primary-50 rounded-lg font-medium flex items-center gap-2"
                    >
                        <CheckIcon className="w-5 h-5" />
                        Mark all as read
                    </button>
                )}
            </div>

            {/* Filters */}
            <div className="flex items-center gap-2 overflow-x-auto pb-2">
                <FunnelIcon className="w-5 h-5 text-gray-400 flex-shrink-0" />
                {filters.map(f => (
                    <button
                        key={f.id}
                        onClick={() => { setFilter(f.id); setPage(1); }}
                        className={clsx(
                            'px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors',
                            filter === f.id
                                ? 'bg-primary-600 text-white'
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        )}
                    >
                        {f.label}
                    </button>
                ))}
            </div>

            {/* Notifications List */}
            <div className="card">
                {loading ? (
                    <div className="flex items-center justify-center py-12">
                        <ArrowPathIcon className="w-8 h-8 animate-spin text-primary-600" />
                    </div>
                ) : notifications.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                        <BellIcon className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                        <p className="text-lg font-medium">No notifications</p>
                        <p className="text-sm mt-1">
                            {filter === 'unread'
                                ? "You've read all your notifications!"
                                : "Check back later for updates from your classrooms"}
                        </p>
                    </div>
                ) : (
                    <div className="divide-y divide-gray-100">
                        {notifications.map(notif => (
                            <div
                                key={notif.id}
                                className={clsx(
                                    'flex items-start gap-4 p-4 group',
                                    !notif.is_read && 'bg-primary-50/30'
                                )}
                            >
                                <div className={`p-2.5 rounded-xl flex-shrink-0 ${getNotificationColor(notif.type)}`}>
                                    {getNotificationIcon(notif.type)}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <Link
                                        href={notif.action_url || '#'}
                                        onClick={() => !notif.is_read && markAsRead(notif.id)}
                                        className="block"
                                    >
                                        <p className={clsx(
                                            'text-gray-900',
                                            !notif.is_read && 'font-semibold'
                                        )}>
                                            {notif.title}
                                        </p>
                                        {notif.message && (
                                            <p className="text-sm text-gray-500 mt-1">{notif.message}</p>
                                        )}
                                        <p className="text-xs text-gray-400 mt-2">
                                            {formatTime(notif.created_at)}
                                        </p>
                                    </Link>
                                </div>
                                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    {!notif.is_read && (
                                        <button
                                            onClick={() => markAsRead(notif.id)}
                                            className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg"
                                            title="Mark as read"
                                        >
                                            <CheckIcon className="w-4 h-4" />
                                        </button>
                                    )}
                                    <button
                                        onClick={() => deleteNotification(notif.id)}
                                        className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg"
                                        title="Delete"
                                    >
                                        <TrashIcon className="w-4 h-4" />
                                    </button>
                                </div>
                                {!notif.is_read && (
                                    <span className="w-2.5 h-2.5 bg-primary-600 rounded-full flex-shrink-0 mt-2"></span>
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="flex items-center justify-center gap-2 pt-4 border-t border-gray-100">
                        <button
                            onClick={() => setPage(p => Math.max(1, p - 1))}
                            disabled={page === 1}
                            className="px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg disabled:opacity-50"
                        >
                            Previous
                        </button>
                        <span className="text-sm text-gray-500">
                            Page {page} of {totalPages}
                        </span>
                        <button
                            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                            disabled={page === totalPages}
                            className="px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg disabled:opacity-50"
                        >
                            Next
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}
