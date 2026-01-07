'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { BellIcon, CheckIcon } from '@heroicons/react/24/outline'
import { useNotifications } from './NotificationProvider'

export function NotificationBell() {
    const router = useRouter()
    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef<HTMLDivElement>(null)
    const { notifications, unreadCount, markAsRead, markAllAsRead } = useNotifications()

    // Close dropdown on outside click
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const handleNotificationClick = (notification: typeof notifications[0]) => {
        markAsRead(notification.id)
        if (notification.action_url) {
            router.push(notification.action_url)
        }
        setIsOpen(false)
    }

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'announcement':
                return '📢'
            case 'material':
                return '📚'
            case 'assignment':
                return '📝'
            case 'meeting':
                return '📹'
            case 'result':
                return '📊'
            default:
                return '🔔'
        }
    }

    const formatTime = (dateString: string) => {
        // Backend returns UTC time - append Z if not present to parse as UTC
        const utcDateString = dateString.endsWith('Z') ? dateString : dateString + 'Z'
        const date = new Date(utcDateString)
        const now = new Date()
        const diff = now.getTime() - date.getTime()
        const minutes = Math.floor(diff / 60000)
        const hours = Math.floor(diff / 3600000)
        const days = Math.floor(diff / 86400000)

        if (minutes < 1) return 'Just now'
        if (minutes < 60) return `${minutes}m ago`
        if (hours < 24) return `${hours}h ago`
        return `${days}d ago`
    }

    return (
        <div className="relative" ref={dropdownRef}>
            {/* Bell Icon Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="relative p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
                <BellIcon className="w-6 h-6 text-gray-600" />
                {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
                        {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                )}
            </button>

            {/* Dropdown */}
            {isOpen && (
                <div className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-xl border border-gray-200 z-50 overflow-hidden">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 bg-gray-50">
                        <h3 className="font-semibold text-gray-900">Notifications</h3>
                        {unreadCount > 0 && (
                            <button
                                onClick={markAllAsRead}
                                className="text-xs text-indigo-600 hover:text-indigo-700 flex items-center gap-1"
                            >
                                <CheckIcon className="w-3 h-3" />
                                Mark all read
                            </button>
                        )}
                    </div>

                    {/* Notification List */}
                    <div className="max-h-96 overflow-y-auto">
                        {notifications.length === 0 ? (
                            <div className="py-8 text-center text-gray-500">
                                <BellIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                                <p className="text-sm">No notifications yet</p>
                            </div>
                        ) : (
                            notifications.map((notification) => (
                                <button
                                    key={notification.id}
                                    onClick={() => handleNotificationClick(notification)}
                                    className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors border-b border-gray-50 last:border-0 ${!notification.is_read ? 'bg-indigo-50/50' : ''
                                        }`}
                                >
                                    <div className="flex gap-3">
                                        <span className="text-xl">{getTypeIcon(notification.type)}</span>
                                        <div className="flex-1 min-w-0">
                                            <p className={`text-sm ${!notification.is_read ? 'font-semibold text-gray-900' : 'text-gray-700'}`}>
                                                {notification.title}
                                            </p>
                                            {notification.message && (
                                                <p className="text-xs text-gray-500 truncate mt-0.5">
                                                    {notification.message}
                                                </p>
                                            )}
                                            <p className="text-xs text-gray-400 mt-1">
                                                {formatTime(notification.created_at)}
                                            </p>
                                        </div>
                                        {!notification.is_read && (
                                            <span className="w-2 h-2 bg-indigo-500 rounded-full flex-shrink-0 mt-2" />
                                        )}
                                    </div>
                                </button>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
