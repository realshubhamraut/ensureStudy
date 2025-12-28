'use client'

import { useState, useEffect } from 'react'
import {
    BellIcon,
    CheckCircleIcon,
    ExclamationCircleIcon,
    InformationCircleIcon
} from '@heroicons/react/24/outline'

interface Notification {
    id: string
    type: 'info' | 'success' | 'warning'
    title: string
    message: string
    time: string
    read: boolean
}

export default function ParentNotificationsPage() {
    const [loading, setLoading] = useState(true)
    const [notifications, setNotifications] = useState<Notification[]>([])

    useEffect(() => {
        // Simulate loading
        setTimeout(() => setLoading(false), 500)
    }, [])

    const getIcon = (type: string) => {
        switch (type) {
            case 'success':
                return <CheckCircleIcon className="w-6 h-6 text-green-600" />
            case 'warning':
                return <ExclamationCircleIcon className="w-6 h-6 text-yellow-600" />
            default:
                return <InformationCircleIcon className="w-6 h-6 text-blue-600" />
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">🔔 Notifications</h1>
                    <p className="text-gray-600">Stay updated on your children's activities</p>
                </div>
                {notifications.length > 0 && (
                    <button className="text-sm text-orange-600 hover:text-orange-700">
                        Mark all as read
                    </button>
                )}
            </div>

            {notifications.length === 0 ? (
                <div className="card text-center py-12">
                    <BellIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No Notifications</h3>
                    <p className="text-gray-500 max-w-md mx-auto">
                        You'll receive notifications about your children's quiz completions, achievements, and more.
                    </p>
                </div>
            ) : (
                <div className="space-y-3">
                    {notifications.map((notif) => (
                        <div
                            key={notif.id}
                            className={`card flex items-start gap-4 ${!notif.read ? 'border-l-4 border-orange-500' : ''}`}
                        >
                            <div className="flex-shrink-0 mt-1">
                                {getIcon(notif.type)}
                            </div>
                            <div className="flex-1">
                                <h3 className="font-semibold text-gray-900">{notif.title}</h3>
                                <p className="text-sm text-gray-600 mt-1">{notif.message}</p>
                                <p className="text-xs text-gray-400 mt-2">{notif.time}</p>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Notification Settings Hint */}
            <div className="card bg-gray-50 border-gray-200">
                <h3 className="font-semibold text-gray-900 mb-2">⚙️ Notification Preferences</h3>
                <p className="text-sm text-gray-600">
                    Customize what notifications you receive in <a href="/parent/settings" className="text-orange-600 hover:underline">Settings</a>.
                </p>
            </div>
        </div>
    )
}
