'use client'

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { getApiBaseUrl } from '@/utils/api'

interface Notification {
    id: string
    type: string
    title: string
    message?: string
    action_url?: string
    is_read: boolean
    created_at: string
    source_id?: string
    source_type?: string
}

interface NotificationContextType {
    notifications: Notification[]
    unreadCount: number
    markAsRead: (id: string) => Promise<void>
    markAllAsRead: () => Promise<void>
    refreshNotifications: () => Promise<void>
}

const NotificationContext = createContext<NotificationContextType | null>(null)

export function useNotifications() {
    const context = useContext(NotificationContext)
    if (!context) {
        throw new Error('useNotifications must be used within NotificationProvider')
    }
    return context
}

interface NotificationProviderProps {
    children: React.ReactNode
}

export function NotificationProvider({ children }: NotificationProviderProps) {
    const [notifications, setNotifications] = useState<Notification[]>([])
    const [unreadCount, setUnreadCount] = useState(0)
    const eventSourceRef = useRef<EventSource | null>(null)
    const audioRef = useRef<HTMLAudioElement | null>(null)

    // Create audio element for chime
    useEffect(() => {
        audioRef.current = new Audio('/sounds/chime.mp3')
        audioRef.current.volume = 0.5
        return () => {
            audioRef.current = null
        }
    }, [])

    // Play notification sound
    const playChime = useCallback(() => {
        if (audioRef.current) {
            audioRef.current.currentTime = 0
            audioRef.current.play().catch(e => {
                // Autoplay might be blocked, ignore
                console.log('Chime blocked by browser:', e)
            })
        }
    }, [])

    // Show toast notification
    const showToast = useCallback((notification: Notification) => {
        // Create toast element
        const toast = document.createElement('div')
        toast.className = 'fixed bottom-4 right-4 bg-white rounded-lg shadow-xl border border-gray-200 p-4 max-w-sm z-[9999] animate-slide-up'
        toast.innerHTML = `
            <div class="flex items-start gap-3">
                <div class="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                    <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
                    </svg>
                </div>
                <div class="flex-1 min-w-0">
                    <p class="font-medium text-gray-900 text-sm">${notification.title}</p>
                    ${notification.message ? `<p class="text-gray-600 text-xs mt-1 truncate">${notification.message}</p>` : ''}
                </div>
                <button onclick="this.parentElement.parentElement.remove()" class="text-gray-400 hover:text-gray-600">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `
        document.body.appendChild(toast)

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.classList.add('animate-fade-out')
            setTimeout(() => toast.remove(), 300)
        }, 5000)
    }, [])

    // Fetch notifications
    const refreshNotifications = useCallback(async () => {
        try {
            const token = localStorage.getItem('accessToken')
            if (!token) return

            const res = await fetch(`${getApiBaseUrl()}/api/notifications/recent?limit=10`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })

            if (res.ok) {
                const data = await res.json()
                setNotifications(data.notifications || [])
                setUnreadCount(data.unread_count || 0)
            }
        } catch (error) {
            console.error('Failed to fetch notifications:', error)
        }
    }, [])

    // Mark notification as read
    const markAsRead = useCallback(async (id: string) => {
        try {
            const token = localStorage.getItem('accessToken')
            await fetch(`${getApiBaseUrl()}/api/notifications/${id}/read`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            })

            setNotifications(prev => prev.map(n =>
                n.id === id ? { ...n, is_read: true } : n
            ))
            setUnreadCount(prev => Math.max(0, prev - 1))
        } catch (error) {
            console.error('Failed to mark as read:', error)
        }
    }, [])

    // Mark all as read
    const markAllAsRead = useCallback(async () => {
        try {
            const token = localStorage.getItem('accessToken')
            await fetch(`${getApiBaseUrl()}/api/notifications/read-all`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            })

            setNotifications(prev => prev.map(n => ({ ...n, is_read: true })))
            setUnreadCount(0)
        } catch (error) {
            console.error('Failed to mark all as read:', error)
        }
    }, [])

    // Connect to SSE stream
    useEffect(() => {
        const token = localStorage.getItem('accessToken')
        if (!token) return

        // Initial fetch
        refreshNotifications()

        // Connect to SSE
        const connectSSE = () => {
            const eventSource = new EventSource(
                `${getApiBaseUrl()}/api/notifications/stream?token=${token}`
            )

            eventSource.onmessage = (event) => {
                try {
                    const notification: Notification = JSON.parse(event.data)

                    // Add to notifications list
                    setNotifications(prev => [notification, ...prev.slice(0, 9)])
                    setUnreadCount(prev => prev + 1)

                    // Play chime and show toast
                    playChime()
                    showToast(notification)
                } catch (e) {
                    // Ignore heartbeat messages
                }
            }

            eventSource.onerror = () => {
                eventSource.close()
                // Reconnect after 5 seconds
                setTimeout(connectSSE, 5000)
            }

            eventSourceRef.current = eventSource
        }

        connectSSE()

        return () => {
            eventSourceRef.current?.close()
        }
    }, [refreshNotifications, playChime, showToast])

    return (
        <NotificationContext.Provider value={{
            notifications,
            unreadCount,
            markAsRead,
            markAllAsRead,
            refreshNotifications
        }}>
            {children}
        </NotificationContext.Provider>
    )
}
