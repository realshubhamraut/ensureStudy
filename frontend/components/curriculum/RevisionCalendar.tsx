'use client'

import { useState, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import {
    ChevronLeftIcon,
    ChevronRightIcon,
    SparklesIcon,
    ArrowPathIcon,
    CheckCircleIcon,
    ClockIcon,
    ExclamationCircleIcon,
    BookOpenIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface RevisionEntry {
    topic_id: string
    topic_name: string
    subject_name: string
    chapter_name: string
    chapter_color: string
    mastery_percentage: number
    quiz_score: number
    interview_score: number
    review_count: number
    scheduled_date: string
    status: 'overdue' | 'due' | 'upcoming' | 'scheduled' | 'new' | 'completed'
    priority: number
    last_activity: string | null
    source: 'assessment' | 'suggested'
}

interface RevisionScheduleResponse {
    week_start: string
    week_end: string
    schedule: { [date: string]: RevisionEntry[] }
    stats: {
        topics_due: number
        topics_overdue: number
        topics_scheduled: number
        avg_mastery: number
    }
}

interface RevisionCalendarProps {
    selectedClassroomIds?: string[]
    selectedClassroomNames?: string[]
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
    switch (status) {
        case 'overdue': return 'bg-red-50 border-red-200 border-l-red-500'
        case 'due': return 'bg-orange-50 border-orange-200 border-l-orange-500'
        case 'upcoming': return 'bg-purple-50 border-purple-200 border-l-purple-500'
        case 'new': return 'bg-blue-50 border-blue-200 border-l-blue-500'
        case 'completed': return 'bg-green-50 border-green-200 border-l-green-500'
        default: return 'bg-gray-50 border-gray-200 border-l-gray-400'
    }
}

function getStatusBadge(status: string): { text: string, color: string } {
    switch (status) {
        case 'overdue': return { text: 'Overdue', color: 'bg-red-100 text-red-700' }
        case 'due': return { text: 'Due', color: 'bg-orange-100 text-orange-700' }
        case 'upcoming': return { text: 'Soon', color: 'bg-purple-100 text-purple-700' }
        case 'new': return { text: 'New', color: 'bg-blue-100 text-blue-700' }
        case 'completed': return { text: 'Done', color: 'bg-green-100 text-green-700' }
        default: return { text: 'Scheduled', color: 'bg-gray-100 text-gray-700' }
    }
}

function getMasteryColor(mastery: number): string {
    if (mastery >= 80) return 'text-green-600'
    if (mastery >= 60) return 'text-blue-600'
    if (mastery >= 40) return 'text-yellow-600'
    if (mastery > 0) return 'text-red-600'
    return 'text-gray-400'
}

function formatDayHeader(dateStr: string): string {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

function isToday(dateStr: string): boolean {
    const today = new Date().toISOString().split('T')[0]
    return dateStr === today
}

// ============================================================================
// Main Component
// ============================================================================

export default function RevisionCalendar({
    selectedClassroomIds = [],
    selectedClassroomNames = []
}: RevisionCalendarProps) {
    const [weekOffset, setWeekOffset] = useState(0)
    const [schedule, setSchedule] = useState<{ [date: string]: RevisionEntry[] }>({})
    const [stats, setStats] = useState({ topics_due: 0, topics_overdue: 0, topics_scheduled: 0, avg_mastery: 0 })
    const [weekStart, setWeekStart] = useState('')
    const [weekEnd, setWeekEnd] = useState('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    // Fetch revision schedule from API
    useEffect(() => {
        const fetchSchedule = async () => {
            setLoading(true)
            setError(null)

            try {
                const token = localStorage.getItem('accessToken')
                const params = new URLSearchParams()
                params.append('week_offset', weekOffset.toString())

                if (selectedClassroomIds.length > 0) {
                    params.append('classroom_ids', selectedClassroomIds.join(','))
                }

                const res = await fetch(
                    `${getApiBaseUrl()}/api/curriculum/revision-schedule?${params.toString()}`,
                    {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    }
                )

                if (!res.ok) {
                    throw new Error('Failed to fetch revision schedule')
                }

                const data: RevisionScheduleResponse = await res.json()
                setSchedule(data.schedule)
                setStats(data.stats)
                setWeekStart(data.week_start)
                setWeekEnd(data.week_end)
            } catch (err) {
                console.error('Failed to fetch revision schedule:', err)
                setError('Failed to load revision schedule')
            } finally {
                setLoading(false)
            }
        }

        fetchSchedule()
    }, [weekOffset, selectedClassroomIds])

    const handleMarkComplete = async (topicId: string) => {
        try {
            const token = localStorage.getItem('accessToken')
            const res = await fetch(
                `${getApiBaseUrl()}/api/curriculum/revision-schedule/mark-complete`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ topic_id: topicId })
                }
            )

            if (res.ok) {
                // Update local state to mark as completed
                setSchedule(prev => {
                    const updated = { ...prev }
                    Object.keys(updated).forEach(date => {
                        updated[date] = updated[date].map(entry =>
                            entry.topic_id === topicId
                                ? { ...entry, status: 'completed' as const }
                                : entry
                        )
                    })
                    return updated
                })
            }
        } catch (err) {
            console.error('Failed to mark complete:', err)
        }
    }

    // Get sorted dates for display
    const sortedDates = Object.keys(schedule).sort()

    return (
        <div className="h-full flex flex-col bg-white rounded-xl border border-purple-200 shadow-sm overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-purple-50 to-indigo-50 border-b border-purple-100">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center">
                        <SparklesIcon className="w-4 h-4 text-white" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-gray-800">AI Revision Schedule</h3>
                        <p className="text-xs text-gray-500">
                            {stats.topics_scheduled > 0
                                ? `${stats.topics_due} due · ${stats.topics_overdue} overdue · ${stats.avg_mastery}% avg mastery`
                                : 'Complete assessments to generate schedule'
                            }
                        </p>
                    </div>
                </div>

                {/* Week Navigation */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setWeekOffset(0)}
                        className="px-2 py-1 text-xs text-purple-600 hover:bg-purple-100 rounded transition-colors"
                    >
                        Today
                    </button>
                    <button
                        onClick={() => setWeekOffset(prev => prev - 1)}
                        className="p-1.5 text-gray-500 hover:bg-purple-100 rounded transition-colors"
                    >
                        <ChevronLeftIcon className="w-4 h-4" />
                    </button>
                    <span className="text-xs text-gray-500 min-w-[100px] text-center">
                        {weekStart && weekEnd
                            ? `${new Date(weekStart).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${new Date(weekEnd).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
                            : ''
                        }
                    </span>
                    <button
                        onClick={() => setWeekOffset(prev => prev + 1)}
                        className="p-1.5 text-gray-500 hover:bg-purple-100 rounded transition-colors"
                    >
                        <ChevronRightIcon className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Calendar Grid */}
            <div className="flex-1 overflow-auto">
                {loading ? (
                    <div className="flex items-center justify-center h-full">
                        <ArrowPathIcon className="w-6 h-6 text-purple-400 animate-spin" />
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                        <ExclamationCircleIcon className="w-8 h-8 text-gray-300 mb-2" />
                        <p className="text-sm">{error}</p>
                    </div>
                ) : sortedDates.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500 p-4">
                        <BookOpenIcon className="w-10 h-10 text-gray-300 mb-2" />
                        <p className="text-sm font-medium">No revision topics scheduled</p>
                        <p className="text-xs text-gray-400 text-center mt-1">
                            Complete assessments or interviews to generate AI revision recommendations
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-7 h-full divide-x divide-purple-100">
                        {sortedDates.map(dateStr => {
                            const entries = schedule[dateStr] || []
                            const todayClass = isToday(dateStr) ? 'bg-purple-50' : ''

                            return (
                                <div key={dateStr} className={`flex flex-col min-h-[120px] ${todayClass}`}>
                                    {/* Day Header */}
                                    <div className={`px-2 py-1.5 text-center border-b border-purple-100 ${isToday(dateStr) ? 'bg-purple-100' : 'bg-gray-50'
                                        }`}>
                                        <span className={`text-xs font-medium ${isToday(dateStr) ? 'text-purple-700' : 'text-gray-600'
                                            }`}>
                                            {formatDayHeader(dateStr)}
                                        </span>
                                    </div>

                                    {/* Entries */}
                                    <div className="flex-1 p-1 space-y-1 overflow-y-auto">
                                        {entries.map(entry => {
                                            const statusBadge = getStatusBadge(entry.status)

                                            return (
                                                <div
                                                    key={entry.topic_id}
                                                    className={`p-1.5 rounded border-l-2 text-xs cursor-pointer hover:shadow-sm transition-shadow ${getStatusColor(entry.status)}`}
                                                    onClick={() => entry.status !== 'completed' && handleMarkComplete(entry.topic_id)}
                                                    title={`${entry.topic_name}\nMastery: ${entry.mastery_percentage}%\nQuiz: ${entry.quiz_score}% | Interview: ${entry.interview_score}%\nClick to mark complete`}
                                                >
                                                    <div className="flex items-start justify-between gap-1">
                                                        <span className="font-medium text-gray-800 truncate leading-tight flex-1">
                                                            {entry.topic_name}
                                                        </span>
                                                        {entry.status === 'completed' ? (
                                                            <CheckCircleIcon className="w-3.5 h-3.5 text-green-500 flex-shrink-0" />
                                                        ) : (
                                                            <span className={`px-1 py-0.5 rounded text-[9px] font-medium ${statusBadge.color}`}>
                                                                {statusBadge.text}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="flex items-center gap-1 mt-0.5">
                                                        <span
                                                            className="w-2 h-2 rounded-full flex-shrink-0"
                                                            style={{ backgroundColor: entry.chapter_color }}
                                                        />
                                                        <span className="text-gray-500 truncate">{entry.subject_name}</span>
                                                        <span className={`ml-auto font-semibold ${getMasteryColor(entry.mastery_percentage)}`}>
                                                            {entry.mastery_percentage > 0 ? `${entry.mastery_percentage}%` : 'New'}
                                                        </span>
                                                    </div>
                                                </div>
                                            )
                                        })}

                                        {entries.length === 0 && (
                                            <div className="flex items-center justify-center h-full text-gray-300">
                                                <ClockIcon className="w-4 h-4" />
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                )}
            </div>
        </div>
    )
}
