'use client'

import { useState } from 'react'
import {
    ChevronLeftIcon,
    ChevronRightIcon,
    ArrowPathIcon,
    CalendarDaysIcon
} from '@heroicons/react/24/outline'

interface ScheduledTopic {
    topic_id: string
    topic_name: string
    date: string
    confidence_score: number
    status: 'scheduled' | 'in_progress' | 'completed'
    unit?: string
    chapter?: string
}

interface WeeklySchedule {
    week_start: string
    week_end: string
    days: { [date: string]: ScheduledTopic[] }
}

interface WeeklyCalendarProps {
    schedule: WeeklySchedule | null
    loading: boolean
    onWeekChange: (offset: number) => void
    onReschedule: (topicId: string, newDate: string) => void
    weekOffset: number
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function getConfidenceColor(score: number): string {
    if (score >= 80) return 'bg-green-100 text-green-700 border-green-200'
    if (score >= 50) return 'bg-yellow-100 text-yellow-700 border-yellow-200'
    if (score > 0) return 'bg-red-100 text-red-700 border-red-200'
    return 'bg-gray-100 text-gray-500 border-gray-200'
}

function getConfidenceBadgeColor(score: number): string {
    if (score >= 80) return 'bg-green-500'
    if (score >= 50) return 'bg-yellow-500'
    if (score > 0) return 'bg-red-500'
    return 'bg-gray-400'
}

function formatDate(dateStr: string): string {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export default function WeeklyCalendar({
    schedule,
    loading,
    onWeekChange,
    onReschedule,
    weekOffset
}: WeeklyCalendarProps) {
    const [selectedTopic, setSelectedTopic] = useState<ScheduledTopic | null>(null)
    const [showReschedule, setShowReschedule] = useState(false)
    const [newDate, setNewDate] = useState('')

    if (loading) {
        return (
            <div className="card p-8 text-center">
                <ArrowPathIcon className="w-8 h-8 animate-spin text-gray-400 mx-auto mb-3" />
                <p className="text-gray-500">Loading schedule...</p>
            </div>
        )
    }

    if (!schedule) {
        return (
            <div className="card p-8 text-center">
                <CalendarDaysIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500">No schedule available</p>
            </div>
        )
    }

    const dates = Object.keys(schedule.days).sort()
    const today = new Date().toISOString().split('T')[0]

    const handleReschedule = () => {
        if (selectedTopic && newDate) {
            onReschedule(selectedTopic.topic_id, newDate)
            setShowReschedule(false)
            setSelectedTopic(null)
            setNewDate('')
        }
    }

    return (
        <div className="space-y-4">
            {/* Week Navigation */}
            <div className="flex items-center justify-between">
                <button
                    onClick={() => onWeekChange(weekOffset - 1)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                    <ChevronLeftIcon className="w-5 h-5 text-gray-600" />
                </button>

                <div className="text-center">
                    <h3 className="font-semibold text-gray-900">
                        {formatDate(schedule.week_start)} - {formatDate(schedule.week_end)}
                    </h3>
                    <p className="text-xs text-gray-500">
                        {weekOffset === 0 ? 'This Week' : weekOffset < 0 ? `${Math.abs(weekOffset)} week${Math.abs(weekOffset) > 1 ? 's' : ''} ago` : `${weekOffset} week${weekOffset > 1 ? 's' : ''} ahead`}
                    </p>
                </div>

                <button
                    onClick={() => onWeekChange(weekOffset + 1)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                    <ChevronRightIcon className="w-5 h-5 text-gray-600" />
                </button>
            </div>

            {/* Calendar Grid */}
            <div className="grid grid-cols-7 gap-2">
                {/* Day Headers */}
                {DAYS.map((day, i) => (
                    <div key={day} className="text-center py-2">
                        <p className="text-xs font-medium text-gray-500 uppercase">{day}</p>
                        <p className="text-sm font-semibold text-gray-700">
                            {dates[i] ? formatDate(dates[i]).split(' ')[1] : ''}
                        </p>
                    </div>
                ))}

                {/* Day Columns */}
                {dates.map((date, i) => {
                    const isToday = date === today
                    const topics = schedule.days[date] || []

                    return (
                        <div
                            key={date}
                            className={`min-h-[120px] rounded-lg p-2 ${isToday
                                    ? 'bg-primary-50 ring-2 ring-primary-500'
                                    : 'bg-gray-50'
                                }`}
                        >
                            <div className="space-y-2">
                                {topics.length === 0 ? (
                                    <p className="text-xs text-gray-400 text-center py-4">
                                        No topics
                                    </p>
                                ) : (
                                    topics.map((topic) => (
                                        <div
                                            key={topic.topic_id}
                                            onClick={() => {
                                                setSelectedTopic(topic)
                                                setShowReschedule(true)
                                            }}
                                            className={`p-2 rounded-lg border cursor-pointer hover:shadow-md transition-shadow ${getConfidenceColor(topic.confidence_score)}`}
                                        >
                                            <p className="text-xs font-medium line-clamp-2 mb-1">
                                                {topic.topic_name}
                                            </p>
                                            <div className="flex items-center gap-1">
                                                <div className={`w-2 h-2 rounded-full ${getConfidenceBadgeColor(topic.confidence_score)}`} />
                                                <span className="text-xs font-semibold">
                                                    {topic.confidence_score > 0 ? `${Math.round(topic.confidence_score)}%` : '—'}
                                                </span>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    )
                })}
            </div>

            {/* Reschedule Modal */}
            {showReschedule && selectedTopic && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-xl max-w-sm w-full p-6">
                        <h3 className="font-semibold text-gray-900 mb-2">
                            Reschedule Topic
                        </h3>
                        <p className="text-sm text-gray-600 mb-4">
                            {selectedTopic.topic_name}
                        </p>

                        <div className="mb-4">
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                New Date
                            </label>
                            <input
                                type="date"
                                value={newDate}
                                onChange={(e) => setNewDate(e.target.value)}
                                className="input-field w-full"
                            />
                        </div>

                        <div className="flex gap-2">
                            <button
                                onClick={() => setShowReschedule(false)}
                                className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleReschedule}
                                disabled={!newDate}
                                className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
                            >
                                Reschedule
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="flex items-center justify-center gap-4 pt-4 border-t">
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                    <span className="text-xs text-gray-600">Mastered (≥80%)</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <span className="text-xs text-gray-600">Learning (50-79%)</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <span className="text-xs text-gray-600">Needs Work (&lt;50%)</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-gray-400" />
                    <span className="text-xs text-gray-600">Not Started</span>
                </div>
            </div>
        </div>
    )
}
