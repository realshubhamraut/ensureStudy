'use client'

import { useState, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import {
    ChevronLeftIcon,
    ChevronRightIcon,
    ClockIcon,
    TrashIcon,
    CheckCircleIcon,
    CalendarDaysIcon,
    Squares2X2Icon,
    ViewColumnsIcon,
    CalendarIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface ScheduleEntry {
    id: string
    classroom_topic_id: string
    topic_name: string
    topic_description?: string
    subject_name: string
    chapter_name: string
    scheduled_date: string
    estimated_hours: number
    status: 'scheduled' | 'in_progress' | 'completed' | 'skipped'
}

type ViewMode = 'today' | '3day' | 'weekly'

interface DroppedTopic {
    topic: {
        id: string
        name: string
        estimated_hours: number
    }
    subjectName: string
    chapterName: string
}

interface StudyCalendarProps {
    onScheduleChange?: () => void
    selectedClassroomIds?: string[] // Empty array = All
    selectedClassroomNames?: string[] // Classroom names for filtering entries
}

// Types for drag data
interface DragData {
    type: 'new_topic' | 'reschedule'
    // For new topics
    topic?: {
        id: string
        name: string
        estimated_hours: number
    }
    subjectName?: string
    chapterName?: string
    // For rescheduling
    entryId?: string
}

// ============================================================================
// Helpers
// ============================================================================

function getWeekDates(weekOffset: number = 0): Date[] {
    const today = new Date()
    const monday = new Date(today)
    monday.setDate(today.getDate() - today.getDay() + 1 + (weekOffset * 7))

    const dates: Date[] = []
    for (let i = 0; i < 7; i++) {
        const date = new Date(monday)
        date.setDate(monday.getDate() + i)
        dates.push(date)
    }
    return dates
}

function getDatesByMode(mode: ViewMode, dayOffset: number = 0): Date[] {
    const today = new Date()
    const dates: Date[] = []

    if (mode === 'today') {
        const date = new Date(today)
        date.setDate(today.getDate() + dayOffset)
        dates.push(date)
    } else if (mode === '3day') {
        for (let i = 0; i < 3; i++) {
            const date = new Date(today)
            date.setDate(today.getDate() + dayOffset + i)
            dates.push(date)
        }
    } else {
        // Weekly - use existing logic
        const monday = new Date(today)
        monday.setDate(today.getDate() - today.getDay() + 1 + (dayOffset * 7))
        for (let i = 0; i < 7; i++) {
            const date = new Date(monday)
            date.setDate(monday.getDate() + i)
            dates.push(date)
        }
    }

    return dates
}

function formatDate(date: Date): string {
    return date.toISOString().split('T')[0]
}

function formatDayHeader(date: Date): string {
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

function isToday(date: Date): boolean {
    const today = new Date()
    return date.toDateString() === today.toDateString()
}

const STATUS_COLORS: Record<string, string> = {
    scheduled: 'bg-blue-50 border-blue-200',
    in_progress: 'bg-yellow-50 border-yellow-200',
    completed: 'bg-green-50 border-green-200',
    skipped: 'bg-gray-50 border-gray-200'
}

// ============================================================================
// Main Component
// ============================================================================

export default function StudyCalendar({ onScheduleChange, selectedClassroomIds = [], selectedClassroomNames = [] }: StudyCalendarProps) {
    const [viewMode, setViewMode] = useState<ViewMode>('weekly')
    const [dayOffset, setDayOffset] = useState(0)
    const [weekOffset, setWeekOffset] = useState(0)
    const [schedule, setSchedule] = useState<Record<string, ScheduleEntry[]>>({})
    const [loading, setLoading] = useState(true)
    const [dragOverDate, setDragOverDate] = useState<string | null>(null)
    const [draggingEntryId, setDraggingEntryId] = useState<string | null>(null)

    // Get dates based on current mode
    const displayDates = viewMode === 'weekly'
        ? getWeekDates(weekOffset)
        : getDatesByMode(viewMode, dayOffset)

    const startDate = formatDate(displayDates[0])
    const endDate = formatDate(displayDates[displayDates.length - 1])

    // Fetch schedule
    useEffect(() => {
        fetchSchedule()
    }, [weekOffset])

    const fetchSchedule = async () => {
        setLoading(true)
        try {
            const res = await fetch(
                `${getApiBaseUrl()}/api/curriculum/study-schedule?start_date=${startDate}&end_date=${endDate}`,
                {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                }
            )
            if (res.ok) {
                const data = await res.json()
                setSchedule(data.schedule || {})
            }
        } catch (error) {
            console.error('Failed to fetch schedule:', error)
        } finally {
            setLoading(false)
        }
    }

    // Filter schedule entries based on selected classroom names
    const getFilteredDayEntries = (dateStr: string): ScheduleEntry[] => {
        const entries = schedule[dateStr] || []
        if (selectedClassroomNames.length === 0) return entries
        return entries.filter(e =>
            selectedClassroomNames.includes(e.subject_name)
        )
    }

    const handleDrop = async (e: React.DragEvent, dateStr: string) => {
        e.preventDefault()
        setDragOverDate(null)
        setDraggingEntryId(null)

        try {
            const data = e.dataTransfer.getData('application/json')
            if (!data) return

            const dragData = JSON.parse(data)

            // Check if this is a reschedule operation
            if (dragData.type === 'reschedule' && dragData.entryId) {
                // Update the schedule entry with new date
                const res = await fetch(`${getApiBaseUrl()}/api/curriculum/study-schedule/${dragData.entryId}`, {
                    method: 'PUT',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ scheduled_date: dateStr })
                })

                if (res.ok) {
                    fetchSchedule()
                    onScheduleChange?.()
                }
            } else if (dragData.topic) {
                // It's a new topic from the sidebar
                const res = await fetch(`${getApiBaseUrl()}/api/curriculum/study-schedule`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        classroom_topic_id: dragData.topic.id,
                        scheduled_date: dateStr
                    })
                })

                if (res.ok) {
                    fetchSchedule()
                    onScheduleChange?.()
                }
            }
        } catch (error) {
            console.error('Failed to handle drop:', error)
        }
    }

    const handleDragOver = (e: React.DragEvent, dateStr: string) => {
        e.preventDefault()
        e.dataTransfer.dropEffect = 'move'
        setDragOverDate(dateStr)
    }

    const handleDragLeave = () => {
        setDragOverDate(null)
    }

    // Handle dragging an existing entry to reschedule
    const handleEntryDragStart = (e: React.DragEvent, entry: ScheduleEntry) => {
        const dragData: DragData = {
            type: 'reschedule',
            entryId: entry.id
        }
        e.dataTransfer.setData('application/json', JSON.stringify(dragData))
        e.dataTransfer.effectAllowed = 'move'
        setDraggingEntryId(entry.id)
    }

    const handleEntryDragEnd = () => {
        setDraggingEntryId(null)
    }

    const removeEntry = async (entryId: string) => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/curriculum/study-schedule/${entryId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                fetchSchedule()
                onScheduleChange?.()
            }
        } catch (error) {
            console.error('Failed to remove entry:', error)
        }
    }

    const markComplete = async (entry: ScheduleEntry) => {
        try {
            const newStatus = entry.status === 'completed' ? 'scheduled' : 'completed'
            const res = await fetch(`${getApiBaseUrl()}/api/curriculum/study-schedule/${entry.id}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ status: newStatus })
            })

            if (res.ok) {
                fetchSchedule()
                onScheduleChange?.()
            }
        } catch (error) {
            console.error('Failed to update entry:', error)
        }
    }

    // Calculate total hours per day (filtered)
    const getDayHours = (dateStr: string): number => {
        const entries = getFilteredDayEntries(dateStr)
        return entries.reduce((sum, e) => sum + (e.estimated_hours || 0), 0)
    }

    return (
        <div className="h-full flex flex-col bg-white rounded-xl border shadow-sm">
            {/* Header */}
            <div className="p-4 border-b flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <CalendarDaysIcon className="w-5 h-5 text-indigo-600" />
                    <h2 className="font-semibold text-gray-900">Study Calendar</h2>
                </div>

                {/* View Mode Toggle */}
                <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-0.5">
                    <button
                        onClick={() => { setViewMode('today'); setDayOffset(0); }}
                        className={`flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-md transition-all ${viewMode === 'today'
                            ? 'bg-white text-indigo-600 shadow-sm'
                            : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        <CalendarIcon className="w-3.5 h-3.5" />
                        Today
                    </button>
                    <button
                        onClick={() => { setViewMode('3day'); setDayOffset(0); }}
                        className={`flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-md transition-all ${viewMode === '3day'
                            ? 'bg-white text-indigo-600 shadow-sm'
                            : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        <ViewColumnsIcon className="w-3.5 h-3.5" />
                        3-Day
                    </button>
                    <button
                        onClick={() => { setViewMode('weekly'); setWeekOffset(0); }}
                        className={`flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-md transition-all ${viewMode === 'weekly'
                            ? 'bg-white text-indigo-600 shadow-sm'
                            : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        <Squares2X2Icon className="w-3.5 h-3.5" />
                        Weekly
                    </button>
                </div>

                {/* Navigation */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => viewMode === 'weekly'
                            ? setWeekOffset(prev => prev - 1)
                            : setDayOffset(prev => prev - (viewMode === 'today' ? 1 : 3))
                        }
                        className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500"
                    >
                        <ChevronLeftIcon className="w-5 h-5" />
                    </button>
                    <button
                        onClick={() => { setWeekOffset(0); setDayOffset(0); }}
                        className="px-3 py-1 text-sm font-medium text-indigo-600 hover:bg-indigo-50 rounded-lg"
                    >
                        Today
                    </button>
                    <button
                        onClick={() => viewMode === 'weekly'
                            ? setWeekOffset(prev => prev + 1)
                            : setDayOffset(prev => prev + (viewMode === 'today' ? 1 : 3))
                        }
                        className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500"
                    >
                        <ChevronRightIcon className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Calendar Grid */}
            <div className="flex-1 overflow-auto p-4">
                {loading ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                    </div>
                ) : (
                    <div className={`grid gap-2 h-full ${viewMode === 'today' ? 'grid-cols-1' :
                        viewMode === '3day' ? 'grid-cols-3' : 'grid-cols-7'
                        }`}>
                        {displayDates.map(date => {
                            const dateStr = formatDate(date)
                            const dayEntries = getFilteredDayEntries(dateStr)
                            const totalHours = getDayHours(dateStr)
                            const isDragOver = dragOverDate === dateStr
                            const isTodayDate = isToday(date)
                            const isDetailedView = viewMode !== 'weekly'

                            return (
                                <div
                                    key={dateStr}
                                    onDrop={(e) => handleDrop(e, dateStr)}
                                    onDragOver={(e) => handleDragOver(e, dateStr)}
                                    onDragLeave={handleDragLeave}
                                    className={`
                                        flex flex-col rounded-lg border-2 transition-all min-h-[300px]
                                        ${isDragOver ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200'}
                                        ${isTodayDate ? 'ring-2 ring-indigo-500' : ''}
                                    `}
                                >
                                    {/* Day Header */}
                                    <div className={`
                                        p-2 text-center border-b
                                        ${isTodayDate ? 'bg-indigo-100' : 'bg-gray-50'}
                                    `}>
                                        <p className={`text-xs font-medium ${isTodayDate ? 'text-indigo-700' : 'text-gray-600'}`}>
                                            {formatDayHeader(date)}
                                        </p>
                                        {totalHours > 0 && (
                                            <p className="text-xs text-gray-400 mt-0.5">
                                                {totalHours.toFixed(1)}h planned
                                            </p>
                                        )}
                                    </div>

                                    {/* Entries */}
                                    <div className="flex-1 p-1.5 space-y-1.5 overflow-y-auto">
                                        {dayEntries.length === 0 && (
                                            <div className="h-full flex items-center justify-center">
                                                <p className="text-xs text-gray-300 text-center">
                                                    Drop topics here
                                                </p>
                                            </div>
                                        )}

                                        {dayEntries.map(entry => (
                                            <div
                                                key={entry.id}
                                                draggable
                                                onDragStart={(e) => handleEntryDragStart(e, entry)}
                                                onDragEnd={handleEntryDragEnd}
                                                className={`
                                                    ${isDetailedView ? 'p-3' : 'p-2'} rounded-lg border group relative cursor-grab active:cursor-grabbing
                                                    ${STATUS_COLORS[entry.status]}
                                                    ${entry.status === 'completed' ? 'opacity-75' : ''}
                                                    ${draggingEntryId === entry.id ? 'opacity-50 ring-2 ring-indigo-400' : ''}
                                                `}
                                            >
                                                {/* Topic Name */}
                                                <p className={`
                                                    font-medium text-gray-800 leading-tight
                                                    ${isDetailedView ? 'text-sm' : 'text-xs'}
                                                    ${entry.status === 'completed' ? 'line-through' : ''}
                                                `}>
                                                    {entry.topic_name}
                                                </p>

                                                {/* Description / Subtopics */}
                                                {entry.topic_description && (
                                                    <p className={`text-gray-500 mt-1 ${isDetailedView ? 'text-xs' : 'text-[10px] truncate'}`}>
                                                        {isDetailedView ? entry.topic_description : entry.topic_description.slice(0, 50) + '...'}
                                                    </p>
                                                )}

                                                {/* Chapter (show in detailed view) */}
                                                {isDetailedView && entry.chapter_name && (
                                                    <p className="text-xs text-gray-600 mt-1">
                                                        📖 {entry.chapter_name}
                                                    </p>
                                                )}

                                                {/* Subject */}
                                                <p className={`text-gray-500 truncate mt-0.5 ${isDetailedView ? 'text-xs' : 'text-[10px]'}`}>
                                                    {entry.subject_name}
                                                </p>

                                                {/* Time */}
                                                <div className={`flex items-center gap-1 mt-1 text-gray-400 ${isDetailedView ? 'text-xs' : 'text-[10px]'}`}>
                                                    <ClockIcon className={isDetailedView ? 'w-4 h-4' : 'w-3 h-3'} />
                                                    <span>{entry.estimated_hours}h estimated</span>
                                                </div>

                                                {/* Actions (show on hover) */}
                                                <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity flex gap-0.5">
                                                    <button
                                                        onClick={() => markComplete(entry)}
                                                        className={`p-1 rounded hover:bg-white/80 ${entry.status === 'completed' ? 'text-green-600' : 'text-gray-400'}`}
                                                        title={entry.status === 'completed' ? 'Mark as pending' : 'Mark complete'}
                                                    >
                                                        <CheckCircleIcon className="w-3.5 h-3.5" />
                                                    </button>
                                                    <button
                                                        onClick={() => removeEntry(entry.id)}
                                                        className="p-1 rounded hover:bg-white/80 text-gray-400 hover:text-red-600"
                                                        title="Remove"
                                                    >
                                                        <TrashIcon className="w-3.5 h-3.5" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
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
