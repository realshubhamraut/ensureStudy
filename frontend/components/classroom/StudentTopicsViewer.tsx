'use client'

import { useState, useEffect } from 'react'
import { getAiServiceUrl, getApiBaseUrl } from '@/utils/api'
import {
    AcademicCapIcon,
    BookOpenIcon,
    ChevronDownIcon,
    ChevronRightIcon,
    ArrowPathIcon,
    ClockIcon,
    SparklesIcon,
    CheckCircleIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface Topic {
    id: string
    name: string
    description: string
    difficulty: 'easy' | 'medium' | 'hard'
    estimated_hours: number
    key_concepts: string[]
    order: number
    mastery_percentage?: number  // From StudentTopicScore
}

interface Chapter {
    id: string
    name: string
    description: string
    color: string
    order: number
    topics: Topic[]
}

interface Props {
    classroomId: string
}

// ============================================================================
// Constants
// ============================================================================

const DIFFICULTIES = [
    { value: 'easy', label: 'Easy', color: 'bg-green-100 text-green-700' },
    { value: 'medium', label: 'Medium', color: 'bg-yellow-100 text-yellow-700' },
    { value: 'hard', label: 'Hard', color: 'bg-red-100 text-red-700' }
]

// ============================================================================
// Main Component
// ============================================================================

export default function StudentTopicsViewer({ classroomId }: Props) {
    // State
    const [chapters, setChapters] = useState<Chapter[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [expandedChapters, setExpandedChapters] = useState<Set<string>>(new Set())

    // ========================================================================
    // Fetch Data
    // ========================================================================

    useEffect(() => {
        fetchHierarchy()
    }, [classroomId])

    const fetchHierarchy = async () => {
        setLoading(true)
        setError('')
        try {
            // Fetch hierarchy and mastery data in parallel
            const [hierarchyRes, masteryRes] = await Promise.all([
                fetch(`${getAiServiceUrl()}/api/classroom-syllabus/hierarchy/${classroomId}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                }),
                fetch(`${getApiBaseUrl()}/api/progress/topic-mastery?classroom_id=${classroomId}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
            ])

            // Build mastery map: topic_id -> mastery_level
            const masteryMap: Record<string, number> = {}
            if (masteryRes.ok) {
                const masteryData = await masteryRes.json()
                for (const t of masteryData.topics || []) {
                    masteryMap[t.topic_id] = t.mastery_level
                }
            }

            if (hierarchyRes.ok) {
                const data = await hierarchyRes.json()
                // Merge mastery data into topics
                const chaptersWithMastery = (data.chapters || []).map((ch: Chapter) => ({
                    ...ch,
                    topics: ch.topics.map(t => ({
                        ...t,
                        mastery_percentage: masteryMap[t.id] ?? 0
                    }))
                }))
                setChapters(chaptersWithMastery)
                // Auto-expand all chapters for students
                if (chaptersWithMastery.length > 0) {
                    setExpandedChapters(new Set(chaptersWithMastery.map((c: Chapter) => c.id)))
                }
            } else if (hierarchyRes.status === 404) {
                // No hierarchy yet
                setChapters([])
            } else {
                setError('Failed to load topics')
            }
        } catch (e) {
            console.error('Fetch error:', e)
            setError('Failed to load topics')
        } finally {
            setLoading(false)
        }
    }

    // ========================================================================
    // Toggle Chapter
    // ========================================================================

    const toggleChapter = (chapterId: string) => {
        setExpandedChapters(prev => {
            const newSet = new Set(prev)
            if (newSet.has(chapterId)) {
                newSet.delete(chapterId)
            } else {
                newSet.add(chapterId)
            }
            return newSet
        })
    }

    // ========================================================================
    // Stats
    // ========================================================================

    const totalTopics = chapters.reduce((sum, ch) => sum + ch.topics.length, 0)
    const totalHours = chapters.reduce((sum, ch) =>
        sum + ch.topics.reduce((tSum, t) => tSum + (t.estimated_hours || 0), 0), 0
    )

    // ========================================================================
    // Render
    // ========================================================================

    if (loading) {
        return (
            <div className="p-8 text-center">
                <ArrowPathIcon className="w-8 h-8 text-indigo-500 animate-spin mx-auto mb-2" />
                <p className="text-gray-500">Loading topics...</p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-indigo-100">
                    <AcademicCapIcon className="w-5 h-5 text-indigo-600" />
                </div>
                <div>
                    <h3 className="font-semibold text-gray-900">Course Topics</h3>
                    <p className="text-sm text-gray-500">
                        {chapters.length} chapters • {totalTopics} topics • {totalHours.toFixed(1)}h total
                    </p>
                </div>
            </div>

            {error && (
                <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>
            )}

            {/* Chapters List */}
            <div className="space-y-3">
                {chapters.map((chapter, chapterIndex) => {
                    const isExpanded = expandedChapters.has(chapter.id)

                    return (
                        <div
                            key={chapter.id}
                            className="border rounded-xl overflow-hidden shadow-sm"
                            style={{ borderColor: chapter.color + '40' }}
                        >
                            {/* Chapter Header */}
                            <button
                                onClick={() => toggleChapter(chapter.id)}
                                className="w-full flex items-center gap-3 p-4 transition-colors hover:bg-gray-50"
                                style={{ backgroundColor: chapter.color + '08' }}
                            >
                                <div
                                    className="w-2 h-12 rounded-full flex-shrink-0"
                                    style={{ backgroundColor: chapter.color }}
                                />

                                {isExpanded ? (
                                    <ChevronDownIcon className="w-5 h-5 text-gray-500 flex-shrink-0" />
                                ) : (
                                    <ChevronRightIcon className="w-5 h-5 text-gray-500 flex-shrink-0" />
                                )}

                                <div className="flex-1 text-left">
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-medium text-gray-400">
                                            Chapter {chapterIndex + 1}
                                        </span>
                                    </div>
                                    <p className="font-semibold text-gray-900">{chapter.name}</p>
                                    <p className="text-xs text-gray-500">
                                        {chapter.topics.length} topics
                                    </p>
                                </div>

                                <div className="px-3 py-1 rounded-full bg-gray-100 text-xs font-medium text-gray-600">
                                    {chapter.topics.reduce((sum, t) => sum + (t.estimated_hours || 0), 0).toFixed(1)}h
                                </div>
                            </button>

                            {/* Topics */}
                            {isExpanded && chapter.topics.length > 0 && (
                                <div className="border-t divide-y" style={{ borderColor: chapter.color + '20' }}>
                                    {chapter.topics.map((topic, topicIndex) => (
                                        <div
                                            key={topic.id}
                                            className="p-4 pl-14 bg-white hover:bg-gray-50 transition-colors"
                                        >
                                            <div className="flex items-center gap-3">
                                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center">
                                                    <span className="text-xs font-medium text-gray-500">{topicIndex + 1}</span>
                                                </div>
                                                <BookOpenIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm font-medium text-gray-800">{topic.name}</p>
                                                    {topic.description && (
                                                        <p className="text-xs text-gray-500 truncate">{topic.description}</p>
                                                    )}
                                                </div>
                                                <span className={`text-xs px-2 py-1 rounded-full ${DIFFICULTIES.find(d => d.value === topic.difficulty)?.color || 'bg-gray-100'
                                                    }`}>
                                                    {topic.difficulty}
                                                </span>
                                                {/* Mastery Badge */}
                                                <span className={`text-xs px-2 py-1 rounded-full font-medium ${topic.mastery_percentage === undefined || topic.mastery_percentage === 0
                                                        ? 'bg-gray-100 text-gray-500'
                                                        : topic.mastery_percentage >= 70
                                                            ? 'bg-green-100 text-green-700'
                                                            : topic.mastery_percentage >= 50
                                                                ? 'bg-yellow-100 text-yellow-700'
                                                                : 'bg-red-100 text-red-700'
                                                    }`}>
                                                    {topic.mastery_percentage === undefined || topic.mastery_percentage === 0
                                                        ? 'New'
                                                        : `${Math.round(topic.mastery_percentage)}%`}
                                                </span>
                                                <span className="text-xs text-gray-500 flex items-center gap-1 whitespace-nowrap">
                                                    <ClockIcon className="w-3.5 h-3.5" />
                                                    {topic.estimated_hours}h
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Empty Topics Message */}
                            {isExpanded && chapter.topics.length === 0 && (
                                <div className="p-4 pl-14 text-sm text-gray-400 bg-gray-50">
                                    No topics in this chapter yet
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            {/* Empty State */}
            {chapters.length === 0 && (
                <div className="text-center py-12 bg-gray-50 rounded-xl">
                    <SparklesIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                    <p className="font-medium text-gray-600">No course topics available yet</p>
                    <p className="text-sm text-gray-400 mt-1">Your teacher will add the course syllabus soon</p>
                </div>
            )}
        </div>
    )
}
