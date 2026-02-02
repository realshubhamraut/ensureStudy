'use client'

import { useState, useEffect } from 'react'
import { getAiServiceUrl, getApiBaseUrl } from '@/utils/api'
import {
    BookOpenIcon,
    ChevronDownIcon,
    ChevronRightIcon,
    CheckCircleIcon,
    ClockIcon,
    AcademicCapIcon,
    SparklesIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface ClassroomTopic {
    id: string
    chapter_id: string
    name: string
    description?: string
    difficulty: string
    estimated_hours: number
    key_concepts: string[]
    order: number
    question_count: number
    // Student score data (if available)
    mastery_percentage?: number
    status?: 'not_started' | 'learning' | 'practicing' | 'mastered'
}

interface Chapter {
    id: string
    classroom_id: string
    name: string
    description?: string
    color: string
    estimated_hours: number
    order: number
    topic_count: number
    topics: ClassroomTopic[]
}

interface ClassroomHierarchy {
    classroom_id: string
    subject_name: string
    chapters: Chapter[]
    total_chapters: number
    total_topics: number
}

interface Props {
    classroomId: string
    userId?: string
    onTopicClick?: (topic: ClassroomTopic) => void
    showMastery?: boolean
}

// ============================================================================
// Helper Components
// ============================================================================

function MasteryBadge({ percentage, status }: { percentage: number; status?: string }) {
    const getColor = () => {
        if (percentage >= 80) return 'bg-green-100 text-green-700 border-green-200'
        if (percentage >= 50) return 'bg-yellow-100 text-yellow-700 border-yellow-200'
        if (percentage > 0) return 'bg-orange-100 text-orange-700 border-orange-200'
        return 'bg-gray-100 text-gray-500 border-gray-200'
    }

    const getIcon = () => {
        if (percentage >= 80) return <CheckCircleIcon className="w-3 h-3" />
        if (percentage >= 50) return <SparklesIcon className="w-3 h-3" />
        if (percentage > 0) return <ClockIcon className="w-3 h-3" />
        return null
    }

    return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${getColor()}`}>
            {getIcon()}
            {percentage > 0 ? `${Math.round(percentage)}%` : 'Not started'}
        </span>
    )
}

function DifficultyBadge({ difficulty }: { difficulty: string }) {
    const colors = {
        easy: 'bg-green-50 text-green-600',
        medium: 'bg-yellow-50 text-yellow-600',
        hard: 'bg-red-50 text-red-600'
    }
    return (
        <span className={`px-2 py-0.5 text-xs rounded ${colors[difficulty as keyof typeof colors] || colors.medium}`}>
            {difficulty}
        </span>
    )
}

// ============================================================================
// Main Component
// ============================================================================

export default function ClassroomTopicHierarchy({
    classroomId,
    userId,
    onTopicClick,
    showMastery = true
}: Props) {
    const [hierarchy, setHierarchy] = useState<ClassroomHierarchy | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [expandedChapters, setExpandedChapters] = useState<Set<string>>(new Set())

    // Fetch hierarchy on mount
    useEffect(() => {
        const fetchHierarchy = async () => {
            setLoading(true)
            setError(null)

            try {
                const params = new URLSearchParams()
                if (showMastery && userId) {
                    params.append('include_scores', 'true')
                    params.append('user_id', userId)
                }

                const res = await fetch(
                    `${getAiServiceUrl()}/api/classroom-syllabus/hierarchy/${classroomId}?${params}`,
                    {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                        }
                    }
                )

                if (!res.ok) {
                    throw new Error('Failed to fetch hierarchy')
                }

                const data = await res.json()
                setHierarchy(data)

                // Expand first chapter by default
                if (data.chapters?.length > 0) {
                    setExpandedChapters(new Set([data.chapters[0].id]))
                }
            } catch (err) {
                console.error('Hierarchy fetch error:', err)
                setError('Failed to load topics')
            } finally {
                setLoading(false)
            }
        }

        if (classroomId) {
            fetchHierarchy()
        }
    }, [classroomId, userId, showMastery])

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

    // Loading state
    if (loading) {
        return (
            <div className="animate-pulse space-y-4">
                {[1, 2, 3].map(i => (
                    <div key={i} className="h-16 bg-gray-100 rounded-lg" />
                ))}
            </div>
        )
    }

    // Error state
    if (error) {
        return (
            <div className="text-center py-8 text-gray-500">
                <BookOpenIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                <p>{error}</p>
            </div>
        )
    }

    // No hierarchy yet
    if (!hierarchy || hierarchy.chapters.length === 0) {
        return (
            <div className="text-center py-8 text-gray-500">
                <AcademicCapIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                <p className="font-medium">No syllabus topics yet</p>
                <p className="text-sm mt-1">Upload a syllabus to see topics here</p>
            </div>
        )
    }

    return (
        <div className="space-y-3">
            {/* Summary header */}
            <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
                <span>{hierarchy.total_chapters} chapters • {hierarchy.total_topics} topics</span>
                {showMastery && (
                    <span className="text-xs">
                        Click topics to practice
                    </span>
                )}
            </div>

            {/* Chapters */}
            {hierarchy.chapters.map((chapter) => {
                const isExpanded = expandedChapters.has(chapter.id)
                const chapterMastery = chapter.topics.length > 0
                    ? chapter.topics.reduce((sum, t) => sum + (t.mastery_percentage || 0), 0) / chapter.topics.length
                    : 0

                return (
                    <div
                        key={chapter.id}
                        className="rounded-lg border overflow-hidden"
                        style={{ borderColor: chapter.color + '40' }}
                    >
                        {/* Chapter Header */}
                        <button
                            onClick={() => toggleChapter(chapter.id)}
                            className="w-full flex items-center gap-3 p-4 hover:bg-gray-50 transition-colors"
                            style={{ backgroundColor: chapter.color + '10' }}
                        >
                            {/* Color indicator */}
                            <div
                                className="w-1 h-12 rounded-full"
                                style={{ backgroundColor: chapter.color }}
                            />

                            {/* Chapter info */}
                            <div className="flex-1 text-left">
                                <div className="flex items-center gap-2">
                                    <h3 className="font-medium text-gray-900">{chapter.name}</h3>
                                    <span className="text-xs text-gray-400">
                                        {chapter.topic_count} topics
                                    </span>
                                </div>
                                {chapter.description && (
                                    <p className="text-sm text-gray-500 mt-0.5 line-clamp-1">
                                        {chapter.description}
                                    </p>
                                )}
                            </div>

                            {/* Chapter mastery (average of topics) */}
                            {showMastery && (
                                <MasteryBadge percentage={chapterMastery} />
                            )}

                            {/* Expand/collapse icon */}
                            {isExpanded ? (
                                <ChevronDownIcon className="w-5 h-5 text-gray-400" />
                            ) : (
                                <ChevronRightIcon className="w-5 h-5 text-gray-400" />
                            )}
                        </button>

                        {/* Topics (expanded) */}
                        {isExpanded && (
                            <div className="border-t divide-y" style={{ borderColor: chapter.color + '20' }}>
                                {chapter.topics.map((topic) => (
                                    <div
                                        key={topic.id}
                                        onClick={() => onTopicClick?.(topic)}
                                        className={`flex items-center gap-3 p-3 pl-8 ${onTopicClick ? 'cursor-pointer hover:bg-gray-50' : ''}`}
                                    >
                                        {/* Topic color bar */}
                                        <div
                                            className="w-0.5 h-8 rounded-full opacity-50"
                                            style={{ backgroundColor: chapter.color }}
                                        />

                                        {/* Topic info */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium text-gray-800 truncate">
                                                    {topic.name}
                                                </span>
                                                <DifficultyBadge difficulty={topic.difficulty} />
                                            </div>
                                            {topic.description && (
                                                <p className="text-xs text-gray-500 truncate mt-0.5">
                                                    {topic.description}
                                                </p>
                                            )}
                                            {topic.key_concepts?.length > 0 && (
                                                <div className="flex flex-wrap gap-1 mt-1">
                                                    {topic.key_concepts.slice(0, 3).map((concept, i) => (
                                                        <span
                                                            key={i}
                                                            className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded"
                                                        >
                                                            {concept}
                                                        </span>
                                                    ))}
                                                    {topic.key_concepts.length > 3 && (
                                                        <span className="text-xs text-gray-400">
                                                            +{topic.key_concepts.length - 3} more
                                                        </span>
                                                    )}
                                                </div>
                                            )}
                                        </div>

                                        {/* Question count */}
                                        {topic.question_count > 0 && (
                                            <span className="text-xs text-gray-400">
                                                {topic.question_count} Q
                                            </span>
                                        )}

                                        {/* Topic mastery */}
                                        {showMastery && (
                                            <MasteryBadge
                                                percentage={topic.mastery_percentage || 0}
                                                status={topic.status}
                                            />
                                        )}
                                    </div>
                                ))}

                                {chapter.topics.length === 0 && (
                                    <div className="p-4 pl-8 text-sm text-gray-400 italic">
                                        No topics in this chapter yet
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )
            })}
        </div>
    )
}
