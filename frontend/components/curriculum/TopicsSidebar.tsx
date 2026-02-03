'use client'

import { useState, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import {
    MagnifyingGlassIcon,
    BookOpenIcon,
    ChevronDownIcon,
    ChevronRightIcon,
    ClockIcon,
    AcademicCapIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface Topic {
    id: string
    name: string
    description?: string
    key_concepts?: string[]
    difficulty: string
    estimated_hours: number
    order: number
    mastery_level?: number  // From StudentTopicScore
}

interface Chapter {
    id: string
    name: string
    color: string
    order: number
    topics: Topic[]
}

interface ClassroomData {
    classroom_id: string
    classroom_name: string
    subject: string
    chapters: Chapter[]
}

interface TopicsSidebarProps {
    onDragStart?: (topic: Topic, subjectName: string, chapterName: string) => void
    selectedClassroomIds?: string[] // Empty array = All
}

// ============================================================================
// Helper Components
// ============================================================================

function DifficultyBadge({ difficulty }: { difficulty: string }) {
    const colors: Record<string, string> = {
        easy: 'bg-green-100 text-green-700',
        medium: 'bg-yellow-100 text-yellow-700',
        hard: 'bg-red-100 text-red-700'
    }
    return (
        <span className={`text-xs px-1.5 py-0.5 rounded-full ${colors[difficulty] || 'bg-gray-100'}`}>
            {difficulty}
        </span>
    )
}

// ============================================================================
// Main Component
// ============================================================================

export default function TopicsSidebar({ onDragStart, selectedClassroomIds = [] }: TopicsSidebarProps) {
    const [classrooms, setClassrooms] = useState<ClassroomData[]>([])
    const [loading, setLoading] = useState(true)
    const [searchQuery, setSearchQuery] = useState('')
    const [expandedClassrooms, setExpandedClassrooms] = useState<Set<string>>(new Set())
    const [expandedChapters, setExpandedChapters] = useState<Set<string>>(new Set())

    // Filter classrooms based on selection
    const filteredClassrooms = selectedClassroomIds.length === 0
        ? classrooms
        : classrooms.filter(c => selectedClassroomIds.includes(c.classroom_id))

    // Fetch enrolled topics
    useEffect(() => {
        fetchTopics()
    }, [])

    const fetchTopics = async () => {
        setLoading(true)
        try {
            // Fetch enrolled topics and mastery data in parallel
            const [topicsRes, masteryRes] = await Promise.all([
                fetch(`${getApiBaseUrl()}/api/curriculum/enrolled-topics`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                }),
                fetch(`${getApiBaseUrl()}/api/progress/topic-mastery`, {
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

            if (topicsRes.ok) {
                const data = await topicsRes.json()
                // Merge mastery data into topics
                const classroomsWithMastery = (data.classrooms || []).map((c: ClassroomData) => ({
                    ...c,
                    chapters: c.chapters.map(ch => ({
                        ...ch,
                        topics: ch.topics.map(t => ({
                            ...t,
                            mastery_level: masteryMap[t.id] ?? 0
                        }))
                    }))
                }))
                setClassrooms(classroomsWithMastery)
                // Auto-expand all classrooms and first chapter of each
                const classroomIds = new Set(classroomsWithMastery.map((c: ClassroomData) => c.classroom_id))
                setExpandedClassrooms(classroomIds)
                const firstChapterIds = new Set(
                    classroomsWithMastery.map((c: ClassroomData) => c.chapters[0]?.id).filter(Boolean)
                )
                setExpandedChapters(firstChapterIds)
            }
        } catch (error) {
            console.error('Failed to fetch enrolled topics:', error)
        } finally {
            setLoading(false)
        }
    }

    const toggleClassroom = (classroomId: string) => {
        setExpandedClassrooms(prev => {
            const newSet = new Set(prev)
            if (newSet.has(classroomId)) {
                newSet.delete(classroomId)
            } else {
                newSet.add(classroomId)
            }
            return newSet
        })
    }

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

    // Filter topics by search query
    const filterTopics = (topics: Topic[]) => {
        if (!searchQuery) return topics
        return topics.filter(t =>
            t.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
    }

    const handleDragStart = (
        e: React.DragEvent,
        topic: Topic,
        subjectName: string,
        chapterName: string
    ) => {
        e.dataTransfer.setData('application/json', JSON.stringify({
            type: 'new_topic',
            topic,
            subjectName,
            chapterName
        }))
        e.dataTransfer.effectAllowed = 'move'
        onDragStart?.(topic, subjectName, chapterName)
    }

    // Total stats (filtered)
    const totalTopics = filteredClassrooms.reduce((sum, c) =>
        sum + c.chapters.reduce((cSum, ch) => cSum + ch.topics.length, 0), 0
    )

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            </div>
        )
    }

    return (
        <div className="h-full flex flex-col bg-white rounded-xl border shadow-sm">
            {/* Header */}
            <div className="p-4 border-b">
                <div className="flex items-center gap-2 mb-3">
                    <AcademicCapIcon className="w-5 h-5 text-indigo-600" />
                    <h2 className="font-semibold text-gray-900">Topics</h2>
                    <span className="ml-auto text-xs text-gray-500">{totalTopics} topics</span>
                </div>

                {/* Search */}
                <div className="relative">
                    <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search topics..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-9 pr-4 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    />
                </div>
            </div>

            {/* Topics Tree */}
            <div className="flex-1 overflow-y-auto p-2">
                {filteredClassrooms.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                        <BookOpenIcon className="w-10 h-10 mx-auto mb-2 text-gray-300" />
                        <p className="text-sm">{classrooms.length === 0 ? 'No enrolled classrooms' : 'No matching classrooms'}</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {filteredClassrooms.map(classroom => {
                            const isClassroomExpanded = expandedClassrooms.has(classroom.classroom_id)

                            return (
                                <div key={classroom.classroom_id} className="border rounded-lg overflow-hidden">
                                    {/* Classroom Header */}
                                    <button
                                        onClick={() => toggleClassroom(classroom.classroom_id)}
                                        className="w-full flex items-center gap-2 p-3 bg-indigo-50 hover:bg-indigo-100 transition-colors text-left"
                                    >
                                        {isClassroomExpanded ? (
                                            <ChevronDownIcon className="w-4 h-4 text-indigo-600 flex-shrink-0" />
                                        ) : (
                                            <ChevronRightIcon className="w-4 h-4 text-indigo-600 flex-shrink-0" />
                                        )}
                                        <span className="font-medium text-indigo-900 truncate">
                                            {classroom.classroom_name}
                                        </span>
                                        <span className="ml-auto text-xs text-indigo-600">
                                            {classroom.chapters.reduce((sum, ch) => sum + ch.topics.length, 0)}
                                        </span>
                                    </button>

                                    {/* Chapters */}
                                    {isClassroomExpanded && (
                                        <div className="border-t">
                                            {classroom.chapters.map(chapter => {
                                                const isChapterExpanded = expandedChapters.has(chapter.id)
                                                const filteredTopics = filterTopics(chapter.topics)

                                                // Skip chapters with no matching topics when searching
                                                if (searchQuery && filteredTopics.length === 0) return null

                                                return (
                                                    <div key={chapter.id}>
                                                        {/* Chapter Header */}
                                                        <button
                                                            onClick={() => toggleChapter(chapter.id)}
                                                            className="w-full flex items-center gap-2 p-2 pl-6 hover:bg-gray-50 transition-colors text-left"
                                                        >
                                                            <div
                                                                className="w-2 h-2 rounded-full flex-shrink-0"
                                                                style={{ backgroundColor: chapter.color || '#6366F1' }}
                                                            />
                                                            {isChapterExpanded ? (
                                                                <ChevronDownIcon className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
                                                            ) : (
                                                                <ChevronRightIcon className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
                                                            )}
                                                            <span className="text-sm text-gray-700 truncate">
                                                                {chapter.name}
                                                            </span>
                                                            <span className="ml-auto text-xs text-gray-400">
                                                                {filteredTopics.length}
                                                            </span>
                                                        </button>

                                                        {/* Topics */}
                                                        {isChapterExpanded && filteredTopics.length > 0 && (
                                                            <div className="border-t border-gray-100">
                                                                {filteredTopics.map(topic => (
                                                                    <div
                                                                        key={topic.id}
                                                                        draggable
                                                                        onDragStart={(e) => handleDragStart(
                                                                            e,
                                                                            topic,
                                                                            classroom.classroom_name,
                                                                            chapter.name
                                                                        )}
                                                                        className="p-3 pl-12 hover:bg-indigo-50 cursor-grab active:cursor-grabbing transition-colors group border-b border-gray-50 last:border-b-0"
                                                                    >
                                                                        <div className="flex items-center gap-2">
                                                                            <BookOpenIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
                                                                            <span className="text-sm font-medium text-gray-800 flex-1 truncate">
                                                                                {topic.name}
                                                                            </span>
                                                                            <DifficultyBadge difficulty={topic.difficulty} />
                                                                            {/* Mastery Badge */}
                                                                            <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${topic.mastery_level === undefined || topic.mastery_level === 0
                                                                                    ? 'bg-gray-100 text-gray-500'
                                                                                    : topic.mastery_level >= 70
                                                                                        ? 'bg-green-100 text-green-700'
                                                                                        : topic.mastery_level >= 50
                                                                                            ? 'bg-yellow-100 text-yellow-700'
                                                                                            : 'bg-red-100 text-red-700'
                                                                                }`}>
                                                                                {topic.mastery_level === undefined || topic.mastery_level === 0
                                                                                    ? 'New'
                                                                                    : `${Math.round(topic.mastery_level)}%`}
                                                                            </span>
                                                                            <span className="text-xs text-gray-400 flex items-center gap-0.5 whitespace-nowrap">
                                                                                <ClockIcon className="w-3 h-3" />
                                                                                {topic.estimated_hours}h
                                                                            </span>
                                                                        </div>
                                                                        {/* Description / Subtopics */}
                                                                        {topic.description && (
                                                                            <p className="mt-1 ml-6 text-xs text-gray-500 line-clamp-2">
                                                                                {topic.description}
                                                                            </p>
                                                                        )}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                )
                                            })}
                                        </div>
                                    )}
                                </div>
                            )
                        })}
                    </div>
                )}
            </div>

            {/* Help Text */}
            <div className="p-3 border-t bg-gray-50 text-xs text-gray-500 text-center">
                Drag topics to the calendar to schedule
            </div>
        </div>
    )
}
