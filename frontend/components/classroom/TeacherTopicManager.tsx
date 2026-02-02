'use client'

import { useState, useEffect } from 'react'
import { getAiServiceUrl, getApiBaseUrl } from '@/utils/api'
import {
    AcademicCapIcon,
    BookOpenIcon,
    ChevronDownIcon,
    ChevronRightIcon,
    PencilIcon,
    TrashIcon,
    PlusIcon,
    XMarkIcon,
    CheckIcon,
    ArrowPathIcon,
    SparklesIcon,
    ClockIcon
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
    onClose?: () => void
    onUpdate?: () => void
}

// ============================================================================
// Constants
// ============================================================================

const CHAPTER_COLORS = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
    '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#14B8A6'
]

const DIFFICULTIES = [
    { value: 'easy', label: 'Easy', color: 'bg-green-100 text-green-700' },
    { value: 'medium', label: 'Medium', color: 'bg-yellow-100 text-yellow-700' },
    { value: 'hard', label: 'Hard', color: 'bg-red-100 text-red-700' }
]

// ============================================================================
// Main Component
// ============================================================================

export default function TeacherTopicManager({ classroomId, onClose, onUpdate }: Props) {
    // State
    const [chapters, setChapters] = useState<Chapter[]>([])
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState('')
    const [expandedChapters, setExpandedChapters] = useState<Set<string>>(new Set())

    // Edit states
    const [editingChapterId, setEditingChapterId] = useState<string | null>(null)
    const [editingTopicId, setEditingTopicId] = useState<string | null>(null)
    const [editChapterName, setEditChapterName] = useState('')
    const [editTopicData, setEditTopicData] = useState<Partial<Topic>>({})

    // New item states
    const [showNewChapterForm, setShowNewChapterForm] = useState(false)
    const [newChapterName, setNewChapterName] = useState('')
    const [addingTopicToChapter, setAddingTopicToChapter] = useState<string | null>(null)
    const [newTopicName, setNewTopicName] = useState('')

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
            const res = await fetch(`${getAiServiceUrl()}/api/classroom-syllabus/hierarchy/${classroomId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                const data = await res.json()
                setChapters(data.chapters || [])
                // Auto-expand first chapter
                if (data.chapters?.length > 0) {
                    setExpandedChapters(new Set([data.chapters[0].id]))
                }
            } else if (res.status === 404) {
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
    // Chapter Operations
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

    const addChapter = async () => {
        if (!newChapterName.trim()) return

        setSaving(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/chapters`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({
                    name: newChapterName,
                    color: CHAPTER_COLORS[chapters.length % CHAPTER_COLORS.length],
                    order: chapters.length
                })
            })

            if (res.ok) {
                const data = await res.json()
                setChapters([...chapters, { ...data.chapter, topics: [] }])
                setNewChapterName('')
                setShowNewChapterForm(false)
                onUpdate?.()
            } else {
                setError('Failed to add chapter')
            }
        } catch (e) {
            setError('Failed to add chapter')
        } finally {
            setSaving(false)
        }
    }

    const updateChapter = async (chapterId: string) => {
        if (!editChapterName.trim()) return

        setSaving(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/chapters/${chapterId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({ name: editChapterName })
            })

            if (res.ok) {
                setChapters(chapters.map(ch =>
                    ch.id === chapterId ? { ...ch, name: editChapterName } : ch
                ))
                setEditingChapterId(null)
                onUpdate?.()
            } else {
                setError('Failed to update chapter')
            }
        } catch (e) {
            setError('Failed to update chapter')
        } finally {
            setSaving(false)
        }
    }

    const deleteChapter = async (chapterId: string) => {
        if (!confirm('Delete this chapter and all its topics?')) return

        setSaving(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/chapters/${chapterId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                setChapters(chapters.filter(ch => ch.id !== chapterId))
                onUpdate?.()
            } else {
                setError('Failed to delete chapter')
            }
        } catch (e) {
            setError('Failed to delete chapter')
        } finally {
            setSaving(false)
        }
    }

    // ========================================================================
    // Topic Operations
    // ========================================================================

    const addTopic = async (chapterId: string) => {
        if (!newTopicName.trim()) return

        setSaving(true)
        try {
            const chapter = chapters.find(ch => ch.id === chapterId)
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/topics`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({
                    chapter_id: chapterId,
                    name: newTopicName,
                    difficulty: 'medium',
                    estimated_hours: 2,
                    order: chapter?.topics.length || 0
                })
            })

            if (res.ok) {
                const data = await res.json()
                setChapters(chapters.map(ch =>
                    ch.id === chapterId
                        ? { ...ch, topics: [...ch.topics, data.topic] }
                        : ch
                ))
                setNewTopicName('')
                setAddingTopicToChapter(null)
                onUpdate?.()
            } else {
                setError('Failed to add topic')
            }
        } catch (e) {
            setError('Failed to add topic')
        } finally {
            setSaving(false)
        }
    }

    const updateTopic = async (chapterId: string, topicId: string) => {
        if (!editTopicData.name?.trim()) return

        setSaving(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/topics/${topicId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify(editTopicData)
            })

            if (res.ok) {
                const data = await res.json()
                setChapters(chapters.map(ch =>
                    ch.id === chapterId
                        ? { ...ch, topics: ch.topics.map(t => t.id === topicId ? data.topic : t) }
                        : ch
                ))
                setEditingTopicId(null)
                setEditTopicData({})
                onUpdate?.()
            } else {
                setError('Failed to update topic')
            }
        } catch (e) {
            setError('Failed to update topic')
        } finally {
            setSaving(false)
        }
    }

    const deleteTopic = async (chapterId: string, topicId: string) => {
        if (!confirm('Delete this topic?')) return

        setSaving(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/topics/${topicId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                setChapters(chapters.map(ch =>
                    ch.id === chapterId
                        ? { ...ch, topics: ch.topics.filter(t => t.id !== topicId) }
                        : ch
                ))
                onUpdate?.()
            } else {
                setError('Failed to delete topic')
            }
        } catch (e) {
            setError('Failed to delete topic')
        } finally {
            setSaving(false)
        }
    }

    const startEditTopic = (topic: Topic) => {
        setEditingTopicId(topic.id)
        setEditTopicData({
            name: topic.name,
            description: topic.description,
            difficulty: topic.difficulty,
            estimated_hours: topic.estimated_hours
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
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-indigo-100">
                        <AcademicCapIcon className="w-5 h-5 text-indigo-600" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-900">Topic Management</h3>
                        <p className="text-sm text-gray-500">
                            {chapters.length} chapters • {totalTopics} topics • {totalHours.toFixed(1)}h total
                        </p>
                    </div>
                </div>
                {onClose && (
                    <button onClick={onClose} className="p-2 text-gray-400 hover:text-gray-600">
                        <XMarkIcon className="w-5 h-5" />
                    </button>
                )}
            </div>

            {error && (
                <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>
            )}

            {/* Chapters List */}
            <div className="space-y-3">
                {chapters.map((chapter) => {
                    const isExpanded = expandedChapters.has(chapter.id)
                    const isEditing = editingChapterId === chapter.id

                    return (
                        <div
                            key={chapter.id}
                            className="border rounded-lg overflow-hidden"
                            style={{ borderColor: chapter.color + '40' }}
                        >
                            {/* Chapter Header */}
                            <div
                                className="flex items-center gap-3 p-3"
                                style={{ backgroundColor: chapter.color + '08' }}
                            >
                                <div
                                    className="w-1.5 h-12 rounded-full flex-shrink-0"
                                    style={{ backgroundColor: chapter.color }}
                                />

                                <button
                                    onClick={() => toggleChapter(chapter.id)}
                                    className="p-1 hover:bg-white/50 rounded"
                                >
                                    {isExpanded ? (
                                        <ChevronDownIcon className="w-4 h-4 text-gray-500" />
                                    ) : (
                                        <ChevronRightIcon className="w-4 h-4 text-gray-500" />
                                    )}
                                </button>

                                {isEditing ? (
                                    <div className="flex-1 flex items-center gap-2">
                                        <input
                                            type="text"
                                            value={editChapterName}
                                            onChange={(e) => setEditChapterName(e.target.value)}
                                            className="flex-1 px-2 py-1 border rounded text-sm"
                                            autoFocus
                                        />
                                        <button
                                            onClick={() => updateChapter(chapter.id)}
                                            disabled={saving}
                                            className="p-1 text-green-600 hover:bg-green-50 rounded"
                                        >
                                            <CheckIcon className="w-4 h-4" />
                                        </button>
                                        <button
                                            onClick={() => setEditingChapterId(null)}
                                            className="p-1 text-gray-400 hover:bg-gray-100 rounded"
                                        >
                                            <XMarkIcon className="w-4 h-4" />
                                        </button>
                                    </div>
                                ) : (
                                    <>
                                        <div className="flex-1">
                                            <p className="font-medium text-gray-900">{chapter.name}</p>
                                            <p className="text-xs text-gray-500">
                                                {chapter.topics.length} topics
                                            </p>
                                        </div>
                                        <button
                                            onClick={() => {
                                                setEditingChapterId(chapter.id)
                                                setEditChapterName(chapter.name)
                                            }}
                                            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-white/50 rounded"
                                        >
                                            <PencilIcon className="w-4 h-4" />
                                        </button>
                                        <button
                                            onClick={() => deleteChapter(chapter.id)}
                                            className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-white/50 rounded"
                                        >
                                            <TrashIcon className="w-4 h-4" />
                                        </button>
                                    </>
                                )}
                            </div>

                            {/* Topics */}
                            {isExpanded && (
                                <div className="border-t divide-y" style={{ borderColor: chapter.color + '20' }}>
                                    {chapter.topics.map((topic) => {
                                        const isEditingTopic = editingTopicId === topic.id

                                        return (
                                            <div key={topic.id} className="p-3 pl-12 bg-white group">
                                                {isEditingTopic ? (
                                                    <div className="space-y-2">
                                                        <input
                                                            type="text"
                                                            value={editTopicData.name || ''}
                                                            onChange={(e) => setEditTopicData({ ...editTopicData, name: e.target.value })}
                                                            className="w-full px-2 py-1 border rounded text-sm"
                                                            placeholder="Topic name"
                                                            autoFocus
                                                        />
                                                        <div className="flex gap-2">
                                                            <select
                                                                value={editTopicData.difficulty || 'medium'}
                                                                onChange={(e) => setEditTopicData({ ...editTopicData, difficulty: e.target.value as any })}
                                                                className="text-xs px-2 py-1 border rounded"
                                                            >
                                                                {DIFFICULTIES.map(d => (
                                                                    <option key={d.value} value={d.value}>{d.label}</option>
                                                                ))}
                                                            </select>
                                                            <input
                                                                type="number"
                                                                value={editTopicData.estimated_hours || 2}
                                                                onChange={(e) => setEditTopicData({ ...editTopicData, estimated_hours: parseFloat(e.target.value) })}
                                                                className="w-16 text-xs px-2 py-1 border rounded"
                                                                step="0.5"
                                                                min="0.5"
                                                            />
                                                            <span className="text-xs text-gray-500 self-center">hours</span>
                                                        </div>
                                                        <div className="flex gap-2">
                                                            <button
                                                                onClick={() => updateTopic(chapter.id, topic.id)}
                                                                disabled={saving}
                                                                className="px-3 py-1 bg-indigo-600 text-white text-xs rounded hover:bg-indigo-700"
                                                            >
                                                                Save
                                                            </button>
                                                            <button
                                                                onClick={() => {
                                                                    setEditingTopicId(null)
                                                                    setEditTopicData({})
                                                                }}
                                                                className="px-3 py-1 border text-gray-600 text-xs rounded hover:bg-gray-50"
                                                            >
                                                                Cancel
                                                            </button>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <div className="flex items-center gap-3">
                                                        <BookOpenIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
                                                        <div className="flex-1 min-w-0">
                                                            <p className="text-sm text-gray-800 truncate">{topic.name}</p>
                                                        </div>
                                                        <span className={`text-xs px-1.5 py-0.5 rounded ${DIFFICULTIES.find(d => d.value === topic.difficulty)?.color || 'bg-gray-100'
                                                            }`}>
                                                            {topic.difficulty}
                                                        </span>
                                                        <span className="text-xs text-gray-500 flex items-center gap-1">
                                                            <ClockIcon className="w-3 h-3" />
                                                            {topic.estimated_hours}h
                                                        </span>
                                                        <button
                                                            onClick={() => startEditTopic(topic)}
                                                            className="p-1 text-gray-400 hover:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity"
                                                        >
                                                            <PencilIcon className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => deleteTopic(chapter.id, topic.id)}
                                                            className="p-1 text-gray-400 hover:text-red-600 opacity-0 group-hover:opacity-100 transition-opacity"
                                                        >
                                                            <TrashIcon className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                        )
                                    })}

                                    {/* Add Topic Form */}
                                    {addingTopicToChapter === chapter.id ? (
                                        <div className="p-3 pl-12 bg-gray-50 flex items-center gap-2">
                                            <input
                                                type="text"
                                                value={newTopicName}
                                                onChange={(e) => setNewTopicName(e.target.value)}
                                                placeholder="New topic name"
                                                className="flex-1 px-2 py-1 border rounded text-sm"
                                                autoFocus
                                                onKeyDown={(e) => e.key === 'Enter' && addTopic(chapter.id)}
                                            />
                                            <button
                                                onClick={() => addTopic(chapter.id)}
                                                disabled={saving || !newTopicName.trim()}
                                                className="px-3 py-1 bg-indigo-600 text-white text-xs rounded hover:bg-indigo-700 disabled:opacity-50"
                                            >
                                                Add
                                            </button>
                                            <button
                                                onClick={() => {
                                                    setAddingTopicToChapter(null)
                                                    setNewTopicName('')
                                                }}
                                                className="p-1 text-gray-400 hover:text-gray-600"
                                            >
                                                <XMarkIcon className="w-4 h-4" />
                                            </button>
                                        </div>
                                    ) : (
                                        <button
                                            onClick={() => setAddingTopicToChapter(chapter.id)}
                                            className="w-full p-2 pl-12 text-sm text-indigo-600 hover:bg-indigo-50 flex items-center gap-2"
                                        >
                                            <PlusIcon className="w-4 h-4" />
                                            Add Topic
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    )
                })}

                {/* Add Chapter */}
                {showNewChapterForm ? (
                    <div className="border rounded-lg p-4 bg-gray-50">
                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                value={newChapterName}
                                onChange={(e) => setNewChapterName(e.target.value)}
                                placeholder="Chapter name"
                                className="flex-1 px-3 py-2 border rounded-lg"
                                autoFocus
                                onKeyDown={(e) => e.key === 'Enter' && addChapter()}
                            />
                            <button
                                onClick={addChapter}
                                disabled={saving || !newChapterName.trim()}
                                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                            >
                                Add
                            </button>
                            <button
                                onClick={() => {
                                    setShowNewChapterForm(false)
                                    setNewChapterName('')
                                }}
                                className="p-2 text-gray-400 hover:text-gray-600"
                            >
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </div>
                    </div>
                ) : (
                    <button
                        onClick={() => setShowNewChapterForm(true)}
                        className="w-full p-4 border-2 border-dashed rounded-lg text-gray-500 hover:text-indigo-600 hover:border-indigo-300 flex items-center justify-center gap-2"
                    >
                        <PlusIcon className="w-5 h-5" />
                        Add Chapter
                    </button>
                )}
            </div>

            {/* Empty State */}
            {chapters.length === 0 && !showNewChapterForm && (
                <div className="text-center py-8 text-gray-500">
                    <SparklesIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                    <p className="font-medium">No topics yet</p>
                    <p className="text-sm">Upload a syllabus to extract topics or add them manually</p>
                </div>
            )}
        </div>
    )
}
