'use client'

import { useState, useEffect } from 'react'
import { XMarkIcon, SparklesIcon, ExclamationTriangleIcon, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Classroom {
    id: string
    name: string
    subject_name: string
}

interface Topic {
    id: string
    name: string
    chapter_id: string
    difficulty: string
    key_concepts: string[]
    mastery_percentage?: number
    status?: string
}

interface Chapter {
    id: string
    name: string
    color: string
    topic_count: number
    topics: Topic[]
}

interface WeakTopic {
    topic_id: string
    topic_name: string
    mastery_percentage: number
    chapter_name: string
}

interface CreateAssessmentModalProps {
    isOpen: boolean
    onClose: () => void
    onCreated: (assessment: any) => void
    classrooms: Classroom[]
}

export default function CreateAssessmentModal({
    isOpen,
    onClose,
    onCreated,
    classrooms
}: CreateAssessmentModalProps) {
    // Form state
    const [title, setTitle] = useState('')
    const [selectedClassroom, setSelectedClassroom] = useState<string>('')
    const [chapters, setChapters] = useState<Chapter[]>([])
    const [selectedChapters, setSelectedChapters] = useState<string[]>([])
    const [selectedTopics, setSelectedTopics] = useState<string[]>([])
    const [expandedChapters, setExpandedChapters] = useState<string[]>([])
    const [weakTopics, setWeakTopics] = useState<WeakTopic[]>([])
    const [includeWeakTopics, setIncludeWeakTopics] = useState(false)
    const [useAIQuestions, setUseAIQuestions] = useState(true)
    const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard' | 'mixed'>('medium')
    const [questionCount, setQuestionCount] = useState(10)
    const [timeLimit, setTimeLimit] = useState(30)

    const [isLoading, setIsLoading] = useState(false)
    const [isLoadingHierarchy, setIsLoadingHierarchy] = useState(false)
    const [noSyllabusFound, setNoSyllabusFound] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Fetch syllabus hierarchy when classroom changes
    useEffect(() => {
        if (!selectedClassroom) {
            setChapters([])
            setSelectedChapters([])
            setSelectedTopics([])
            setWeakTopics([])
            setNoSyllabusFound(false)
            return
        }

        const fetchHierarchy = async () => {
            setIsLoadingHierarchy(true)
            setNoSyllabusFound(false)
            try {
                // Fetch syllabus-extracted hierarchy from AI service
                const res = await fetch(`/api/ai/classroom-syllabus/hierarchy/${selectedClassroom}`)
                if (res.ok) {
                    const data = await res.json()
                    const fetchedChapters = data.chapters || []
                    setChapters(fetchedChapters)

                    if (fetchedChapters.length === 0) {
                        setNoSyllabusFound(true)
                    }
                } else {
                    // Fallback to core service chapters endpoint
                    const fallbackRes = await fetch(`/api/classrooms/${selectedClassroom}/chapters`)
                    if (fallbackRes.ok) {
                        const fallbackData = await fallbackRes.json()
                        setChapters(fallbackData.chapters || [])
                        if ((fallbackData.chapters || []).length === 0) {
                            setNoSyllabusFound(true)
                        }
                    }
                }
            } catch (err) {
                console.error('Failed to fetch hierarchy:', err)
                setNoSyllabusFound(true)
            }
            setIsLoadingHierarchy(false)
        }

        const fetchWeakTopics = async () => {
            try {
                const res = await fetch(`/api/assessments/weak-topics?classroom_id=${selectedClassroom}`)
                if (res.ok) {
                    const data = await res.json()
                    setWeakTopics(data.weak_topics || [])
                }
            } catch (err) {
                console.error('Failed to fetch weak topics:', err)
            }
        }

        fetchHierarchy()
        fetchWeakTopics()
    }, [selectedClassroom])

    // Auto-select weak topics when checkbox is checked
    useEffect(() => {
        if (includeWeakTopics && weakTopics.length > 0) {
            const weakTopicIds = weakTopics.map(w => w.topic_id)
            setSelectedTopics(prev => {
                const combined = new Set([...prev, ...weakTopicIds])
                return Array.from(combined)
            })
        }
    }, [includeWeakTopics, weakTopics])

    const toggleChapterExpand = (chapterId: string) => {
        setExpandedChapters(prev =>
            prev.includes(chapterId)
                ? prev.filter(id => id !== chapterId)
                : [...prev, chapterId]
        )
    }

    const handleChapterToggle = (chapterId: string) => {
        const chapter = chapters.find(c => c.id === chapterId)
        if (!chapter) return

        const isSelected = selectedChapters.includes(chapterId)

        if (isSelected) {
            // Deselect chapter and all its topics
            setSelectedChapters(prev => prev.filter(id => id !== chapterId))
            const chapterTopicIds = chapter.topics.map(t => t.id)
            setSelectedTopics(prev => prev.filter(id => !chapterTopicIds.includes(id)))
        } else {
            // Select chapter and all its topics
            setSelectedChapters(prev => [...prev, chapterId])
            const chapterTopicIds = chapter.topics.map(t => t.id)
            setSelectedTopics(prev => [...new Set([...prev, ...chapterTopicIds])])
            // Auto-expand when selected
            if (!expandedChapters.includes(chapterId)) {
                setExpandedChapters(prev => [...prev, chapterId])
            }
        }
    }

    const handleTopicToggle = (topicId: string, chapterId: string) => {
        setSelectedTopics(prev =>
            prev.includes(topicId)
                ? prev.filter(id => id !== topicId)
                : [...prev, topicId]
        )
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)
        setIsLoading(true)

        try {
            const classroom = classrooms.find(c => c.id === selectedClassroom)

            if (useAIQuestions) {
                // Generate questions using AI from syllabus content
                const aiRes = await fetch('/api/ai/questions/generate-assessment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        classroom_id: selectedClassroom,
                        chapter_ids: selectedChapters,
                        topic_ids: selectedTopics,
                        include_weak_topics: includeWeakTopics,
                        num_questions: questionCount,
                        difficulty: difficulty,
                        title: title || undefined,
                        time_limit_minutes: timeLimit
                    })
                })

                if (!aiRes.ok) {
                    throw new Error('Failed to generate questions')
                }

                const aiData = await aiRes.json()

                if (!aiData.success) {
                    throw new Error(aiData.error || 'Failed to generate questions')
                }

                // Now create the assessment with generated questions
                const createRes = await fetch('/api/assessments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: aiData.title || title,
                        topic: classroom?.subject_name || 'General',
                        subject: classroom?.subject_name || 'General',
                        questions: aiData.questions,
                        difficulty: difficulty,
                        time_limit_minutes: timeLimit,
                        assessment_type: 'self_practice',
                        classroom_id: selectedClassroom,
                        use_ai_questions: true,
                        source_topics: selectedTopics,
                        source_chapters: selectedChapters,
                        include_weak_topics: includeWeakTopics
                    })
                })

                if (!createRes.ok) {
                    throw new Error('Failed to create assessment')
                }

                const assessment = await createRes.json()
                onCreated(assessment.assessment)
            } else {
                // Create assessment without AI
                const createRes = await fetch('/api/assessments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: title || `${classroom?.subject_name || 'General'} Assessment`,
                        topic: classroom?.subject_name || 'General',
                        subject: classroom?.subject_name || 'General',
                        questions: [],
                        difficulty: difficulty,
                        time_limit_minutes: timeLimit,
                        assessment_type: 'self_practice',
                        classroom_id: selectedClassroom,
                        use_ai_questions: false,
                        source_topics: selectedTopics,
                        source_chapters: selectedChapters,
                        include_weak_topics: includeWeakTopics
                    })
                })

                if (!createRes.ok) {
                    throw new Error('Failed to create assessment')
                }

                const assessment = await createRes.json()
                onCreated(assessment.assessment)
            }

            onClose()
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Something went wrong')
        } finally {
            setIsLoading(false)
        }
    }

    if (!isOpen) return null

    const selectedTopicCount = selectedTopics.length
    const totalTopics = chapters.reduce((sum, c) => sum + (c.topics?.length || 0), 0)

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between bg-gradient-to-r from-primary-500 to-primary-600">
                    <h2 className="text-xl font-bold text-white">Create New Assessment</h2>
                    <button
                        onClick={onClose}
                        className="p-1 rounded-full hover:bg-white/20 transition-colors"
                    >
                        <XMarkIcon className="w-6 h-6 text-white" />
                    </button>
                </div>

                {/* Body */}
                <form onSubmit={handleSubmit} className="overflow-y-auto max-h-[calc(90vh-140px)]">
                    <div className="p-6 space-y-6">
                        {/* Title */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Assessment Title (Optional)
                            </label>
                            <input
                                type="text"
                                value={title}
                                onChange={(e) => setTitle(e.target.value)}
                                placeholder="e.g., Chapter 5 Review Quiz"
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                            />
                        </div>

                        {/* Classroom Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Subject / Classroom *
                            </label>
                            <select
                                value={selectedClassroom}
                                onChange={(e) => setSelectedClassroom(e.target.value)}
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                required
                            >
                                <option value="">Select a classroom...</option>
                                {classrooms.map(classroom => (
                                    <option key={classroom.id} value={classroom.id}>
                                        {classroom.name} ({classroom.subject_name})
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Chapters & Topics Hierarchy */}
                        {selectedClassroom && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Syllabus Topics
                                    {selectedTopicCount > 0 && (
                                        <span className="ml-2 text-primary-600">({selectedTopicCount} selected)</span>
                                    )}
                                </label>

                                {isLoadingHierarchy ? (
                                    <div className="p-4 text-center text-gray-500">
                                        <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                                        Loading syllabus topics...
                                    </div>
                                ) : noSyllabusFound ? (
                                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-3">
                                        <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                                        <div>
                                            <p className="text-sm text-yellow-800 font-medium">No syllabus topics found</p>
                                            <p className="text-xs text-yellow-700 mt-1">
                                                Please upload a syllabus PDF in the classroom settings to enable topic-based assessments.
                                            </p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="border border-gray-200 rounded-lg max-h-64 overflow-y-auto">
                                        {chapters.map(chapter => (
                                            <div key={chapter.id} className="border-b border-gray-100 last:border-b-0">
                                                {/* Chapter Header */}
                                                <div
                                                    className="flex items-center gap-2 p-3 hover:bg-gray-50 cursor-pointer"
                                                    onClick={() => toggleChapterExpand(chapter.id)}
                                                >
                                                    <button
                                                        type="button"
                                                        className="p-0.5"
                                                    >
                                                        {expandedChapters.includes(chapter.id) ? (
                                                            <ChevronDownIcon className="w-4 h-4 text-gray-500" />
                                                        ) : (
                                                            <ChevronRightIcon className="w-4 h-4 text-gray-500" />
                                                        )}
                                                    </button>
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedChapters.includes(chapter.id)}
                                                        onChange={(e) => {
                                                            e.stopPropagation()
                                                            handleChapterToggle(chapter.id)
                                                        }}
                                                        className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                                                    />
                                                    <div
                                                        className="w-3 h-3 rounded-full"
                                                        style={{ backgroundColor: chapter.color || '#3B82F6' }}
                                                    />
                                                    <span className="font-medium text-gray-900 flex-1">
                                                        {chapter.name}
                                                    </span>
                                                    <span className="text-xs text-gray-500">
                                                        {chapter.topics?.length || 0} topics
                                                    </span>
                                                </div>

                                                {/* Topics */}
                                                {expandedChapters.includes(chapter.id) && chapter.topics?.length > 0 && (
                                                    <div className="bg-gray-50 px-4 py-2 space-y-1">
                                                        {chapter.topics.map(topic => {
                                                            const isWeak = weakTopics.some(w => w.topic_id === topic.id)
                                                            return (
                                                                <label
                                                                    key={topic.id}
                                                                    className={clsx(
                                                                        'flex items-center gap-2 p-2 rounded-md cursor-pointer transition-colors',
                                                                        selectedTopics.includes(topic.id)
                                                                            ? 'bg-primary-100'
                                                                            : 'hover:bg-gray-100'
                                                                    )}
                                                                >
                                                                    <input
                                                                        type="checkbox"
                                                                        checked={selectedTopics.includes(topic.id)}
                                                                        onChange={() => handleTopicToggle(topic.id, chapter.id)}
                                                                        className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                                                                    />
                                                                    <span className="text-sm text-gray-700 flex-1">
                                                                        {topic.name}
                                                                    </span>
                                                                    {isWeak && (
                                                                        <span className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded-full">
                                                                            Weak
                                                                        </span>
                                                                    )}
                                                                    <span className={clsx(
                                                                        'text-xs px-1.5 py-0.5 rounded-full',
                                                                        topic.difficulty === 'easy' ? 'bg-green-100 text-green-700' :
                                                                            topic.difficulty === 'hard' ? 'bg-red-100 text-red-700' :
                                                                                'bg-yellow-100 text-yellow-700'
                                                                    )}>
                                                                        {topic.difficulty}
                                                                    </span>
                                                                </label>
                                                            )
                                                        })}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Options */}
                        <div className="space-y-4 pt-4 border-t border-gray-200">
                            {weakTopics.length > 0 && (
                                <div className="flex items-center gap-3">
                                    <input
                                        type="checkbox"
                                        id="includeWeakTopics"
                                        checked={includeWeakTopics}
                                        onChange={(e) => setIncludeWeakTopics(e.target.checked)}
                                        className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                                    />
                                    <label htmlFor="includeWeakTopics" className="text-sm text-gray-700">
                                        Include my weak topics ({weakTopics.length} found)
                                    </label>
                                </div>
                            )}

                            <div className="flex items-center gap-3">
                                <input
                                    type="checkbox"
                                    id="useAIQuestions"
                                    checked={useAIQuestions}
                                    onChange={(e) => setUseAIQuestions(e.target.checked)}
                                    className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                                />
                                <label htmlFor="useAIQuestions" className="text-sm text-gray-700 flex items-center gap-1">
                                    <SparklesIcon className="w-4 h-4 text-yellow-500" />
                                    Generate questions with AI from syllabus content
                                </label>
                            </div>
                        </div>

                        {/* Difficulty */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Difficulty
                            </label>
                            <div className="flex gap-2">
                                {(['easy', 'medium', 'hard', 'mixed'] as const).map(level => (
                                    <button
                                        key={level}
                                        type="button"
                                        onClick={() => setDifficulty(level)}
                                        className={clsx(
                                            'px-4 py-2 rounded-lg text-sm font-medium capitalize transition-colors',
                                            difficulty === level
                                                ? level === 'easy' ? 'bg-green-500 text-white'
                                                    : level === 'medium' ? 'bg-yellow-500 text-white'
                                                        : level === 'hard' ? 'bg-red-500 text-white'
                                                            : 'bg-purple-500 text-white'
                                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                        )}
                                    >
                                        {level}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Question Count & Time */}
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Number of Questions
                                </label>
                                <select
                                    value={questionCount}
                                    onChange={(e) => setQuestionCount(Number(e.target.value))}
                                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                                >
                                    {[5, 10, 15, 20, 25, 30].map(n => (
                                        <option key={n} value={n}>{n} questions</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Time Limit (minutes)
                                </label>
                                <input
                                    type="number"
                                    min={5}
                                    max={180}
                                    value={timeLimit}
                                    onChange={(e) => setTimeLimit(Number(e.target.value))}
                                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                                />
                            </div>
                        </div>

                        {/* Error */}
                        {error && (
                            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                                {error}
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-end gap-3">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isLoading || !selectedClassroom || (chapters.length > 0 && selectedTopics.length === 0)}
                            className={clsx(
                                'px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
                                isLoading || !selectedClassroom || (chapters.length > 0 && selectedTopics.length === 0)
                                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    : 'bg-primary-600 text-white hover:bg-primary-700'
                            )}
                        >
                            {isLoading ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    {useAIQuestions ? 'Generating...' : 'Creating...'}
                                </>
                            ) : (
                                <>
                                    {useAIQuestions && <SparklesIcon className="w-4 h-4" />}
                                    Create Assessment
                                </>
                            )}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
