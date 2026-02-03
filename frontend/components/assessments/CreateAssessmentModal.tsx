'use client'

import { useState, useEffect } from 'react'
import { XMarkIcon, ChevronDownIcon, ChevronRightIcon, SparklesIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'
import { getApiBaseUrl, getAiServiceUrl } from '@/utils/api'
import LearningAgentStatus from './LearningAgentStatus'

interface Chapter {
    id: string
    name: string
    color?: string
    topics: Topic[]
}

interface Topic {
    id: string
    name: string
    difficulty: 'easy' | 'medium' | 'hard'
    confidence?: number  // Mastery percentage from StudentTopicScore (0-100)
    mcq_attempts?: number
    status?: 'not_started' | 'learning' | 'practicing' | 'mastered'
}

interface WeakTopic {
    topic_id: string
    topic_name: string
    score: number
}

interface Classroom {
    id: string
    name: string
    subject_name: string
}

interface CreateAssessmentModalProps {
    isOpen: boolean
    onClose: () => void
    classrooms: Classroom[]
    onSuccess?: (assessment?: any) => void
}

export default function CreateAssessmentModal({
    isOpen,
    onClose,
    classrooms,
    onSuccess
}: CreateAssessmentModalProps) {
    const [title, setTitle] = useState('')
    const [selectedClassroom, setSelectedClassroom] = useState('')
    const [chapters, setChapters] = useState<Chapter[]>([])
    const [selectedChapters, setSelectedChapters] = useState<string[]>([])
    const [selectedTopics, setSelectedTopics] = useState<string[]>([])
    const [expandedChapters, setExpandedChapters] = useState<string[]>([])
    const [weakTopics, setWeakTopics] = useState<WeakTopic[]>([])
    // Learning Agent handles question generation automatically
    const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard' | 'mixed'>('mixed')
    const [questionCount, setQuestionCount] = useState(10)
    const [timeLimit, setTimeLimit] = useState(30)
    const [isLoading, setIsLoading] = useState(false)
    const [isLoadingHierarchy, setIsLoadingHierarchy] = useState(false)
    const [noSyllabusFound, setNoSyllabusFound] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [topicScores, setTopicScores] = useState<Record<string, { confidence: number, attempts: number }>>({}
    )

    // Fetch syllabus hierarchy when classroom changes
    useEffect(() => {
        if (!selectedClassroom) {
            setChapters([])
            setSelectedChapters([])
            setSelectedTopics([])
            setWeakTopics([])
            setNoSyllabusFound(false)
            setTopicScores({})
            return
        }

        const fetchHierarchy = async () => {
            setIsLoadingHierarchy(true)
            setNoSyllabusFound(false)
            try {
                const token = typeof window !== 'undefined' ? localStorage.getItem('accessToken') : null

                // Fetch chapters and topics from core service
                const res = await fetch(`${getApiBaseUrl()}/api/classroom/${selectedClassroom}/chapters`)

                // Also fetch topic mastery scores
                const scoresRes = await fetch(`${getApiBaseUrl()}/api/progress/topic-mastery?classroom_id=${selectedClassroom}`, {
                    headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                })

                // Build scores map
                const scoresMap: Record<string, { confidence: number, attempts: number }> = {}
                if (scoresRes.ok) {
                    const scoresData = await scoresRes.json()
                    for (const t of (scoresData.topics || [])) {
                        scoresMap[t.topic_id] = {
                            confidence: t.mastery_level || 0,
                            attempts: t.total_attempts || 0
                        }
                    }
                    setTopicScores(scoresMap)
                }

                if (res.ok) {
                    const data = await res.json()
                    const fetchedChapters = Array.isArray(data) ? data : (data.chapters || [])

                    // Merge confidence scores into topics
                    const chaptersWithScores = fetchedChapters.map((ch: Chapter) => ({
                        ...ch,
                        topics: (ch.topics || []).map((t: Topic) => ({
                            ...t,
                            confidence: scoresMap[t.id]?.confidence || 0,
                            mcq_attempts: scoresMap[t.id]?.attempts || 0
                        }))
                    }))

                    setChapters(chaptersWithScores)

                    if (fetchedChapters.length === 0) {
                        setNoSyllabusFound(true)
                    } else {
                        // Auto-expand first chapter
                        setExpandedChapters([fetchedChapters[0].id])
                    }
                } else {
                    console.error('Failed to fetch chapters:', res.status)
                    setNoSyllabusFound(true)
                }
            } catch (err) {
                console.error('Failed to fetch hierarchy:', err)
                setNoSyllabusFound(true)
            }
            setIsLoadingHierarchy(false)
        }

        const fetchWeakTopics = async () => {
            try {
                const token = typeof window !== 'undefined' ? localStorage.getItem('accessToken') : null
                const res = await fetch(`${getApiBaseUrl()}/api/assessments/weak-topics?classroom_id=${selectedClassroom}`, {
                    headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                })
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

    // Note: Weak topics auto-selection removed - Learning Agent handles topic prioritization

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
            setSelectedTopics(prev => Array.from(new Set([...prev, ...chapterTopicIds])))
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

    // Quick selection helpers
    const selectAllTopics = () => {
        const allTopicIds = chapters.flatMap(c => c.topics?.map(t => t.id) || [])
        const allChapterIds = chapters.map(c => c.id)
        setSelectedTopics(allTopicIds)
        setSelectedChapters(allChapterIds)
        setExpandedChapters(allChapterIds)
    }

    const clearAllSelections = () => {
        setSelectedTopics([])
        setSelectedChapters([])
    }

    const selectWeakTopicsOnly = () => {
        const weakTopicIds = weakTopics.map(w => w.topic_id)
        setSelectedTopics(weakTopicIds)
        // Also select chapters containing weak topics
        const chaptersWithWeakTopics = chapters.filter(c =>
            c.topics?.some(t => weakTopicIds.includes(t.id))
        )
        setSelectedChapters(chaptersWithWeakTopics.map(c => c.id))
        setExpandedChapters(chaptersWithWeakTopics.map(c => c.id))
    }

    const selectByDifficulty = (diff: string) => {
        const matchingTopicIds = chapters.flatMap(c =>
            c.topics?.filter(t => t.difficulty === diff).map(t => t.id) || []
        )
        const chaptersWithMatching = chapters.filter(c =>
            c.topics?.some(t => t.difficulty === diff)
        )
        setSelectedTopics(matchingTopicIds)
        setSelectedChapters(chaptersWithMatching.map(c => c.id))
        setExpandedChapters(chaptersWithMatching.map(c => c.id))
    }

    const getSelectionStats = () => {
        const totalTopics = chapters.reduce((sum, c) => sum + (c.topics?.length || 0), 0)
        const weakCount = weakTopics.length
        const easyCount = chapters.flatMap(c => c.topics || []).filter(t => t.difficulty === 'easy').length
        const hardCount = chapters.flatMap(c => c.topics || []).filter(t => t.difficulty === 'hard').length
        return { totalTopics, weakCount, easyCount, hardCount }
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)
        setIsLoading(true)

        try {
            const classroom = classrooms.find(c => c.id === selectedClassroom)

            // Generate AI questions via Type 5 Learning Agent
            const aiRes = await fetch(`${getAiServiceUrl()}/api/questions/generate-assessment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    classroom_id: selectedClassroom,
                    topic_ids: selectedTopics,
                    chapter_ids: [],
                    difficulty,
                    num_questions: questionCount,
                    question_type: 'mcq',
                    title: title || undefined
                })
            })


            if (!aiRes.ok) {
                throw new Error('Failed to generate AI questions')
            }

            const aiData = await aiRes.json()
            console.log('AI generated questions:', aiData)

            // Check if AI generation was successful
            if (!aiData.success || !aiData.questions || aiData.questions.length === 0) {
                throw new Error(aiData.error || 'No questions were generated. Please try again.')
            }

            // Create assessment with generated questions
            const createRes = await fetch(`${getApiBaseUrl()}/api/assessments/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({
                    title: title || `${classroom?.subject_name || 'Assessment'} Quiz`,
                    classroom_id: selectedClassroom,
                    topic_ids: selectedTopics,
                    questions: aiData.questions || [],
                    time_limit_minutes: timeLimit,
                    difficulty,
                    assessment_type: 'self_practice'
                })
            })

            if (!createRes.ok) {
                const errData = await createRes.json().catch(() => ({}))
                console.error('Create assessment error:', errData)
                throw new Error(errData.error || 'Failed to create assessment')
            }

            const createdAssessment = await createRes.json()
            onSuccess?.(createdAssessment)

            onClose()
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Something went wrong')
        } finally {
            setIsLoading(false)
        }
    }

    if (!isOpen) return null

    const selectedTopicCount = selectedTopics.length
    const stats = getSelectionStats()

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between bg-gradient-to-r from-primary-500 to-primary-600">
                    <h2 className="text-xl font-bold text-white">Create New Assessment</h2>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-white/20 rounded-lg transition-colors"
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
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                                required
                            >
                                <option value="">Select a classroom</option>
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
                                    <div className="space-y-3">
                                        {/* Quick Selection Buttons */}
                                        <div className="flex flex-wrap gap-2">
                                            <button
                                                type="button"
                                                onClick={selectAllTopics}
                                                className="px-3 py-1.5 text-xs font-medium bg-primary-100 text-primary-700 rounded-full hover:bg-primary-200 transition-colors"
                                            >
                                                ✓ Select All ({stats.totalTopics})
                                            </button>
                                            {weakTopics.length > 0 && (
                                                <button
                                                    type="button"
                                                    onClick={selectWeakTopicsOnly}
                                                    className="px-3 py-1.5 text-xs font-medium bg-red-100 text-red-700 rounded-full hover:bg-red-200 transition-colors"
                                                >
                                                    🎯 Weak Topics ({stats.weakCount})
                                                </button>
                                            )}
                                            {stats.easyCount > 0 && (
                                                <button
                                                    type="button"
                                                    onClick={() => selectByDifficulty('easy')}
                                                    className="px-3 py-1.5 text-xs font-medium bg-green-100 text-green-700 rounded-full hover:bg-green-200 transition-colors"
                                                >
                                                    Easy ({stats.easyCount})
                                                </button>
                                            )}
                                            {stats.hardCount > 0 && (
                                                <button
                                                    type="button"
                                                    onClick={() => selectByDifficulty('hard')}
                                                    className="px-3 py-1.5 text-xs font-medium bg-orange-100 text-orange-700 rounded-full hover:bg-orange-200 transition-colors"
                                                >
                                                    Hard ({stats.hardCount})
                                                </button>
                                            )}
                                            {selectedTopics.length > 0 && (
                                                <button
                                                    type="button"
                                                    onClick={clearAllSelections}
                                                    className="px-3 py-1.5 text-xs font-medium bg-gray-100 text-gray-600 rounded-full hover:bg-gray-200 transition-colors"
                                                >
                                                    ✕ Clear All
                                                </button>
                                            )}
                                        </div>

                                        {/* Selection Summary */}
                                        {selectedTopics.length > 0 && (
                                            <div className="text-xs text-gray-500 bg-gray-50 px-3 py-2 rounded-lg">
                                                <span className="font-medium text-primary-600">{selectedTopics.length}</span> topics from{' '}
                                                <span className="font-medium text-primary-600">{selectedChapters.length}</span> chapter(s) selected
                                            </div>
                                        )}

                                        {/* Chapters & Topics List */}
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
                                                                        {/* Confidence score badge */}
                                                                        {typeof topic.confidence === 'number' && (
                                                                            <span className={clsx(
                                                                                'text-xs px-1.5 py-0.5 rounded-full font-medium',
                                                                                topic.confidence >= 70 ? 'bg-green-100 text-green-700' :
                                                                                    topic.confidence >= 50 ? 'bg-yellow-100 text-yellow-700' :
                                                                                        topic.confidence > 0 ? 'bg-red-100 text-red-700' :
                                                                                            'bg-gray-100 text-gray-500'
                                                                            )}>
                                                                                {topic.confidence > 0 ? `${Math.round(topic.confidence)}%` : 'New'}
                                                                            </span>
                                                                        )}
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
                                    </div>
                                )}

                                {/* Learning Agent Status - Type 5 AI */}
                                <div className="pt-4 border-t border-gray-200 mt-4">
                                    <LearningAgentStatus
                                        classroomId={selectedClassroom}
                                        selectedTopics={selectedTopics}
                                    />
                                </div>

                                {/* Difficulty */}
                                <div className="pt-4">
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
                                <div className="grid grid-cols-2 gap-4 pt-4">
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
                                    Generating...
                                </>
                            ) : (
                                <>
                                    <SparklesIcon className="w-4 h-4" />
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
