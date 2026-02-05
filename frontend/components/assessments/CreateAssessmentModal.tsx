'use client'

import { useState, useEffect } from 'react'
import {
    XMarkIcon,
    ChevronDownIcon,
    ChevronRightIcon,
    SparklesIcon,
    ExclamationTriangleIcon,
    AcademicCapIcon,
    ClockIcon,
    CheckCircleIcon,
    BookOpenIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'
import { getApiBaseUrl, getAiServiceUrl } from '@/utils/api'
import LearningAgentStatus from './LearningAgentStatus'

interface Chapter {
    id: string
    name: string
    color?: string
    topics: Topic[]
    classroomId?: string
    classroomName?: string
}

interface Topic {
    id: string
    name: string
    difficulty: 'easy' | 'medium' | 'hard'
    confidence?: number
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
    // Multi-subject selection
    const [selectedClassrooms, setSelectedClassrooms] = useState<string[]>([])
    const [chapters, setChapters] = useState<Chapter[]>([])
    const [selectedChapters, setSelectedChapters] = useState<string[]>([])
    const [selectedTopics, setSelectedTopics] = useState<string[]>([])
    const [expandedChapters, setExpandedChapters] = useState<string[]>([])
    const [weakTopics, setWeakTopics] = useState<WeakTopic[]>([])
    const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard' | 'mixed'>('mixed')
    const [questionCount, setQuestionCount] = useState(10)
    const [timeLimit, setTimeLimit] = useState(30)
    const [isLoading, setIsLoading] = useState(false)
    const [isLoadingHierarchy, setIsLoadingHierarchy] = useState(false)
    const [noSyllabusFound, setNoSyllabusFound] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [topicScores, setTopicScores] = useState<Record<string, { confidence: number, attempts: number }>>({})
    // Form step for visual progress
    const [currentStep, setCurrentStep] = useState(1)

    // Fetch syllabus hierarchy when classrooms change
    useEffect(() => {
        if (selectedClassrooms.length === 0) {
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
                const allChapters: Chapter[] = []
                const scoresMap: Record<string, { confidence: number, attempts: number }> = {}

                // Fetch from all selected classrooms
                for (const classroomId of selectedClassrooms) {
                    const classroom = classrooms.find(c => c.id === classroomId)

                    // Fetch chapters and topics
                    const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/chapters`)

                    // Fetch topic scores
                    const scoresRes = await fetch(`${getApiBaseUrl()}/api/progress/topic-mastery?classroom_id=${classroomId}`, {
                        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                    })

                    if (scoresRes.ok) {
                        const scoresData = await scoresRes.json()
                        for (const t of (scoresData.topics || [])) {
                            scoresMap[t.topic_id] = {
                                confidence: t.mastery_level || 0,
                                attempts: t.total_attempts || 0
                            }
                        }
                    }

                    if (res.ok) {
                        const data = await res.json()
                        const fetchedChapters = Array.isArray(data) ? data : (data.chapters || [])

                        // Add classroom info and scores to chapters
                        const chaptersWithMeta = fetchedChapters.map((ch: Chapter) => ({
                            ...ch,
                            classroomId,
                            classroomName: classroom?.subject_name || classroom?.name,
                            topics: (ch.topics || []).map((t: Topic) => ({
                                ...t,
                                confidence: scoresMap[t.id]?.confidence || 0,
                                mcq_attempts: scoresMap[t.id]?.attempts || 0
                            }))
                        }))

                        allChapters.push(...chaptersWithMeta)
                    }
                }

                setChapters(allChapters)
                setTopicScores(scoresMap)

                if (allChapters.length === 0) {
                    setNoSyllabusFound(true)
                } else {
                    // Auto-expand first chapter
                    setExpandedChapters([allChapters[0].id])
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
                const allWeakTopics: WeakTopic[] = []

                for (const classroomId of selectedClassrooms) {
                    const res = await fetch(`${getApiBaseUrl()}/api/assessments/weak-topics?classroom_id=${classroomId}`, {
                        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                    })
                    if (res.ok) {
                        const data = await res.json()
                        allWeakTopics.push(...(data.weak_topics || []))
                    }
                }
                setWeakTopics(allWeakTopics)
            } catch (err) {
                console.error('Failed to fetch weak topics:', err)
            }
        }

        fetchHierarchy()
        fetchWeakTopics()
    }, [selectedClassrooms, classrooms])

    // Handle classroom selection toggle
    const toggleClassroom = (classroomId: string) => {
        setSelectedClassrooms(prev =>
            prev.includes(classroomId)
                ? prev.filter(id => id !== classroomId)
                : [...prev, classroomId]
        )
    }

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
            setSelectedChapters(prev => prev.filter(id => id !== chapterId))
            const chapterTopicIds = chapter.topics.map(t => t.id).filter((id): id is string => id != null)
            setSelectedTopics(prev => prev.filter(id => !chapterTopicIds.includes(id)))
        } else {
            setSelectedChapters(prev => [...prev, chapterId])
            const chapterTopicIds = chapter.topics.map(t => t.id).filter((id): id is string => id != null)
            setSelectedTopics(prev => Array.from(new Set([...prev, ...chapterTopicIds])))
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

    const selectAllTopics = () => {
        const allTopicIds = chapters.flatMap(c => c.topics?.map(t => t.id) || []).filter((id): id is string => id != null)
        const allChapterIds = chapters.map(c => c.id).filter((id): id is string => id != null)
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
        const chaptersWithWeakTopics = chapters.filter(c =>
            c.topics?.some(t => weakTopicIds.includes(t.id))
        )
        setSelectedChapters(chaptersWithWeakTopics.map(c => c.id))
        setExpandedChapters(chaptersWithWeakTopics.map(c => c.id))
    }

    const getSelectionStats = () => {
        const totalTopics = chapters.reduce((sum, c) => sum + (c.topics?.length || 0), 0)
        const weakCount = weakTopics.length
        const easyCount = chapters.flatMap(c => c.topics || []).filter(t => t.difficulty === 'easy').length
        const hardCount = chapters.flatMap(c => c.topics || []).filter(t => t.difficulty === 'hard').length
        return { totalTopics, weakCount, easyCount, hardCount }
    }

    // Auto-suggest question count based on selected topics
    const suggestedQuestionCount = Math.min(30, Math.max(5, Math.ceil(selectedTopics.length * 2)))

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)

        // Validation guard
        if (selectedClassrooms.length === 0) {
            setError('Please select at least one subject')
            return
        }

        setIsLoading(true)

        try {
            const selectedSubjects = selectedClassrooms.map(id =>
                classrooms.find(c => c.id === id)?.subject_name
            ).filter(Boolean).join(', ')

            // Generate AI questions via Learning Agent
            // Filter out null/undefined values from topic IDs
            const validTopicIds = selectedTopics.filter((id): id is string => id != null && id !== '')

            const requestBody: Record<string, any> = {
                classroom_id: selectedClassrooms[0] || '', // Primary classroom (required)
                classroom_ids: selectedClassrooms.filter(id => id != null),
                topic_ids: validTopicIds,
                chapter_ids: [],
                difficulty: difficulty,
                num_questions: questionCount,
                question_type: 'mcq',
                time_limit_minutes: timeLimit,
                include_weak_topics: false
            }

            // Only include title if provided
            if (title && title.trim()) {
                requestBody.title = title.trim()
            }

            console.log('[Assessment] Generating questions with:', requestBody)

            const aiRes = await fetch(`${getAiServiceUrl()}/api/questions/generate-assessment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            })

            if (!aiRes.ok) {
                const errorData = await aiRes.json().catch(() => ({}))
                console.error('[Assessment] AI generation error:', errorData)

                // Parse FastAPI validation errors properly
                let errorMessage = 'Failed to generate AI questions'
                if (Array.isArray(errorData.detail)) {
                    // FastAPI validation error format: [{loc: [...], msg: "...", type: "..."}]
                    errorMessage = errorData.detail.map((e: any) =>
                        `${e.loc?.join('.') || 'field'}: ${e.msg}`
                    ).join('; ')
                } else if (typeof errorData.detail === 'string') {
                    errorMessage = errorData.detail
                } else if (errorData.error) {
                    errorMessage = errorData.error
                }

                throw new Error(errorMessage)
            }

            const aiData = await aiRes.json()

            if (!aiData.success || !aiData.questions || aiData.questions.length === 0) {
                throw new Error(aiData.error || 'No questions were generated. Please try again.')
            }

            // Create assessment
            const autoTitle = title ||
                (selectedClassrooms.length > 1
                    ? `Mixed Assessment: ${selectedSubjects}`
                    : `${selectedSubjects} Quiz`)

            const createRes = await fetch(`${getApiBaseUrl()}/api/assessments/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({
                    title: autoTitle,
                    classroom_id: selectedClassrooms[0],
                    classroom_ids: selectedClassrooms,
                    topic_ids: selectedTopics,
                    questions: aiData.questions || [],
                    time_limit_minutes: timeLimit,
                    difficulty,
                    assessment_type: 'self_practice',
                    is_mixed: selectedClassrooms.length > 1
                })
            })

            if (!createRes.ok) {
                const errData = await createRes.json().catch(() => ({}))
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

    // Group chapters by classroom for better organization
    const chaptersByClassroom = selectedClassrooms.map(classroomId => ({
        classroom: classrooms.find(c => c.id === classroomId),
        chapters: chapters.filter(ch => ch.classroomId === classroomId)
    }))

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-indigo-600 to-purple-600">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-white/20 rounded-lg">
                                <AcademicCapIcon className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white">Create Assessment</h2>
                                <p className="text-sm text-white/70">Generate AI-powered quiz questions</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                        >
                            <XMarkIcon className="w-6 h-6 text-white" />
                        </button>
                    </div>

                    {/* Progress Steps */}
                    <div className="flex items-center gap-2 mt-4">
                        {[
                            { step: 1, label: 'Subjects', icon: BookOpenIcon },
                            { step: 2, label: 'Topics', icon: CheckCircleIcon },
                            { step: 3, label: 'Settings', icon: ClockIcon }
                        ].map(({ step, label, icon: Icon }) => (
                            <div key={step} className="flex items-center">
                                <div className={clsx(
                                    'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all',
                                    currentStep >= step
                                        ? 'bg-white text-indigo-600'
                                        : 'bg-white/20 text-white/70'
                                )}>
                                    <Icon className="w-4 h-4" />
                                    {label}
                                </div>
                                {step < 3 && (
                                    <div className={clsx(
                                        'w-8 h-0.5 mx-1',
                                        currentStep > step ? 'bg-white' : 'bg-white/30'
                                    )} />
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Body */}
                <form onSubmit={handleSubmit} className="overflow-y-auto max-h-[calc(90vh-180px)]">
                    <div className="p-6 space-y-6">

                        {/* Step 1: Subject/Classroom Selection */}
                        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-5 border border-indigo-100">
                            <label className="flex items-center gap-2 text-sm font-semibold text-gray-800 mb-3">
                                <BookOpenIcon className="w-5 h-5 text-indigo-600" />
                                Select Subjects
                                <span className="text-xs text-gray-500 font-normal">(Select one or more for mixed assessment)</span>
                            </label>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                {classrooms.map(classroom => (
                                    <label
                                        key={classroom.id}
                                        className={clsx(
                                            'flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all',
                                            selectedClassrooms.includes(classroom.id)
                                                ? 'border-indigo-500 bg-white shadow-sm'
                                                : 'border-transparent bg-white/50 hover:bg-white hover:border-gray-200'
                                        )}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedClassrooms.includes(classroom.id)}
                                            onChange={() => {
                                                toggleClassroom(classroom.id)
                                                setCurrentStep(selectedClassrooms.length >= 1 ? 2 : 1)
                                            }}
                                            className="w-5 h-5 text-indigo-600 rounded focus:ring-indigo-500"
                                        />
                                        <div className="flex-1">
                                            <span className="font-medium text-gray-900">{classroom.name}</span>
                                            <span className="ml-2 text-xs px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full">
                                                {classroom.subject_name}
                                            </span>
                                        </div>
                                    </label>
                                ))}
                            </div>

                            {selectedClassrooms.length > 1 && (
                                <div className="mt-3 flex items-center gap-2 px-3 py-2 bg-purple-100 rounded-lg">
                                    <SparklesIcon className="w-4 h-4 text-purple-600" />
                                    <span className="text-sm text-purple-800 font-medium">
                                        Mixed Assessment: Questions from {selectedClassrooms.length} subjects
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Title (Optional) */}
                        {selectedClassrooms.length > 0 && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Assessment Title <span className="text-gray-400">(Optional)</span>
                                </label>
                                <input
                                    type="text"
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                    placeholder={selectedClassrooms.length > 1
                                        ? "e.g., Mixed Subject Review"
                                        : "e.g., Chapter 5 Review Quiz"
                                    }
                                    className="w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                                />
                            </div>
                        )}

                        {/* Step 2: Chapters & Topics Hierarchy */}
                        {selectedClassrooms.length > 0 && (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                                        <CheckCircleIcon className="w-5 h-5 text-green-600" />
                                        Select Topics
                                        {selectedTopicCount > 0 && (
                                            <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                                                {selectedTopicCount} selected
                                            </span>
                                        )}
                                    </label>
                                </div>

                                {isLoadingHierarchy ? (
                                    <div className="p-8 text-center bg-gray-50 rounded-xl">
                                        <div className="w-8 h-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                                        <p className="text-gray-500">Loading syllabus topics...</p>
                                    </div>
                                ) : noSyllabusFound ? (
                                    <div className="p-5 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-3">
                                        <ExclamationTriangleIcon className="w-6 h-6 text-amber-600 flex-shrink-0" />
                                        <div>
                                            <p className="font-medium text-amber-800">No syllabus topics found</p>
                                            <p className="text-sm text-amber-700 mt-1">
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
                                                className="px-3 py-1.5 text-xs font-medium bg-indigo-100 text-indigo-700 rounded-full hover:bg-indigo-200 transition-colors"
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

                                        {/* Chapters grouped by Classroom */}
                                        <div className="border border-gray-200 rounded-xl overflow-hidden max-h-72 overflow-y-auto">
                                            {chaptersByClassroom.map(({ classroom, chapters: classroomChapters }) => (
                                                <div key={classroom?.id}>
                                                    {/* Classroom Header (when multi-select) */}
                                                    {selectedClassrooms.length > 1 && (
                                                        <div className="sticky top-0 bg-gradient-to-r from-indigo-500 to-purple-500 px-4 py-2 text-white font-medium text-sm z-10">
                                                            {classroom?.subject_name || classroom?.name}
                                                        </div>
                                                    )}

                                                    {classroomChapters.map(chapter => (
                                                        <div key={chapter.id} className="border-b border-gray-100 last:border-b-0">
                                                            {/* Chapter Header */}
                                                            <div
                                                                className="flex items-center gap-2 px-4 py-3 hover:bg-gray-50 cursor-pointer"
                                                                onClick={() => toggleChapterExpand(chapter.id)}
                                                            >
                                                                <button type="button" className="p-0.5">
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
                                                                        setCurrentStep(2)
                                                                    }}
                                                                    className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500"
                                                                />
                                                                <div
                                                                    className="w-3 h-3 rounded-full"
                                                                    style={{ backgroundColor: chapter.color || '#6366F1' }}
                                                                />
                                                                <span className="font-medium text-gray-900 flex-1">{chapter.name}</span>
                                                                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
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
                                                                                    'flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors',
                                                                                    selectedTopics.includes(topic.id)
                                                                                        ? 'bg-indigo-100'
                                                                                        : 'hover:bg-gray-100'
                                                                                )}
                                                                            >
                                                                                <input
                                                                                    type="checkbox"
                                                                                    checked={selectedTopics.includes(topic.id)}
                                                                                    onChange={() => {
                                                                                        handleTopicToggle(topic.id, chapter.id)
                                                                                        setCurrentStep(2)
                                                                                    }}
                                                                                    className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500"
                                                                                />
                                                                                <span className="text-sm text-gray-700 flex-1">{topic.name}</span>

                                                                                {/* Badges */}
                                                                                <div className="flex items-center gap-1.5">
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
                                                                                </div>
                                                                            </label>
                                                                        )
                                                                    })}
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Learning Agent Status */}
                                {selectedTopics.length > 0 && (
                                    <div className="pt-4 border-t border-gray-200">
                                        <LearningAgentStatus
                                            classroomId={selectedClassrooms[0]}
                                            selectedTopics={selectedTopics}
                                        />
                                    </div>
                                )}

                                {/* Step 3: Settings Card */}
                                {selectedTopics.length > 0 && (
                                    <div
                                        className="bg-gradient-to-br from-gray-50 to-slate-50 rounded-xl p-5 border border-gray-200"
                                        onClick={() => setCurrentStep(3)}
                                    >
                                        <label className="flex items-center gap-2 text-sm font-semibold text-gray-800 mb-4">
                                            <ClockIcon className="w-5 h-5 text-blue-600" />
                                            Assessment Settings
                                        </label>

                                        {/* Difficulty */}
                                        <div className="mb-4">
                                            <span className="text-sm text-gray-600 mb-2 block">Difficulty Level</span>
                                            <div className="flex gap-2">
                                                {(['easy', 'medium', 'hard', 'mixed'] as const).map(level => (
                                                    <button
                                                        key={level}
                                                        type="button"
                                                        onClick={() => setDifficulty(level)}
                                                        className={clsx(
                                                            'flex-1 px-3 py-2 rounded-lg text-sm font-medium capitalize transition-all',
                                                            difficulty === level
                                                                ? level === 'easy' ? 'bg-green-500 text-white shadow-md'
                                                                    : level === 'medium' ? 'bg-yellow-500 text-white shadow-md'
                                                                        : level === 'hard' ? 'bg-red-500 text-white shadow-md'
                                                                            : 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-md'
                                                                : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-200'
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
                                                <span className="text-sm text-gray-600 mb-2 block">Questions</span>
                                                <div className="relative">
                                                    <select
                                                        value={questionCount}
                                                        onChange={(e) => setQuestionCount(Number(e.target.value))}
                                                        className="w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 appearance-none bg-white"
                                                    >
                                                        {[5, 10, 15, 20, 25, 30].map(n => (
                                                            <option key={n} value={n}>{n} questions</option>
                                                        ))}
                                                    </select>
                                                    <ChevronDownIcon className="w-4 h-4 text-gray-500 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                                                </div>
                                                {suggestedQuestionCount !== questionCount && (
                                                    <button
                                                        type="button"
                                                        onClick={() => setQuestionCount(suggestedQuestionCount)}
                                                        className="text-xs text-indigo-600 mt-1 hover:underline"
                                                    >
                                                        Suggested: {suggestedQuestionCount}
                                                    </button>
                                                )}
                                            </div>
                                            <div>
                                                <span className="text-sm text-gray-600 mb-2 block">Time Limit</span>
                                                <div className="relative">
                                                    <input
                                                        type="number"
                                                        min={5}
                                                        max={180}
                                                        value={timeLimit}
                                                        onChange={(e) => setTimeLimit(Number(e.target.value))}
                                                        className="w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500"
                                                    />
                                                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-gray-500">min</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Error */}
                                {error && (
                                    <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex items-start gap-2">
                                        <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0" />
                                        {error}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-between">
                        <div className="text-sm text-gray-500">
                            {selectedClassrooms.length > 0 && selectedTopics.length > 0 && (
                                <span>
                                    <strong>{selectedTopics.length}</strong> topics from{' '}
                                    <strong>{selectedClassrooms.length}</strong> subject(s)
                                </span>
                            )}
                        </div>
                        <div className="flex gap-3">
                            <button
                                type="button"
                                onClick={onClose}
                                className="px-5 py-2.5 text-gray-700 hover:bg-gray-100 rounded-xl transition-colors font-medium"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={isLoading || selectedClassrooms.length === 0 || (chapters.length > 0 && selectedTopics.length === 0)}
                                className={clsx(
                                    'px-6 py-2.5 rounded-xl font-medium transition-all flex items-center gap-2',
                                    isLoading || selectedClassrooms.length === 0 || (chapters.length > 0 && selectedTopics.length === 0)
                                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                        : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-lg hover:scale-[1.02]'
                                )}
                            >
                                {isLoading ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                        Generating...
                                    </>
                                ) : (
                                    <>
                                        <SparklesIcon className="w-5 h-5" />
                                        Create Assessment
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    )
}
