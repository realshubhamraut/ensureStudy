'use client'

import { useState, useEffect } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import {
    AcademicCapIcon,
    CalendarDaysIcon,
    CloudArrowUpIcon,
    SparklesIcon,
    ChartBarIcon,
    BookOpenIcon,
    ArrowPathIcon,
    ChevronRightIcon,
    UserCircleIcon,
    TrashIcon,
    EyeIcon,
    EyeSlashIcon,
    Cog6ToothIcon
} from '@heroicons/react/24/outline'
import dynamic from 'next/dynamic'

// Dynamic imports
const LearningStyleQuiz = dynamic(() => import('@/components/curriculum/LearningStyleQuiz'), { ssr: false })
const ExamPrepModal = dynamic(() => import('@/components/curriculum/ExamPrepModal'), { ssr: false })
const SyllabusUploadModal = dynamic(() => import('@/components/curriculum/SyllabusUploadModal'), { ssr: false })
const WeeklyCalendar = dynamic(() => import('@/components/curriculum/WeeklyCalendar'), { ssr: false })

// ============================================================================
// Types
// ============================================================================

interface Curriculum {
    id: string
    subject_name: string
    total_topics: number
    start_date: string
    end_date: string
}

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

interface TopicScore {
    topic_id: string
    topic_name: string
    confidence_score: number
    assessment_score?: number
    interview_score?: number
    status: string
}

// ============================================================================
// Main Component
// ============================================================================

export default function CurriculumPage() {
    const [curricula, setCurricula] = useState<Curriculum[]>([])
    const [selectedCurriculumId, setSelectedCurriculumId] = useState<string | null>(null)
    const [weeklySchedule, setWeeklySchedule] = useState<WeeklySchedule | null>(null)
    const [topicScores, setTopicScores] = useState<TopicScore[]>([])
    const [loading, setLoading] = useState(true)
    const [scheduleLoading, setScheduleLoading] = useState(false)
    const [weekOffset, setWeekOffset] = useState(0)
    const [mounted, setMounted] = useState(false)  // Client-side only guard

    // Modal states
    const [showLearningStyleQuiz, setShowLearningStyleQuiz] = useState(false)
    const [showExamPrepModal, setShowExamPrepModal] = useState(false)
    const [showSyllabusUpload, setShowSyllabusUpload] = useState(false)

    // Config modal state - which curriculum is being reconfigured
    const [configCurriculumId, setConfigCurriculumId] = useState<string | null>(null)

    // Hidden curricula state (for hiding subjects from calendar view)
    const [hiddenCurricula, setHiddenCurricula] = useState<Set<string>>(new Set())
    const [hoverCurriculumId, setHoverCurriculumId] = useState<string | null>(null)

    // ========================================================================
    // API Calls
    // ========================================================================

    const fetchCurricula = async () => {
        try {
            const userId = localStorage.getItem('userId') || 'demo-user'
            const res = await fetch(`${getAiServiceUrl()}/api/curriculum/user/${userId}`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                setCurricula(data.curricula || [])
                // Auto-select first curriculum
                if (data.curricula?.length > 0 && !selectedCurriculumId) {
                    setSelectedCurriculumId(data.curricula[0].id)
                }
            }
        } catch (error) {
            console.error('Failed to fetch curricula:', error)
        } finally {
            setLoading(false)
        }
    }

    const fetchWeeklySchedule = async (curriculumId: string, offset: number) => {
        setScheduleLoading(true)
        try {
            const res = await fetch(
                `${getAiServiceUrl()}/api/curriculum/schedule/${curriculumId}?week_offset=${offset}`,
                { headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` } }
            )
            if (res.ok) {
                const data = await res.json()
                setWeeklySchedule(data)
            }
        } catch (error) {
            console.error('Failed to fetch schedule:', error)
        } finally {
            setScheduleLoading(false)
        }
    }

    const fetchTopicScores = async () => {
        try {
            const userId = localStorage.getItem('userId') || 'demo-user'
            const url = selectedCurriculumId
                ? `${getAiServiceUrl()}/api/curriculum/topic-scores/${userId}?curriculum_id=${selectedCurriculumId}`
                : `${getAiServiceUrl()}/api/curriculum/topic-scores/${userId}`

            const res = await fetch(url, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                setTopicScores(data.topics || [])
            }
        } catch (error) {
            console.error('Failed to fetch topic scores:', error)
        }
    }

    const handleReschedule = async (topicId: string, newDate: string) => {
        if (!selectedCurriculumId) return

        try {
            await fetch(`${getAiServiceUrl()}/api/curriculum/reschedule`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    topic_id: topicId,
                    curriculum_id: selectedCurriculumId,
                    new_date: newDate
                })
            })
            // Refresh schedule
            fetchWeeklySchedule(selectedCurriculumId, weekOffset)
        } catch (error) {
            console.error('Failed to reschedule:', error)
        }
    }

    const handleWeekChange = (newOffset: number) => {
        setWeekOffset(newOffset)
        if (selectedCurriculumId) {
            fetchWeeklySchedule(selectedCurriculumId, newOffset)
        }
    }

    // Effects
    useEffect(() => {
        setMounted(true)  // Mark as client-side mounted
        fetchCurricula()
    }, [])

    useEffect(() => {
        if (selectedCurriculumId) {
            fetchWeeklySchedule(selectedCurriculumId, weekOffset)
            fetchTopicScores()
        }
    }, [selectedCurriculumId])

    // ========================================================================
    // Stats Calculation
    // ========================================================================

    const stats = {
        totalTopics: topicScores.length,
        mastered: topicScores.filter(t => t.confidence_score >= 80).length,
        learning: topicScores.filter(t => t.confidence_score >= 50 && t.confidence_score < 80).length,
        needsWork: topicScores.filter(t => t.confidence_score > 0 && t.confidence_score < 50).length,
        notStarted: topicScores.filter(t => t.confidence_score === 0).length,
        avgConfidence: topicScores.length > 0
            ? Math.round(topicScores.reduce((sum, t) => sum + t.confidence_score, 0) / topicScores.length)
            : 0
    }

    // ========================================================================
    // Render
    // ========================================================================

    // Show loading until mounted on client to prevent hydration mismatch
    if (!mounted || loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <ArrowPathIcon className="w-8 h-8 animate-spin text-gray-400" />
            </div>
        )
    }

    return (
        <div className="space-y-6" suppressHydrationWarning>
            {/* Header */}
            <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                        <CalendarDaysIcon className="w-7 h-7 text-primary-600" />
                        My Study Schedule
                    </h1>
                    <p className="text-gray-600 mt-1">Weekly learning calendar with confidence tracking</p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowSyllabusUpload(true)}
                        className="px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
                    >
                        <CloudArrowUpIcon className="w-4 h-4" />
                        Upload PDF
                    </button>
                </div>
            </div>

            {/* Curriculum Selector (if multiple) */}
            {curricula.length >= 1 && (
                <div className="flex gap-3 overflow-x-auto pb-2 items-center">
                    {/* ALL button to show all subjects */}
                    <button
                        onClick={() => {
                            // If any are hidden, show all; otherwise do nothing
                            if (hiddenCurricula.size > 0) {
                                setHiddenCurricula(new Set())
                            }
                        }}
                        className={`px-3 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${hiddenCurricula.size === 0
                            ? 'bg-primary-600 text-white'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                            }`}
                    >
                        ALL
                    </button>

                    {/* Subject tabs with checkbox and delete */}
                    {curricula.map(c => {
                        const isHidden = hiddenCurricula.has(c.id)
                        const isHovered = hoverCurriculumId === c.id
                        return (
                            <div
                                key={c.id}
                                className="relative flex items-center"
                                onMouseEnter={() => setHoverCurriculumId(c.id)}
                                onMouseLeave={() => setHoverCurriculumId(null)}
                            >
                                <div
                                    onClick={() => {
                                        setSelectedCurriculumId(c.id)
                                    }}
                                    className={`px-3 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all flex items-center gap-2 cursor-pointer ${selectedCurriculumId === c.id
                                        ? 'bg-primary-100 text-primary-700'
                                        : isHidden
                                            ? 'bg-gray-100 text-gray-400'
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                        }`}
                                    role="button"
                                    tabIndex={0}
                                >
                                    {/* Checkbox for visibility */}
                                    <input
                                        type="checkbox"
                                        checked={!isHidden}
                                        onClick={(e) => e.stopPropagation()}
                                        onChange={(e) => {
                                            e.stopPropagation()
                                            setHiddenCurricula(prev => {
                                                const newSet = new Set(prev)
                                                if (newSet.has(c.id)) {
                                                    newSet.delete(c.id)
                                                } else {
                                                    newSet.add(c.id)
                                                }
                                                return newSet
                                            })
                                        }}
                                        className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500 cursor-pointer"
                                    />
                                    <span className={isHidden ? 'line-through' : ''}>
                                        {c.subject_name}
                                    </span>

                                    {/* Config and Delete buttons on right, visible on hover */}
                                    <span
                                        className={`transition-all duration-200 overflow-hidden flex items-center ${isHovered ? 'max-w-20 opacity-100 ml-1' : 'max-w-0 opacity-0'}`}
                                    >
                                        {/* Config/Reconfigure button */}
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                setConfigCurriculumId(c.id)
                                                setShowSyllabusUpload(true)
                                            }}
                                            className="p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
                                            title="Reconfigure topics"
                                        >
                                            <Cog6ToothIcon className="w-4 h-4" />
                                        </button>
                                        {/* Delete button */}
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                if (confirm(`Delete "${c.subject_name}" and all its topics? This cannot be undone.`)) {
                                                    const deleteCurriculum = async () => {
                                                        try {
                                                            const userId = localStorage.getItem('userId') || 'demo-user'
                                                            await fetch(`${getAiServiceUrl()}/api/curriculum/${c.id}?user_id=${userId}`, {
                                                                method: 'DELETE',
                                                                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
                                                            })
                                                            setCurricula(prev => prev.filter(curr => curr.id !== c.id))
                                                            if (selectedCurriculumId === c.id) {
                                                                setSelectedCurriculumId(null)
                                                                setWeeklySchedule(null)
                                                                setTopicScores([])
                                                            }
                                                        } catch (error) {
                                                            console.error('Failed to delete curriculum:', error)
                                                            alert('Failed to delete curriculum')
                                                        }
                                                    }
                                                    deleteCurriculum()
                                                }
                                            }}
                                            className="p-1 text-red-500 hover:text-red-600 hover:bg-red-50 rounded"
                                            title="Delete subject"
                                        >
                                            <TrashIcon className="w-4 h-4" />
                                        </button>
                                    </span>
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}

            {/* Main Content */}
            {!selectedCurriculumId ? (
                /* No Curriculum - Empty State */
                <div className="card text-center py-12">
                    <BookOpenIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Study Schedule Yet</h3>
                    <p className="text-gray-500 mb-6 max-w-md mx-auto">
                        Upload a syllabus PDF to generate your personalized study schedule with confidence tracking.
                    </p>
                    <button
                        onClick={() => setShowSyllabusUpload(true)}
                        className="btn-primary inline-flex items-center gap-2"
                    >
                        <CloudArrowUpIcon className="w-5 h-5" />
                        Upload Syllabus
                    </button>
                </div>
            ) : (
                /* Has Curriculum - Show Calendar */
                <div className="space-y-6">
                    {/* Stats Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                        <div className="card p-4 text-center">
                            <p className="text-2xl font-bold text-gray-900">{stats.avgConfidence}%</p>
                            <p className="text-xs text-gray-500">Avg Confidence</p>
                        </div>
                        <div className="card p-4 text-center bg-green-50">
                            <p className="text-2xl font-bold text-green-600">{stats.mastered}</p>
                            <p className="text-xs text-gray-500">Mastered</p>
                        </div>
                        <div className="card p-4 text-center bg-yellow-50">
                            <p className="text-2xl font-bold text-yellow-600">{stats.learning}</p>
                            <p className="text-xs text-gray-500">Learning</p>
                        </div>
                        <div className="card p-4 text-center bg-red-50">
                            <p className="text-2xl font-bold text-red-600">{stats.needsWork}</p>
                            <p className="text-xs text-gray-500">Needs Work</p>
                        </div>
                        <div className="card p-4 text-center">
                            <p className="text-2xl font-bold text-gray-600">{stats.notStarted}</p>
                            <p className="text-xs text-gray-500">Not Started</p>
                        </div>
                    </div>

                    {/* Weekly Calendar */}
                    <div className="card">
                        {selectedCurriculumId && hiddenCurricula.has(selectedCurriculumId) ? (
                            <div className="p-8 text-center">
                                <EyeSlashIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                                <p className="text-gray-500 font-medium">Subject Hidden</p>
                                <p className="text-sm text-gray-400 mt-1">
                                    Topics for this subject are hidden from the calendar view.
                                </p>
                                <button
                                    onClick={() => {
                                        setHiddenCurricula(prev => {
                                            const newSet = new Set(prev)
                                            newSet.delete(selectedCurriculumId!)
                                            return newSet
                                        })
                                    }}
                                    className="mt-4 px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 inline-flex items-center gap-2"
                                >
                                    <EyeIcon className="w-4 h-4" />
                                    Show Topics
                                </button>
                            </div>
                        ) : (
                            <WeeklyCalendar
                                schedule={weeklySchedule}
                                loading={scheduleLoading}
                                onWeekChange={handleWeekChange}
                                onReschedule={handleReschedule}
                                weekOffset={weekOffset}
                            />
                        )}
                    </div>

                    {/* Score Sources Info */}
                    <div className="card p-4 bg-blue-50 border-blue-100">
                        <div className="flex items-start gap-3">
                            <ChartBarIcon className="w-5 h-5 text-blue-600 mt-0.5" />
                            <div>
                                <p className="font-medium text-blue-900">Confidence scores are calculated from:</p>
                                <ul className="text-sm text-blue-700 mt-1 space-y-1">
                                    <li>• <strong>Assessments</strong> - Quiz and test scores (40%)</li>
                                    <li>• <strong>Mock Interviews</strong> - Interview performance (40%)</li>
                                    <li>• <strong>Study Progress</strong> - Material completion (20%)</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Modals */}
            {showLearningStyleQuiz && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto p-6">
                        <LearningStyleQuiz
                            onComplete={(primary, secondary) => {
                                console.log('Learning style:', primary, secondary)
                                setShowLearningStyleQuiz(false)
                            }}
                            onSkip={() => setShowLearningStyleQuiz(false)}
                        />
                    </div>
                </div>
            )}

            {showExamPrepModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <ExamPrepModal
                        topics={topicScores.map(t => ({ id: t.topic_id, name: t.topic_name }))}
                        onClose={() => setShowExamPrepModal(false)}
                    />
                </div>
            )}

            {showSyllabusUpload && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <SyllabusUploadModal
                        onClose={() => {
                            setShowSyllabusUpload(false)
                            setConfigCurriculumId(null)
                        }}
                        onSuccess={() => {
                            setShowSyllabusUpload(false)
                            const currId = configCurriculumId
                            setConfigCurriculumId(null)
                            fetchCurricula()
                            // If reconfiguring current curriculum, refresh the schedule
                            if (currId && currId === selectedCurriculumId) {
                                setWeekOffset(0)
                                fetchWeeklySchedule(currId, 0)
                                fetchTopicScores()
                            }
                        }}
                        curriculumId={configCurriculumId || undefined}
                        subjectName={configCurriculumId ? curricula.find(c => c.id === configCurriculumId)?.subject_name : undefined}
                    />
                </div>
            )}
        </div>
    )
}
