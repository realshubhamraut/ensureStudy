'use client'

import { useState, useEffect } from 'react'
import { getApiBaseUrl } from '@/utils/api'
import {
    ChartBarIcon,
    AcademicCapIcon,
    TrophyIcon,
    FireIcon,
    ArrowTrendingUpIcon,
    ClockIcon,
    CheckCircleIcon,
    ArrowPathIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface TopicScore {
    topic_id: string
    topic_name: string
    chapter_name: string
    chapter_color: string
    mastery_level: number // 0-100
    quiz_score: number
    interview_score: number
    total_attempts: number
    last_activity: string
}

interface ChapterProgress {
    chapter_id: string
    name: string
    color: string
    topics_count: number
    topics_mastered: number
    average_mastery: number
}

interface ProgressStats {
    total_topics: number
    topics_started: number
    topics_mastered: number
    average_mastery: number
    total_study_hours: number
    current_streak: number
}

interface Props {
    classroomId?: string
    userId?: string
    compact?: boolean
}

// ============================================================================
// Mastery Level Helpers
// ============================================================================

const getMasteryColor = (level: number): string => {
    if (level >= 80) return 'text-green-600 bg-green-100'
    if (level >= 60) return 'text-blue-600 bg-blue-100'
    if (level >= 40) return 'text-yellow-600 bg-yellow-100'
    if (level >= 20) return 'text-orange-600 bg-orange-100'
    return 'text-red-600 bg-red-100'
}

const getMasteryLabel = (level: number): string => {
    if (level >= 80) return 'Mastered'
    if (level >= 60) return 'Proficient'
    if (level >= 40) return 'Developing'
    if (level >= 20) return 'Beginner'
    return 'Not Started'
}

const getMasteryBarColor = (level: number): string => {
    if (level >= 80) return 'bg-green-500'
    if (level >= 60) return 'bg-blue-500'
    if (level >= 40) return 'bg-yellow-500'
    if (level >= 20) return 'bg-orange-500'
    return 'bg-red-500'
}

// ============================================================================
// Main Component
// ============================================================================

export default function ProgressDashboard({ classroomId, userId, compact = false }: Props) {
    const [stats, setStats] = useState<ProgressStats | null>(null)
    const [chapterProgress, setChapterProgress] = useState<ChapterProgress[]>([])
    const [topicScores, setTopicScores] = useState<TopicScore[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [selectedChapter, setSelectedChapter] = useState<string | null>(null)

    // ========================================================================
    // Fetch Data
    // ========================================================================

    useEffect(() => {
        fetchProgress()
    }, [classroomId, userId])

    const fetchProgress = async () => {
        setLoading(true)
        setError('')

        try {
            const params = new URLSearchParams()
            if (classroomId) params.append('classroom_id', classroomId)
            if (userId) params.append('user_id', userId)

            const res = await fetch(`${getApiBaseUrl()}/api/progress/topic-mastery?${params}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                const data = await res.json()
                setStats(data.stats || {
                    total_topics: data.topics?.length || 0,
                    topics_started: data.topics?.filter((t: TopicScore) => t.total_attempts > 0).length || 0,
                    topics_mastered: data.topics?.filter((t: TopicScore) => t.mastery_level >= 80).length || 0,
                    average_mastery: data.topics?.reduce((sum: number, t: TopicScore) => sum + t.mastery_level, 0) / (data.topics?.length || 1) || 0,
                    total_study_hours: 0,
                    current_streak: 0
                })
                setChapterProgress(data.chapters || [])
                setTopicScores(data.topics || [])
            } else {
                // Use demo data if no endpoint available
                setDemoData()
            }
        } catch (e) {
            console.error('Fetch progress error:', e)
            setDemoData()
        } finally {
            setLoading(false)
        }
    }

    const setDemoData = () => {
        // Demo data for visualization
        setStats({
            total_topics: 24,
            topics_started: 18,
            topics_mastered: 8,
            average_mastery: 62,
            total_study_hours: 45.5,
            current_streak: 7
        })
        setChapterProgress([
            { chapter_id: '1', name: 'Algebra', color: '#3B82F6', topics_count: 6, topics_mastered: 4, average_mastery: 78 },
            { chapter_id: '2', name: 'Geometry', color: '#10B981', topics_count: 5, topics_mastered: 2, average_mastery: 55 },
            { chapter_id: '3', name: 'Calculus', color: '#F59E0B', topics_count: 8, topics_mastered: 2, average_mastery: 48 },
            { chapter_id: '4', name: 'Statistics', color: '#8B5CF6', topics_count: 5, topics_mastered: 0, average_mastery: 32 }
        ])
        setTopicScores([
            { topic_id: '1', topic_name: 'Linear Equations', chapter_name: 'Algebra', chapter_color: '#3B82F6', mastery_level: 92, quiz_score: 88, interview_score: 95, total_attempts: 5, last_activity: '2h ago' },
            { topic_id: '2', topic_name: 'Quadratic Functions', chapter_name: 'Algebra', chapter_color: '#3B82F6', mastery_level: 85, quiz_score: 80, interview_score: 90, total_attempts: 4, last_activity: '1d ago' },
            { topic_id: '3', topic_name: 'Polynomials', chapter_name: 'Algebra', chapter_color: '#3B82F6', mastery_level: 65, quiz_score: 70, interview_score: 60, total_attempts: 3, last_activity: '2d ago' },
            { topic_id: '4', topic_name: 'Triangles', chapter_name: 'Geometry', chapter_color: '#10B981', mastery_level: 72, quiz_score: 75, interview_score: 68, total_attempts: 4, last_activity: '3d ago' },
            { topic_id: '5', topic_name: 'Circles', chapter_name: 'Geometry', chapter_color: '#10B981', mastery_level: 45, quiz_score: 50, interview_score: 40, total_attempts: 2, last_activity: '5d ago' },
            { topic_id: '6', topic_name: 'Derivatives', chapter_name: 'Calculus', chapter_color: '#F59E0B', mastery_level: 58, quiz_score: 60, interview_score: 55, total_attempts: 3, last_activity: '1d ago' },
            { topic_id: '7', topic_name: 'Integrals', chapter_name: 'Calculus', chapter_color: '#F59E0B', mastery_level: 35, quiz_score: 40, interview_score: 30, total_attempts: 2, last_activity: '4d ago' },
            { topic_id: '8', topic_name: 'Probability', chapter_name: 'Statistics', chapter_color: '#8B5CF6', mastery_level: 28, quiz_score: 30, interview_score: 25, total_attempts: 1, last_activity: '1w ago' }
        ])
    }

    // ========================================================================
    // Filter topics by chapter
    // ========================================================================

    const filteredTopics = selectedChapter
        ? topicScores.filter(t => t.chapter_name === selectedChapter)
        : topicScores

    // ========================================================================
    // Render
    // ========================================================================

    if (loading) {
        return (
            <div className="p-8 text-center">
                <ArrowPathIcon className="w-8 h-8 text-indigo-500 animate-spin mx-auto mb-2" />
                <p className="text-gray-500">Loading progress...</p>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Stats Cards */}
            <div className={`grid gap-4 ${compact ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-4'}`}>
                <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl p-4 text-white">
                    <div className="flex items-center gap-2 mb-2">
                        <TrophyIcon className="w-5 h-5 opacity-80" />
                        <span className="text-sm opacity-80">Mastered</span>
                    </div>
                    <p className="text-2xl font-bold">{stats?.topics_mastered || 0}</p>
                    <p className="text-xs opacity-70">of {stats?.total_topics || 0} topics</p>
                </div>

                <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl p-4 text-white">
                    <div className="flex items-center gap-2 mb-2">
                        <ArrowTrendingUpIcon className="w-5 h-5 opacity-80" />
                        <span className="text-sm opacity-80">Avg Mastery</span>
                    </div>
                    <p className="text-2xl font-bold">{Math.round(stats?.average_mastery || 0)}%</p>
                    <p className="text-xs opacity-70">{getMasteryLabel(stats?.average_mastery || 0)}</p>
                </div>

                <div className="bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl p-4 text-white">
                    <div className="flex items-center gap-2 mb-2">
                        <FireIcon className="w-5 h-5 opacity-80" />
                        <span className="text-sm opacity-80">Streak</span>
                    </div>
                    <p className="text-2xl font-bold">{stats?.current_streak || 0}</p>
                    <p className="text-xs opacity-70">days in a row</p>
                </div>

                <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl p-4 text-white">
                    <div className="flex items-center gap-2 mb-2">
                        <ClockIcon className="w-5 h-5 opacity-80" />
                        <span className="text-sm opacity-80">Study Time</span>
                    </div>
                    <p className="text-2xl font-bold">{stats?.total_study_hours?.toFixed(1) || 0}h</p>
                    <p className="text-xs opacity-70">total hours</p>
                </div>
            </div>

            {/* Chapter Progress Bars */}
            <div className="bg-white rounded-xl border p-4">
                <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <ChartBarIcon className="w-5 h-5 text-gray-500" />
                    Chapter Progress
                </h3>
                <div className="space-y-4">
                    {chapterProgress.map((chapter) => (
                        <div
                            key={chapter.chapter_id}
                            className={`cursor-pointer transition-all ${selectedChapter === chapter.name ? 'ring-2 ring-indigo-300 rounded-lg p-2 -m-2' : ''
                                }`}
                            onClick={() => setSelectedChapter(
                                selectedChapter === chapter.name ? null : chapter.name
                            )}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-2">
                                    <div
                                        className="w-3 h-3 rounded-full"
                                        style={{ backgroundColor: chapter.color }}
                                    />
                                    <span className="text-sm font-medium text-gray-700">{chapter.name}</span>
                                </div>
                                <div className="flex items-center gap-3 text-xs text-gray-500">
                                    <span>{chapter.topics_mastered}/{chapter.topics_count} mastered</span>
                                    <span className="font-medium">{Math.round(chapter.average_mastery)}%</span>
                                </div>
                            </div>
                            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full transition-all duration-500"
                                    style={{
                                        width: `${chapter.average_mastery}%`,
                                        backgroundColor: chapter.color
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Topic Details */}
            <div className="bg-white rounded-xl border p-4">
                <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <AcademicCapIcon className="w-5 h-5 text-gray-500" />
                    Topic Mastery
                    {selectedChapter && (
                        <span className="text-sm font-normal text-gray-400">
                            — {selectedChapter}
                            <button
                                onClick={() => setSelectedChapter(null)}
                                className="ml-2 text-indigo-500 hover:underline"
                            >
                                Show all
                            </button>
                        </span>
                    )}
                </h3>

                <div className="space-y-3">
                    {filteredTopics.map((topic) => (
                        <div
                            key={topic.topic_id}
                            className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                        >
                            {/* Color indicator */}
                            <div
                                className="w-1 h-10 rounded-full flex-shrink-0"
                                style={{ backgroundColor: topic.chapter_color }}
                            />

                            {/* Topic info */}
                            <div className="flex-1 min-w-0">
                                <p className="font-medium text-gray-800 truncate">{topic.topic_name}</p>
                                <p className="text-xs text-gray-500">{topic.chapter_name} • {topic.last_activity}</p>
                            </div>

                            {/* Scores */}
                            <div className="flex items-center gap-3 text-xs">
                                <div className="text-center">
                                    <p className="text-gray-400">Quiz</p>
                                    <p className="font-medium">{topic.quiz_score}%</p>
                                </div>
                                <div className="text-center">
                                    <p className="text-gray-400">Interview</p>
                                    <p className="font-medium">{topic.interview_score}%</p>
                                </div>
                            </div>

                            {/* Mastery badge */}
                            <div className={`px-2 py-1 rounded-lg text-xs font-medium ${getMasteryColor(topic.mastery_level)}`}>
                                {topic.mastery_level}%
                            </div>

                            {/* Mastery bar */}
                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden flex-shrink-0">
                                <div
                                    className={`h-full rounded-full transition-all duration-500 ${getMasteryBarColor(topic.mastery_level)}`}
                                    style={{ width: `${topic.mastery_level}%` }}
                                />
                            </div>

                            {/* Mastered check */}
                            {topic.mastery_level >= 80 && (
                                <CheckCircleIcon className="w-5 h-5 text-green-500 flex-shrink-0" />
                            )}
                        </div>
                    ))}

                    {filteredTopics.length === 0 && (
                        <div className="text-center py-8 text-gray-400">
                            <AcademicCapIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                            <p>No topics found</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Mastery Distribution */}
            {!compact && (
                <div className="bg-white rounded-xl border p-4">
                    <h3 className="font-semibold text-gray-900 mb-4">Mastery Distribution</h3>
                    <div className="flex items-end gap-1 h-32">
                        {[0, 20, 40, 60, 80].map((threshold) => {
                            const count = topicScores.filter(t =>
                                t.mastery_level >= threshold && t.mastery_level < threshold + 20
                            ).length
                            const maxCount = Math.max(...[0, 20, 40, 60, 80].map(th =>
                                topicScores.filter(t => t.mastery_level >= th && t.mastery_level < th + 20).length
                            ), 1)
                            const height = (count / maxCount) * 100

                            return (
                                <div key={threshold} className="flex-1 flex flex-col items-center gap-1">
                                    <div
                                        className={`w-full rounded-t transition-all duration-500 ${getMasteryBarColor(threshold + 10)}`}
                                        style={{ height: `${Math.max(height, 5)}%` }}
                                    />
                                    <span className="text-xs text-gray-500">{threshold}-{threshold + 19}%</span>
                                    <span className="text-xs font-medium">{count}</span>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}
