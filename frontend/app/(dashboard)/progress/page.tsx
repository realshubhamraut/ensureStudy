'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    ChartBarIcon,
    ArrowTrendingUpIcon,
    ArrowTrendingDownIcon,
    AcademicCapIcon,
    FireIcon,
    ArrowPathIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface TopicProgress {
    topic: string
    subject: string
    confidence: number
    isWeak: boolean
    timesStudied: number
    lastStudied: string
}

interface ProgressOverview {
    avgConfidence: number
    topicsMastered: number
    topicsNeedAttention: number
    studyStreak: number
    totalTopics: number
    subjects: {
        subject: string
        avgConfidence: number
        topicCount: number
    }[]
}

// Helper to get API base URL
function getApiBaseUrl() {
    if (typeof window !== 'undefined') {
        return process.env.NEXT_PUBLIC_API_URL || `http://${window.location.hostname}:5000`
    }
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
}

export default function ProgressPage() {
    const { data: session } = useSession()
    const [topics, setTopics] = useState<TopicProgress[]>([])
    const [overview, setOverview] = useState<ProgressOverview | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const apiUrl = getApiBaseUrl()

    useEffect(() => {
        if (!session?.accessToken) return

        async function fetchProgress() {
            try {
                setLoading(true)
                setError(null)

                // Fetch both overview and topics list in parallel
                const [overviewRes, topicsRes] = await Promise.all([
                    fetch(`${apiUrl}/api/progress/overview`, {
                        headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                    }),
                    fetch(`${apiUrl}/api/progress/topics-list`, {
                        headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                    })
                ])

                if (overviewRes.ok) {
                    const overviewData = await overviewRes.json()
                    setOverview(overviewData)
                }

                if (topicsRes.ok) {
                    const topicsData = await topicsRes.json()
                    setTopics(topicsData)
                }

                // If no data found, use empty arrays
                if (!overviewRes.ok && !topicsRes.ok) {
                    setError('No progress data found. Start learning to track your progress!')
                }
            } catch (err) {
                console.error('Error fetching progress:', err)
                setError('Failed to load progress data')
            } finally {
                setLoading(false)
            }
        }

        fetchProgress()
    }, [session?.accessToken, apiUrl])

    // Calculate derived data
    const subjects = overview?.subjects || []
    const weakTopics = topics.filter(p => p.isWeak)
    const strongTopics = topics.filter(p => !p.isWeak)
    const avgConfidence = overview?.avgConfidence || 0
    const studyStreak = overview?.studyStreak || 0
    const topicsMastered = overview?.topicsMastered || strongTopics.length
    const topicsNeedAttention = overview?.topicsNeedAttention || weakTopics.length

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <ArrowPathIcon className="w-8 h-8 animate-spin text-primary-600" />
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Error Banner */}
            {error && topics.length === 0 && (
                <div className="card bg-blue-50 border-l-4 border-blue-500 text-blue-700">
                    <p>{error}</p>
                    <p className="text-sm mt-2">Ask the AI Tutor some questions or take an assessment to start tracking your progress.</p>
                </div>
            )}

            {/* Stats Overview */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <StatCard
                    label="Overall Confidence"
                    value={`${avgConfidence}%`}
                    icon={<ChartBarIcon className="w-5 h-5" />}
                    trend={avgConfidence > 60 ? 'up' : 'down'}
                />
                <StatCard
                    label="Topics Mastered"
                    value={topicsMastered.toString()}
                    icon={<AcademicCapIcon className="w-5 h-5" />}
                    trend="up"
                />
                <StatCard
                    label="Needs Attention"
                    value={topicsNeedAttention.toString()}
                    icon={<ArrowTrendingDownIcon className="w-5 h-5" />}
                    trend="down"
                />
                <StatCard
                    label="Study Streak"
                    value={`${studyStreak} days`}
                    icon={<FireIcon className="w-5 h-5" />}
                    trend="up"
                />
            </div>

            {/* Subject Breakdown */}
            <div className="card">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Progress by Subject</h2>
                {subjects.length > 0 ? (
                    <div className="space-y-4">
                        {subjects.map((subjectData) => (
                            <div key={subjectData.subject}>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="font-medium text-gray-700">{subjectData.subject}</span>
                                    <span className="text-sm text-gray-500">{subjectData.avgConfidence}%</span>
                                </div>
                                <div className="progress-bar">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: `${subjectData.avgConfidence}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-500 text-sm">No subjects tracked yet</p>
                )}
            </div>

            {/* Topics Grid */}
            <div className="grid md:grid-cols-2 gap-6">
                {/* Weak Topics */}
                <div className="card border-l-4 border-warning-500">
                    <div className="flex items-center gap-2 mb-4">
                        <ArrowTrendingDownIcon className="w-5 h-5 text-warning-600" />
                        <h2 className="text-lg font-bold text-gray-900">Topics to Improve</h2>
                    </div>
                    <div className="space-y-3">
                        {weakTopics.length > 0 ? (
                            weakTopics.map((topic, idx) => (
                                <TopicRow key={idx} {...topic} />
                            ))
                        ) : (
                            <p className="text-gray-500 text-sm">No weak topics - great job!</p>
                        )}
                    </div>
                </div>

                {/* Strong Topics */}
                <div className="card border-l-4 border-success-500">
                    <div className="flex items-center gap-2 mb-4">
                        <ArrowTrendingUpIcon className="w-5 h-5 text-success-600" />
                        <h2 className="text-lg font-bold text-gray-900">Strong Topics</h2>
                    </div>
                    <div className="space-y-3">
                        {strongTopics.length > 0 ? (
                            strongTopics.map((topic, idx) => (
                                <TopicRow key={idx} {...topic} />
                            ))
                        ) : (
                            <p className="text-gray-500 text-sm">Keep studying to build strong topics!</p>
                        )}
                    </div>
                </div>
            </div>

            {/* All Topics Table */}
            {topics.length > 0 && (
                <div className="card overflow-x-auto">
                    <h2 className="text-lg font-bold text-gray-900 mb-4">All Topics</h2>
                    <table className="w-full">
                        <thead>
                            <tr className="text-left text-sm text-gray-500 border-b border-gray-200">
                                <th className="pb-3 font-medium">Topic</th>
                                <th className="pb-3 font-medium">Subject</th>
                                <th className="pb-3 font-medium">Confidence</th>
                                <th className="pb-3 font-medium">Times Studied</th>
                                <th className="pb-3 font-medium">Last Studied</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                            {topics.map((topic, idx) => (
                                <tr key={idx} className="text-sm">
                                    <td className="py-3 font-medium text-gray-900">{topic.topic}</td>
                                    <td className="py-3 text-gray-600">{topic.subject}</td>
                                    <td className="py-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-20 h-2 bg-gray-200 rounded-full">
                                                <div
                                                    className={clsx(
                                                        'h-full rounded-full',
                                                        topic.confidence >= 70 ? 'bg-green-500' :
                                                            topic.confidence >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                                                    )}
                                                    style={{ width: `${topic.confidence}%` }}
                                                />
                                            </div>
                                            <span className="text-gray-600">{topic.confidence}%</span>
                                        </div>
                                    </td>
                                    <td className="py-3 text-gray-600">{topic.timesStudied}</td>
                                    <td className="py-3 text-gray-500">{topic.lastStudied}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}

function StatCard({
    label,
    value,
    icon,
    trend
}: {
    label: string
    value: string
    icon: React.ReactNode
    trend: 'up' | 'down'
}) {
    return (
        <div className="card flex items-center gap-4">
            <div className={clsx(
                'p-3 rounded-lg',
                trend === 'up' ? 'bg-green-100 text-green-600' : 'bg-orange-100 text-orange-600'
            )}>
                {icon}
            </div>
            <div>
                <p className="text-2xl font-bold text-gray-900">{value}</p>
                <p className="text-sm text-gray-500">{label}</p>
            </div>
        </div>
    )
}

function TopicRow({ topic, subject, confidence, lastStudied }: TopicProgress) {
    return (
        <div className="flex items-center justify-between">
            <div>
                <p className="font-medium text-gray-900">{topic}</p>
                <p className="text-sm text-gray-500">{subject}</p>
            </div>
            <div className="text-right">
                <p className={clsx(
                    'font-bold',
                    confidence >= 70 ? 'text-green-600' :
                        confidence >= 50 ? 'text-yellow-600' : 'text-red-600'
                )}>
                    {confidence}%
                </p>
                <p className="text-xs text-gray-400">{lastStudied}</p>
            </div>
        </div>
    )
}
