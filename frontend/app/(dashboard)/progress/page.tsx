'use client'

import { useSession } from 'next-auth/react'
import {
    ChartBarIcon,
    ArrowTrendingUpIcon,
    ArrowTrendingDownIcon,
    AcademicCapIcon,
    FireIcon
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

const mockProgress: TopicProgress[] = [
    { topic: 'Cell Structure', subject: 'Biology', confidence: 85, isWeak: false, timesStudied: 8, lastStudied: '2 hours ago' },
    { topic: 'Photosynthesis', subject: 'Biology', confidence: 42, isWeak: true, timesStudied: 3, lastStudied: '1 day ago' },
    { topic: 'Quadratic Equations', subject: 'Math', confidence: 38, isWeak: true, timesStudied: 5, lastStudied: '3 days ago' },
    { topic: 'Newton\'s Laws', subject: 'Physics', confidence: 72, isWeak: false, timesStudied: 6, lastStudied: '1 day ago' },
    { topic: 'World War II', subject: 'History', confidence: 35, isWeak: true, timesStudied: 2, lastStudied: '5 days ago' },
    { topic: 'Chemical Bonding', subject: 'Chemistry', confidence: 68, isWeak: false, timesStudied: 4, lastStudied: '2 days ago' },
]

export default function ProgressPage() {
    const { data: session } = useSession()

    const subjects = Array.from(new Set(mockProgress.map(p => p.subject)))
    const weakTopics = mockProgress.filter(p => p.isWeak)
    const strongTopics = mockProgress.filter(p => !p.isWeak)
    const avgConfidence = Math.round(mockProgress.reduce((sum, p) => sum + p.confidence, 0) / mockProgress.length)

    return (
        <div className="space-y-6">
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
                    value={strongTopics.length.toString()}
                    icon={<AcademicCapIcon className="w-5 h-5" />}
                    trend="up"
                />
                <StatCard
                    label="Needs Attention"
                    value={weakTopics.length.toString()}
                    icon={<ArrowTrendingDownIcon className="w-5 h-5" />}
                    trend="down"
                />
                <StatCard
                    label="Study Streak"
                    value="7 days"
                    icon={<FireIcon className="w-5 h-5" />}
                    trend="up"
                />
            </div>

            {/* Subject Breakdown */}
            <div className="card">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Progress by Subject</h2>
                <div className="space-y-4">
                    {subjects.map((subject) => {
                        const subjectTopics = mockProgress.filter(p => p.subject === subject)
                        const avg = Math.round(subjectTopics.reduce((sum, p) => sum + p.confidence, 0) / subjectTopics.length)

                        return (
                            <div key={subject}>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="font-medium text-gray-700">{subject}</span>
                                    <span className="text-sm text-gray-500">{avg}%</span>
                                </div>
                                <div className="progress-bar">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: `${avg}%` }}
                                    />
                                </div>
                            </div>
                        )
                    })}
                </div>
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
                        {weakTopics.map((topic, idx) => (
                            <TopicRow key={idx} {...topic} />
                        ))}
                    </div>
                </div>

                {/* Strong Topics */}
                <div className="card border-l-4 border-success-500">
                    <div className="flex items-center gap-2 mb-4">
                        <ArrowTrendingUpIcon className="w-5 h-5 text-success-600" />
                        <h2 className="text-lg font-bold text-gray-900">Strong Topics</h2>
                    </div>
                    <div className="space-y-3">
                        {strongTopics.map((topic, idx) => (
                            <TopicRow key={idx} {...topic} />
                        ))}
                    </div>
                </div>
            </div>

            {/* All Topics Table */}
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
                        {mockProgress.map((topic, idx) => (
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
