'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import Link from 'next/link'
import {
    ClipboardDocumentListIcon,
    PlayIcon,
    CheckCircleIcon,
    ClockIcon,
    TrophyIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Assessment {
    id: string
    title: string
    topic: string
    subject: string
    difficulty: 'easy' | 'medium' | 'hard'
    num_questions: number
    time_limit_minutes: number
    completed?: boolean
    score?: number
}

const mockAssessments: Assessment[] = [
    {
        id: '1',
        title: 'Cell Biology Basics',
        topic: 'Cell Structure',
        subject: 'Biology',
        difficulty: 'medium',
        num_questions: 10,
        time_limit_minutes: 15,
        completed: true,
        score: 85,
    },
    {
        id: '2',
        title: 'Quadratic Equations',
        topic: 'Algebra',
        subject: 'Math',
        difficulty: 'hard',
        num_questions: 12,
        time_limit_minutes: 20,
    },
    {
        id: '3',
        title: 'World War II Overview',
        topic: 'Modern History',
        subject: 'History',
        difficulty: 'easy',
        num_questions: 8,
        time_limit_minutes: 10,
        completed: true,
        score: 92,
    },
    {
        id: '4',
        title: 'Chemical Bonding',
        topic: 'Chemistry Basics',
        subject: 'Chemistry',
        difficulty: 'medium',
        num_questions: 15,
        time_limit_minutes: 25,
    },
]

export default function AssessmentsPage() {
    const { data: session } = useSession()
    const [assessments, setAssessments] = useState<Assessment[]>(mockAssessments)
    const [filter, setFilter] = useState<'all' | 'pending' | 'completed'>('all')

    const filteredAssessments = assessments.filter((a) => {
        if (filter === 'pending') return !a.completed
        if (filter === 'completed') return a.completed
        return true
    })

    const difficultyColors = {
        easy: 'bg-green-100 text-green-700',
        medium: 'bg-yellow-100 text-yellow-700',
        hard: 'bg-red-100 text-red-700',
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">📝 Assessments</h1>
                    <p className="text-gray-500">Complete quizzes assigned by your teachers</p>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <StatCard
                    label="Total Completed"
                    value={assessments.filter(a => a.completed).length.toString()}
                    icon={<CheckCircleIcon className="w-5 h-5" />}
                    color="green"
                />
                <StatCard
                    label="Average Score"
                    value={`${Math.round(
                        assessments.filter(a => a.score).reduce((sum, a) => sum + (a.score || 0), 0) /
                        assessments.filter(a => a.score).length || 0
                    )}%`}
                    icon={<TrophyIcon className="w-5 h-5" />}
                    color="blue"
                />
                <StatCard
                    label="Pending"
                    value={assessments.filter(a => !a.completed).length.toString()}
                    icon={<ClockIcon className="w-5 h-5" />}
                    color="orange"
                />
                <StatCard
                    label="This Week"
                    value="3"
                    icon={<PlayIcon className="w-5 h-5" />}
                    color="purple"
                />
            </div>

            {/* Filter Tabs */}
            <div className="flex gap-2 border-b border-gray-200">
                {(['all', 'pending', 'completed'] as const).map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setFilter(tab)}
                        className={clsx(
                            'px-4 py-2 font-medium text-sm border-b-2 -mb-px transition-colors capitalize',
                            filter === tab
                                ? 'border-primary-600 text-primary-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700'
                        )}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            {/* Assessments List */}
            <div className="grid md:grid-cols-2 gap-4">
                {filteredAssessments.map((assessment) => (
                    <div key={assessment.id} className="card-hover">
                        <div className="flex items-start justify-between mb-3">
                            <div>
                                <h3 className="font-bold text-gray-900">{assessment.title}</h3>
                                <p className="text-sm text-gray-500">{assessment.topic}</p>
                            </div>
                            <span className={clsx(
                                'px-2 py-1 rounded-full text-xs font-medium capitalize',
                                difficultyColors[assessment.difficulty]
                            )}>
                                {assessment.difficulty}
                            </span>
                        </div>

                        <div className="flex items-center gap-4 text-sm text-gray-500 mb-4">
                            <span className="flex items-center gap-1">
                                <ClipboardDocumentListIcon className="w-4 h-4" />
                                {assessment.num_questions} questions
                            </span>
                            <span className="flex items-center gap-1">
                                <ClockIcon className="w-4 h-4" />
                                {assessment.time_limit_minutes} min
                            </span>
                        </div>

                        {assessment.completed ? (
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <CheckCircleIcon className="w-5 h-5 text-green-500" />
                                    <span className="text-green-600 font-medium">Completed</span>
                                </div>
                                <span className="text-2xl font-bold text-gray-900">
                                    {assessment.score}%
                                </span>
                            </div>
                        ) : (
                            <Link
                                href={`/assessments/${assessment.id}`}
                                className="btn-primary w-full flex items-center justify-center gap-2"
                            >
                                <PlayIcon className="w-5 h-5" />
                                Start Quiz
                            </Link>
                        )}
                    </div>
                ))}
            </div>

            {filteredAssessments.length === 0 && (
                <div className="text-center py-12">
                    <ClipboardDocumentListIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No assessments found</p>
                </div>
            )}
        </div>
    )
}

function StatCard({
    label,
    value,
    icon,
    color
}: {
    label: string
    value: string
    icon: React.ReactNode
    color: 'green' | 'blue' | 'orange' | 'purple'
}) {
    const colors = {
        green: 'bg-green-100 text-green-600',
        blue: 'bg-blue-100 text-blue-600',
        orange: 'bg-orange-100 text-orange-600',
        purple: 'bg-purple-100 text-purple-600',
    }

    return (
        <div className="card flex items-center gap-4">
            <div className={clsx('p-3 rounded-lg', colors[color])}>
                {icon}
            </div>
            <div>
                <p className="text-2xl font-bold text-gray-900">{value}</p>
                <p className="text-sm text-gray-500">{label}</p>
            </div>
        </div>
    )
}
