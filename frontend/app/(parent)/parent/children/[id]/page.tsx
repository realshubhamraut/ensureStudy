'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import {
    ArrowLeftIcon,
    ChartBarIcon,
    TrophyIcon,
    ClockIcon,
    BookOpenIcon,
    AcademicCapIcon
} from '@heroicons/react/24/outline'

export default function ChildDetailPage() {
    const params = useParams()
    const studentId = params?.id as string
    const [loading, setLoading] = useState(true)
    const [child, setChild] = useState<any>(null)

    useEffect(() => {
        // Simulate loading child data
        setTimeout(() => {
            setChild({
                id: studentId,
                name: 'Student',
                grade: '10',
                subjects: ['Mathematics', 'Physics', 'Chemistry']
            })
            setLoading(false)
        }, 500)
    }, [studentId])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Back Button */}
            <Link
                href="/parent/children"
                className="inline-flex items-center gap-2 text-gray-600 hover:text-orange-600 transition-colors"
            >
                <ArrowLeftIcon className="w-4 h-4" />
                Back to Children
            </Link>

            {/* Header */}
            <div className="flex items-center gap-4">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-orange-400 to-amber-500 flex items-center justify-center shadow-lg">
                    <span className="text-white font-bold text-3xl">
                        {child?.name?.[0]?.toUpperCase() || 'S'}
                    </span>
                </div>
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">{child?.name || 'Student'}</h1>
                    <p className="text-gray-600">Class {child?.grade || '--'}</p>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200">
                    <ChartBarIcon className="w-8 h-8 text-blue-600 mb-2" />
                    <p className="text-2xl font-bold text-gray-900">--</p>
                    <p className="text-sm text-gray-600">Quizzes Taken</p>
                </div>
                <div className="card bg-gradient-to-br from-green-50 to-emerald-50 border-green-200">
                    <TrophyIcon className="w-8 h-8 text-green-600 mb-2" />
                    <p className="text-2xl font-bold text-gray-900">--</p>
                    <p className="text-sm text-gray-600">Average Score</p>
                </div>
                <div className="card bg-gradient-to-br from-purple-50 to-violet-50 border-purple-200">
                    <ClockIcon className="w-8 h-8 text-purple-600 mb-2" />
                    <p className="text-2xl font-bold text-gray-900">--</p>
                    <p className="text-sm text-gray-600">Study Hours</p>
                </div>
                <div className="card bg-gradient-to-br from-orange-50 to-amber-50 border-orange-200">
                    <BookOpenIcon className="w-8 h-8 text-orange-600 mb-2" />
                    <p className="text-2xl font-bold text-gray-900">--</p>
                    <p className="text-sm text-gray-600">Topics Mastered</p>
                </div>
            </div>

            {/* Subjects */}
            <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Enrolled Subjects</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {(child?.subjects || []).map((subject: string, index: number) => (
                        <div key={index} className="p-4 bg-gray-50 rounded-lg">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-lg bg-orange-100 flex items-center justify-center">
                                    <AcademicCapIcon className="w-5 h-5 text-orange-600" />
                                </div>
                                <div>
                                    <p className="font-medium text-gray-900">{subject}</p>
                                    <p className="text-sm text-gray-500">-- topics</p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Recent Activity */}
            <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h2>
                <div className="text-center py-8 text-gray-500">
                    <ClockIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                    <p>No recent activity to display</p>
                </div>
            </div>
        </div>
    )
}
