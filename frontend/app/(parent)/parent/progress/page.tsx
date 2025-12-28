'use client'

import { useState, useEffect } from 'react'
import {
    ChartBarIcon,
    TrophyIcon,
    AcademicCapIcon,
    ArrowTrendingUpIcon
} from '@heroicons/react/24/outline'

interface LinkedChild {
    id: string
    student_id: string
    name: string
    email: string
}

export default function ParentProgressPage() {
    const [loading, setLoading] = useState(true)
    const [linkedChildren, setLinkedChildren] = useState<LinkedChild[]>([])

    useEffect(() => {
        fetchLinkedChildren()
    }, [])

    const fetchLinkedChildren = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/students/linked-children', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setLinkedChildren(data.children || [])
            }
        } catch (error) {
            console.error('Failed to fetch linked children:', error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900">📊 Progress Overview</h1>
                <p className="text-gray-600">Track your children's learning progress</p>
            </div>

            {linkedChildren.length === 0 ? (
                <div className="card text-center py-12">
                    <ChartBarIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No Children Linked</h3>
                    <p className="text-gray-500">Link a child to view their progress.</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {linkedChildren.map((child) => (
                        <div key={child.id} className="card">
                            <div className="flex items-center gap-4 mb-6">
                                <div className="w-12 h-12 rounded-full bg-orange-100 flex items-center justify-center">
                                    <span className="text-orange-600 font-bold text-lg">
                                        {child.name?.[0]?.toUpperCase() || 'S'}
                                    </span>
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-900 text-lg">{child.name}</h3>
                                    <p className="text-sm text-gray-500">{child.email}</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                                <div className="p-4 bg-blue-50 rounded-lg">
                                    <ChartBarIcon className="w-6 h-6 text-blue-600 mb-2" />
                                    <p className="text-xl font-bold text-gray-900">--</p>
                                    <p className="text-sm text-gray-600">Quizzes</p>
                                </div>
                                <div className="p-4 bg-green-50 rounded-lg">
                                    <TrophyIcon className="w-6 h-6 text-green-600 mb-2" />
                                    <p className="text-xl font-bold text-gray-900">--</p>
                                    <p className="text-sm text-gray-600">Avg Score</p>
                                </div>
                                <div className="p-4 bg-purple-50 rounded-lg">
                                    <AcademicCapIcon className="w-6 h-6 text-purple-600 mb-2" />
                                    <p className="text-xl font-bold text-gray-900">--</p>
                                    <p className="text-sm text-gray-600">Topics</p>
                                </div>
                                <div className="p-4 bg-orange-50 rounded-lg">
                                    <ArrowTrendingUpIcon className="w-6 h-6 text-orange-600 mb-2" />
                                    <p className="text-xl font-bold text-gray-900">--</p>
                                    <p className="text-sm text-gray-600">Streak</p>
                                </div>
                            </div>

                            {/* Progress Bar */}
                            <div className="mt-4">
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-gray-600">Overall Progress</span>
                                    <span className="text-gray-900 font-medium">0%</span>
                                </div>
                                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                                    <div className="h-full bg-orange-500 rounded-full" style={{ width: '0%' }}></div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
