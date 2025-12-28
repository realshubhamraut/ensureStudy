'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
    AcademicCapIcon,
    ChartBarIcon,
    TrophyIcon,
    LinkIcon,
    PlusIcon,
    ArrowRightIcon
} from '@heroicons/react/24/outline'

interface LinkedChild {
    id: string
    student_id: string
    name: string
    email: string
    relationship_type: string
    linked_at: string
}

export default function ParentDashboard() {
    const { data: session } = useSession()
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
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">
                        👋 Welcome, {session?.user?.name?.split(' ')[0] || 'Parent'}!
                    </h1>
                    <p className="text-gray-600">Monitor your children's learning progress</p>
                </div>
                <Link
                    href="/parent/settings"
                    className="btn-primary bg-orange-600 hover:bg-orange-700 flex items-center gap-2"
                >
                    <LinkIcon className="w-4 h-4" />
                    Link Child
                </Link>
            </div>

            {/* Stats Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="card bg-gradient-to-br from-orange-50 to-amber-50 border-orange-200">
                    <div className="flex items-center gap-3">
                        <div className="p-3 rounded-xl bg-orange-100">
                            <AcademicCapIcon className="w-6 h-6 text-orange-600" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-gray-900">{linkedChildren.length}</p>
                            <p className="text-sm text-gray-600">Children Linked</p>
                        </div>
                    </div>
                </div>
                <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200">
                    <div className="flex items-center gap-3">
                        <div className="p-3 rounded-xl bg-blue-100">
                            <ChartBarIcon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-gray-900">--</p>
                            <p className="text-sm text-gray-600">Avg. Progress</p>
                        </div>
                    </div>
                </div>
                <div className="card bg-gradient-to-br from-green-50 to-emerald-50 border-green-200">
                    <div className="flex items-center gap-3">
                        <div className="p-3 rounded-xl bg-green-100">
                            <TrophyIcon className="w-6 h-6 text-green-600" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-gray-900">--</p>
                            <p className="text-sm text-gray-600">Assessments Done</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Linked Children */}
            {linkedChildren.length === 0 ? (
                <div className="card text-center py-12 border-2 border-dashed border-gray-300">
                    <AcademicCapIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No Children Linked Yet</h3>
                    <p className="text-gray-500 max-w-md mx-auto mb-6">
                        Link your child's account to view their learning progress, assessments, and more.
                    </p>
                    <Link
                        href="/parent/settings"
                        className="inline-flex items-center gap-2 btn-primary bg-orange-600 hover:bg-orange-700"
                    >
                        <PlusIcon className="w-5 h-5" />
                        Link Your Child
                    </Link>
                </div>
            ) : (
                <div className="space-y-4">
                    <h2 className="text-lg font-semibold text-gray-900">Your Children</h2>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {linkedChildren.map((child) => (
                            <div key={child.id} className="card hover:shadow-lg transition-shadow">
                                <div className="flex items-start gap-4">
                                    <div className="w-14 h-14 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0">
                                        <span className="text-orange-600 font-bold text-xl">
                                            {child.name?.[0]?.toUpperCase() || 'S'}
                                        </span>
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <h3 className="font-semibold text-gray-900 text-lg">{child.name}</h3>
                                        <p className="text-sm text-gray-500">{child.email}</p>
                                        <p className="text-xs text-orange-600 mt-1 capitalize">{child.relationship_type}</p>
                                    </div>
                                </div>

                                <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-gray-100">
                                    <div className="text-center">
                                        <p className="text-xl font-bold text-gray-900">--</p>
                                        <p className="text-xs text-gray-500">Quizzes</p>
                                    </div>
                                    <div className="text-center">
                                        <p className="text-xl font-bold text-gray-900">--</p>
                                        <p className="text-xs text-gray-500">Avg Score</p>
                                    </div>
                                    <div className="text-center">
                                        <p className="text-xl font-bold text-gray-900">--</p>
                                        <p className="text-xs text-gray-500">Topics</p>
                                    </div>
                                </div>

                                <Link
                                    href={`/parent/children/${child.student_id}`}
                                    className="mt-4 flex items-center justify-center gap-2 w-full py-2 bg-gray-50 hover:bg-orange-50 text-gray-600 hover:text-orange-600 rounded-lg transition-colors text-sm font-medium"
                                >
                                    View Details
                                    <ArrowRightIcon className="w-4 h-4" />
                                </Link>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* How It Works */}
            <div className="card bg-gradient-to-br from-orange-50 to-amber-50 border-orange-200">
                <h3 className="font-semibold text-orange-900 mb-3">📋 How Parent Linking Works</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="flex items-start gap-3">
                        <span className="w-8 h-8 rounded-full bg-orange-200 text-orange-700 flex items-center justify-center font-bold text-sm flex-shrink-0">1</span>
                        <p className="text-sm text-orange-700">Your child logs into their ensureStudy account</p>
                    </div>
                    <div className="flex items-start gap-3">
                        <span className="w-8 h-8 rounded-full bg-orange-200 text-orange-700 flex items-center justify-center font-bold text-sm flex-shrink-0">2</span>
                        <p className="text-sm text-orange-700">They go to Settings and find their unique link code</p>
                    </div>
                    <div className="flex items-start gap-3">
                        <span className="w-8 h-8 rounded-full bg-orange-200 text-orange-700 flex items-center justify-center font-bold text-sm flex-shrink-0">3</span>
                        <p className="text-sm text-orange-700">You enter the code in your Settings page</p>
                    </div>
                    <div className="flex items-start gap-3">
                        <span className="w-8 h-8 rounded-full bg-orange-200 text-orange-700 flex items-center justify-center font-bold text-sm flex-shrink-0">4</span>
                        <p className="text-sm text-orange-700">Instantly access their progress and reports!</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
