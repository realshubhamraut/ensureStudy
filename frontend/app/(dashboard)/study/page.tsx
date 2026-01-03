'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
    BookOpenIcon,
    ChevronRightIcon,
    CheckCircleIcon,
    ClockIcon
} from '@heroicons/react/24/outline'

interface Subject {
    id: string
    name: string
    code: string
    description: string
    color: string
    topic_count: number
}

export default function SubjectsPage() {
    const { data: session } = useSession()
    const [subjects, setSubjects] = useState<Subject[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchSubjects = async () => {
            try {
                const res = await fetch(`${getApiBaseUrl()}/api/curriculum/subjects`, {
                    headers: {
                        'Authorization': `Bearer ${(session as any)?.accessToken}`
                    }
                })
                if (res.ok) {
                    const data = await res.json()
                    setSubjects(data.subjects)
                }
            } catch (error) {
                console.error('Failed to fetch subjects:', error)
            } finally {
                setLoading(false)
            }
        }

        if (session) {
            fetchSubjects()
        }
    }, [session])

    // Placeholder subjects if none exist
    const placeholderSubjects = [
        { id: '1', name: 'Physics', code: 'PHY', color: '#3B82F6', description: 'Mechanics, Thermodynamics, Optics', topic_count: 12 },
        { id: '2', name: 'Chemistry', code: 'CHE', color: '#10B981', description: 'Organic, Inorganic, Physical', topic_count: 15 },
        { id: '3', name: 'Mathematics', code: 'MAT', color: '#8B5CF6', description: 'Calculus, Algebra, Geometry', topic_count: 18 },
        { id: '4', name: 'Biology', code: 'BIO', color: '#F59E0B', description: 'Botany, Zoology, Human Physiology', topic_count: 14 },
    ]

    const displaySubjects = subjects.length > 0 ? subjects : placeholderSubjects

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
            <div>
                <h1 className="text-2xl font-bold text-gray-900">My Subjects</h1>
                <p className="text-gray-600">Select a subject to start learning</p>
            </div>

            {/* Subjects Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {displaySubjects.map((subject) => (
                    <Link
                        key={subject.id}
                        href={`/study/${subject.id}`}
                        className="card-hover group"
                    >
                        <div className="flex items-start gap-4">
                            <div
                                className="w-14 h-14 rounded-xl flex items-center justify-center text-white font-bold text-lg"
                                style={{ backgroundColor: subject.color }}
                            >
                                {subject.code}
                            </div>
                            <div className="flex-1">
                                <h3 className="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                                    {subject.name}
                                </h3>
                                <p className="text-sm text-gray-500 mt-1">{subject.description}</p>
                                <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
                                    <span className="flex items-center gap-1">
                                        <BookOpenIcon className="w-4 h-4" />
                                        {subject.topic_count} Topics
                                    </span>
                                </div>
                            </div>
                            <ChevronRightIcon className="w-5 h-5 text-gray-400 group-hover:text-primary-600 transition-colors" />
                        </div>
                    </Link>
                ))}
            </div>

            {/* Note */}
            <div className="card bg-blue-50 border-blue-200">
                <p className="text-blue-700 text-sm">
                    <strong>Tip:</strong> Complete topic assessments to track your progress and unlock personalized
                    recommendations. Each topic has multiple subtopics with MCQ assessments.
                </p>
            </div>
        </div>
    )
}
