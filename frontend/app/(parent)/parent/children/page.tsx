'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
    AcademicCapIcon,
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

export default function ChildrenPage() {
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
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">👨‍👩‍👧‍👦 Your Children</h1>
                    <p className="text-gray-600">Manage and view your linked children's accounts</p>
                </div>
                <Link
                    href="/parent/settings"
                    className="btn-primary bg-orange-600 hover:bg-orange-700 flex items-center gap-2"
                >
                    <PlusIcon className="w-4 h-4" />
                    Link Another Child
                </Link>
            </div>

            {linkedChildren.length === 0 ? (
                <div className="card text-center py-16 border-2 border-dashed border-gray-300">
                    <AcademicCapIcon className="w-20 h-20 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">No Children Linked</h3>
                    <p className="text-gray-500 max-w-md mx-auto mb-6">
                        Link your child's account to start monitoring their learning journey.
                    </p>
                    <Link
                        href="/parent/settings"
                        className="inline-flex items-center gap-2 btn-primary bg-orange-600 hover:bg-orange-700"
                    >
                        <PlusIcon className="w-5 h-5" />
                        Link Your First Child
                    </Link>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {linkedChildren.map((child) => (
                        <div key={child.id} className="card hover:shadow-lg transition-shadow group">
                            <div className="flex items-center gap-4 mb-4">
                                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-orange-400 to-amber-500 flex items-center justify-center flex-shrink-0 shadow-lg">
                                    <span className="text-white font-bold text-2xl">
                                        {child.name?.[0]?.toUpperCase() || 'S'}
                                    </span>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <h3 className="font-semibold text-gray-900 text-lg truncate">{child.name}</h3>
                                    <p className="text-sm text-gray-500 truncate">{child.email}</p>
                                </div>
                            </div>

                            <div className="bg-gray-50 rounded-lg p-3 mb-4">
                                <div className="flex justify-between text-sm">
                                    <span className="text-gray-500">Relationship</span>
                                    <span className="text-gray-900 capitalize font-medium">{child.relationship_type}</span>
                                </div>
                                <div className="flex justify-between text-sm mt-1">
                                    <span className="text-gray-500">Linked On</span>
                                    <span className="text-gray-900">
                                        {child.linked_at ? new Date(child.linked_at).toLocaleDateString() : '--'}
                                    </span>
                                </div>
                            </div>

                            <Link
                                href={`/parent/children/${child.student_id}`}
                                className="flex items-center justify-center gap-2 w-full py-3 bg-orange-50 hover:bg-orange-100 text-orange-600 rounded-lg transition-colors font-medium group-hover:bg-orange-100"
                            >
                                View Progress
                                <ArrowRightIcon className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                            </Link>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
