'use client'

import { getApiBaseUrl } from '@/utils/api'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import Link from 'next/link'
import {
    PlusIcon,
    ClipboardDocumentIcon,
    AcademicCapIcon,
    UsersIcon,
    ArrowPathIcon,
    XMarkIcon
} from '@heroicons/react/24/outline'

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    subject: string
    join_code: string
    is_active: boolean
    student_count: number
}

export default function TeacherClassroomsPage() {
    const { data: session } = useSession()
    const [classrooms, setClassrooms] = useState<Classroom[]>([])
    const [loading, setLoading] = useState(true)
    const [showCreateModal, setShowCreateModal] = useState(false)
    const [creating, setCreating] = useState(false)
    const [formData, setFormData] = useState({
        name: '',
        grade: '',
        section: '',
        subject: ''
    })

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || getApiBaseUrl()

    useEffect(() => {
        fetchClassrooms()
    }, [session?.accessToken])

    const fetchClassrooms = async () => {
        if (!session?.accessToken) return
        try {
            const res = await fetch(`${apiUrl}/api/classroom`, {
                headers: {
                    'Authorization': `Bearer ${session.accessToken}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setClassrooms(data.classrooms || [])
            }
        } catch (error) {
            console.error('Failed to fetch classrooms:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleCreate = async () => {
        if (!formData.name) {
            alert('Please enter a classroom name')
            return
        }

        setCreating(true)
        try {
            const res = await fetch(`${apiUrl}/api/classroom`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })

            if (res.ok) {
                const data = await res.json()
                setClassrooms([...classrooms, data.classroom])
                setShowCreateModal(false)
                setFormData({ name: '', grade: '', section: '', subject: '' })
                alert(`Classroom created! Share code: ${data.classroom.join_code}`)
            }
        } catch (error) {
            alert('Failed to create classroom')
        } finally {
            setCreating(false)
        }
    }

    const copyCode = (code: string) => {
        navigator.clipboard.writeText(code)
        alert('Code copied!')
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <ArrowPathIcon className="w-8 h-8 animate-spin text-purple-600" />
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">My Classrooms</h1>
                    <p className="text-gray-600">Create and manage your classrooms</p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="btn-primary flex items-center gap-2"
                >
                    <PlusIcon className="w-5 h-5" />
                    Create Classroom
                </button>
            </div>

            {/* Empty State */}
            {classrooms.length === 0 ? (
                <div className="text-center py-16 bg-gray-50 rounded-xl">
                    <AcademicCapIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No classrooms yet</h3>
                    <p className="text-gray-500 mb-4">Create your first classroom to get started</p>
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="btn-primary"
                    >
                        Create Classroom
                    </button>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {classrooms.map((classroom) => (
                        <Link
                            key={classroom.id}
                            href={`/teacher/classroom/${classroom.id}`}
                            className="card-hover group"
                        >
                            <div className="flex items-start justify-between mb-4">
                                <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500">
                                    <AcademicCapIcon className="w-6 h-6 text-white" />
                                </div>
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${classroom.is_active
                                    ? 'bg-green-100 text-green-700'
                                    : 'bg-gray-100 text-gray-700'
                                    }`}>
                                    {classroom.is_active ? 'Active' : 'Inactive'}
                                </span>
                            </div>

                            <h3 className="font-semibold text-gray-900 text-lg group-hover:text-purple-600 transition-colors">
                                {classroom.name}
                            </h3>
                            <p className="text-sm text-gray-500 mt-1">
                                {classroom.grade && `Grade ${classroom.grade}`}
                                {classroom.section && ` • ${classroom.section}`}
                                {classroom.subject && ` • ${classroom.subject}`}
                            </p>

                            {/* Join Code */}
                            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-gray-500 uppercase tracking-wide">Join Code</p>
                                        <code className="text-lg font-mono font-bold text-purple-600">
                                            {classroom.join_code}
                                        </code>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.preventDefault()
                                            e.stopPropagation()
                                            copyCode(classroom.join_code)
                                        }}
                                        className="p-2 text-gray-400 hover:text-purple-600 hover:bg-purple-50 rounded-lg"
                                    >
                                        <ClipboardDocumentIcon className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>

                            {/* Stats */}
                            <div className="mt-4 pt-4 border-t border-gray-100 flex items-center gap-4 text-sm text-gray-500">
                                <div className="flex items-center gap-1">
                                    <UsersIcon className="w-4 h-4" />
                                    {classroom.student_count || 0} students
                                </div>
                            </div>
                        </Link>
                    ))}
                </div>
            )}

            {/* Create Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold text-gray-900">Create Classroom</h2>
                            <button onClick={() => setShowCreateModal(false)} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Classroom Name *
                                </label>
                                <input
                                    type="text"
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    className="input-field"
                                    placeholder="e.g., Physics Class 10-A"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Grade</label>
                                    <select
                                        value={formData.grade}
                                        onChange={(e) => setFormData({ ...formData, grade: e.target.value })}
                                        className="input-field"
                                    >
                                        <option value="">Select</option>
                                        {['9', '10', '11', '12'].map(g => (
                                            <option key={g} value={g}>{g}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Section</label>
                                    <input
                                        type="text"
                                        value={formData.section}
                                        onChange={(e) => setFormData({ ...formData, section: e.target.value })}
                                        className="input-field"
                                        placeholder="A, B, Science..."
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
                                <input
                                    type="text"
                                    value={formData.subject}
                                    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                    className="input-field"
                                    placeholder="Physics, Mathematics..."
                                />
                            </div>
                        </div>

                        <div className="flex gap-3 mt-6">
                            <button
                                onClick={() => setShowCreateModal(false)}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleCreate}
                                disabled={creating}
                                className="flex-1 btn-primary flex items-center justify-center gap-2"
                            >
                                {creating ? (
                                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                ) : (
                                    <PlusIcon className="w-5 h-5" />
                                )}
                                {creating ? 'Creating...' : 'Create'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
