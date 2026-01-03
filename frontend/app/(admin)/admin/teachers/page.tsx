'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    MagnifyingGlassIcon,
    UserMinusIcon,
    PlusIcon,
    XMarkIcon,
    EnvelopeIcon,
    PhoneIcon,
    AcademicCapIcon
} from '@heroicons/react/24/outline'

interface Teacher {
    id: string
    email: string
    username: string
    first_name: string
    last_name: string
    phone?: string
    created_at: string
    is_active: boolean
    classrooms?: { id: string; name: string; student_count: number }[]
}

export default function TeachersPage() {
    const { data: session } = useSession()
    const [teachers, setTeachers] = useState<Teacher[]>([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const [showInviteModal, setShowInviteModal] = useState(false)
    const [selectedTeacher, setSelectedTeacher] = useState<Teacher | null>(null)
    const [accessToken, setAccessToken] = useState('')

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Fetch teachers
                const res = await fetch(`${getApiBaseUrl()}/api/admin/teachers`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (res.ok) {
                    const data = await res.json()
                    setTeachers(data.teachers)
                }

                // Fetch organization for access token
                const dashRes = await fetch(`${getApiBaseUrl()}/api/admin/dashboard`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (dashRes.ok) {
                    const dashData = await dashRes.json()
                    setAccessToken(dashData.access_token)
                }
            } catch (error) {
                console.error('Failed to fetch teachers:', error)
            } finally {
                setLoading(false)
            }
        }

        fetchData()
    }, [session])

    const fetchTeacherDetails = async (teacherId: string) => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/admin/teachers/${teacherId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setSelectedTeacher(data.teacher)
            } else {
                // If detail endpoint doesn't exist, use list data
                const teacher = teachers.find(t => t.id === teacherId)
                if (teacher) setSelectedTeacher(teacher)
            }
        } catch (error) {
            // Fallback to list data
            const teacher = teachers.find(t => t.id === teacherId)
            if (teacher) setSelectedTeacher(teacher)
        }
    }

    const removeTeacher = async (teacherId: string) => {
        if (!confirm('Are you sure you want to remove this teacher?')) return

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/admin/teachers/${teacherId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                setTeachers(teachers.filter(t => t.id !== teacherId))
                setSelectedTeacher(null)
            }
        } catch (error) {
            console.error('Failed to remove teacher:', error)
        }
    }

    const filteredTeachers = teachers.filter(t =>
        t.email.toLowerCase().includes(search.toLowerCase()) ||
        t.first_name?.toLowerCase().includes(search.toLowerCase()) ||
        t.last_name?.toLowerCase().includes(search.toLowerCase()) ||
        t.phone?.includes(search)
    )

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Teacher Detail Modal */}
            {selectedTeacher && (
                <div className="fixed inset-0 bg-black/50 z-[100] flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl p-6 w-full max-w-lg shadow-2xl">
                        <div className="flex items-start justify-between mb-6">
                            <div className="flex items-center gap-4">
                                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-400 to-violet-500 flex items-center justify-center">
                                    <span className="text-white font-bold text-2xl">
                                        {selectedTeacher.first_name?.[0] || selectedTeacher.username[0]}
                                    </span>
                                </div>
                                <div>
                                    <h2 className="text-xl font-bold text-gray-900">
                                        {selectedTeacher.first_name} {selectedTeacher.last_name}
                                    </h2>
                                    <p className="text-gray-500">@{selectedTeacher.username}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setSelectedTeacher(null)}
                                className="p-2 text-gray-400 hover:text-gray-600"
                            >
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        {/* Contact Info */}
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                                    <EnvelopeIcon className="w-4 h-4" />
                                    Email
                                </div>
                                <p className="font-medium text-gray-900">{selectedTeacher.email}</p>
                            </div>
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                                    <PhoneIcon className="w-4 h-4" />
                                    Phone
                                </div>
                                <p className="font-medium text-gray-900">
                                    {selectedTeacher.phone || 'Not provided'}
                                </p>
                            </div>
                        </div>

                        {/* Status & Date */}
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="p-4 bg-purple-50 rounded-lg">
                                <p className="text-xs text-purple-600 mb-1">Status</p>
                                <span className={`px-2 py-1 text-xs rounded-full ${selectedTeacher.is_active
                                        ? 'bg-green-100 text-green-700'
                                        : 'bg-red-100 text-red-700'
                                    }`}>
                                    {selectedTeacher.is_active ? 'Active' : 'Inactive'}
                                </span>
                            </div>
                            <div className="p-4 bg-blue-50 rounded-lg">
                                <p className="text-xs text-blue-600 mb-1">Joined</p>
                                <p className="font-semibold text-gray-900">
                                    {new Date(selectedTeacher.created_at).toLocaleDateString()}
                                </p>
                            </div>
                        </div>

                        {/* Classrooms */}
                        {selectedTeacher.classrooms && selectedTeacher.classrooms.length > 0 && (
                            <div className="mb-6">
                                <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                                    <AcademicCapIcon className="w-5 h-5 text-purple-600" />
                                    Assigned Classrooms
                                </h3>
                                <div className="space-y-2">
                                    {selectedTeacher.classrooms.map(c => (
                                        <div key={c.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                            <span className="font-medium text-gray-900">{c.name}</span>
                                            <span className="text-sm text-gray-500">{c.student_count} students</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Actions */}
                        <div className="flex gap-3 pt-4 border-t border-gray-200">
                            <button
                                onClick={() => setSelectedTeacher(null)}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                            >
                                Close
                            </button>
                            <button
                                onClick={() => removeTeacher(selectedTeacher.id)}
                                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                            >
                                Remove Teacher
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">👩‍🏫 Teachers</h1>
                    <p className="text-gray-600">{teachers.length} teachers in your organization</p>
                </div>
                <button
                    onClick={() => setShowInviteModal(true)}
                    className="btn-primary flex items-center gap-2"
                >
                    <PlusIcon className="w-5 h-5" />
                    Invite Teacher
                </button>
            </div>

            {/* Search */}
            <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                    type="text"
                    placeholder="Search by name, email, or phone..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="input-field pl-10"
                />
            </div>

            {/* Teachers List */}
            {filteredTeachers.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-xl">
                    <AcademicCapIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">No teachers yet</p>
                    <p className="text-sm text-gray-400 mt-1">
                        Share your access token with teachers to let them register
                    </p>
                    <button
                        onClick={() => setShowInviteModal(true)}
                        className="btn-primary mt-4"
                    >
                        Invite Teachers
                    </button>
                </div>
            ) : (
                <div className="space-y-3">
                    {filteredTeachers.map((teacher) => (
                        <div
                            key={teacher.id}
                            className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
                            onClick={() => fetchTeacherDetails(teacher.id)}
                        >
                            <div className="p-4">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-400 to-violet-500 flex items-center justify-center">
                                            <span className="text-white font-semibold">
                                                {teacher.first_name?.[0] || teacher.username[0]}
                                            </span>
                                        </div>
                                        <div>
                                            <div className="font-semibold text-gray-900">
                                                {teacher.first_name} {teacher.last_name}
                                            </div>
                                            <div className="text-sm text-gray-500 flex items-center gap-3">
                                                <span>{teacher.email}</span>
                                                {teacher.phone && (
                                                    <span className="flex items-center gap-1">
                                                        <PhoneIcon className="w-3 h-3" />
                                                        {teacher.phone}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <span className={`px-2 py-1 text-xs rounded-full ${teacher.is_active
                                                ? 'bg-green-100 text-green-700'
                                                : 'bg-red-100 text-red-700'
                                            }`}>
                                            {teacher.is_active ? 'Active' : 'Inactive'}
                                        </span>
                                        <span className="text-xs text-gray-400">
                                            {new Date(teacher.created_at).toLocaleDateString()}
                                        </span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); removeTeacher(teacher.id); }}
                                            className="text-red-600 hover:text-red-700 p-2 hover:bg-red-50 rounded-lg"
                                            title="Remove teacher"
                                        >
                                            <UserMinusIcon className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Invite Modal */}
            {showInviteModal && (
                <div className="fixed inset-0 bg-black/50 z-[100] flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold text-gray-900">Invite Teachers</h2>
                            <button onClick={() => setShowInviteModal(false)} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                                <p className="text-sm text-blue-800 mb-3">
                                    Share this access token with teachers. They can use it during registration to join your organization.
                                </p>
                                <div className="flex items-center gap-2 p-3 bg-white rounded-lg border border-blue-200">
                                    <code className="flex-1 text-lg font-mono font-bold text-blue-600">
                                        {accessToken}
                                    </code>
                                    <button
                                        onClick={() => {
                                            navigator.clipboard.writeText(accessToken)
                                            alert('Token copied!')
                                        }}
                                        className="btn-primary text-sm px-3 py-1"
                                    >
                                        Copy
                                    </button>
                                </div>
                            </div>

                            <div className="text-center text-gray-500 text-sm">
                                — or —
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Send Invite by Email
                                </label>
                                <div className="flex gap-2">
                                    <input
                                        type="email"
                                        placeholder="teacher@email.com"
                                        className="input-field flex-1"
                                    />
                                    <button className="btn-primary flex items-center gap-2 px-4">
                                        <EnvelopeIcon className="w-4 h-4" />
                                        Send
                                    </button>
                                </div>
                                <p className="text-xs text-gray-400 mt-1">
                                    (Email invites coming soon)
                                </p>
                            </div>
                        </div>

                        <button
                            onClick={() => setShowInviteModal(false)}
                            className="w-full mt-6 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
