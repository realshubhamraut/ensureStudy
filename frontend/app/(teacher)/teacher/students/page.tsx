'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState, useEffect } from 'react'
import {
    MagnifyingGlassIcon,
    PencilIcon,
    XMarkIcon,
    UserIcon,
    PhoneIcon,
    EnvelopeIcon
} from '@heroicons/react/24/outline'

interface Student {
    id: string
    email: string
    username: string
    first_name: string
    last_name: string
    phone?: string
    avatar_url?: string
    profile?: {
        grade_level: string
        board: string
        link_code: string
    }
    parent?: {
        id: string
        name: string
        email: string
        phone: string
    }
}

export default function TeacherStudentsPage() {
    const [students, setStudents] = useState<Student[]>([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const [editingStudent, setEditingStudent] = useState<Student | null>(null)
    const [showEditModal, setShowEditModal] = useState(false)
    const [editForm, setEditForm] = useState({
        first_name: '',
        last_name: '',
        phone: '',
        profile: {
            grade_level: '',
            board: ''
        }
    })

    useEffect(() => {
        fetchStudents()
    }, [])

    const fetchStudents = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/teacher/students`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setStudents(data.students)
            }
        } catch (error) {
            console.error('Failed to fetch students:', error)
        } finally {
            setLoading(false)
        }
    }

    const openEditModal = (student: Student) => {
        setEditingStudent(student)
        setEditForm({
            first_name: student.first_name || '',
            last_name: student.last_name || '',
            phone: student.phone || '',
            profile: {
                grade_level: student.profile?.grade_level || '',
                board: student.profile?.board || ''
            }
        })
        setShowEditModal(true)
    }

    const handleSave = async () => {
        if (!editingStudent) return

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/teacher/students/${editingStudent.id}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(editForm)
            })

            if (res.ok) {
                const data = await res.json()
                setStudents(students.map(s =>
                    s.id === editingStudent.id ? { ...s, ...data.student } : s
                ))
                setShowEditModal(false)
                alert('Student updated successfully!')
            }
        } catch (error) {
            console.error('Failed to update student:', error)
            alert('Failed to update student')
        }
    }

    const filteredStudents = students.filter(s =>
        s.email.toLowerCase().includes(search.toLowerCase()) ||
        s.first_name?.toLowerCase().includes(search.toLowerCase()) ||
        s.last_name?.toLowerCase().includes(search.toLowerCase())
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
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900">My Students</h1>
                <p className="text-gray-600">{students.length} students in your organization</p>
            </div>

            {/* Search */}
            <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                    type="text"
                    placeholder="Search students..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="input-field pl-10"
                />
            </div>

            {/* Students Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredStudents.map((student) => (
                    <div key={student.id} className="card-hover">
                        <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-3">
                                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center text-white font-bold text-lg">
                                    {student.first_name?.[0] || student.username[0]}
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-900">
                                        {student.first_name} {student.last_name}
                                    </h3>
                                    <p className="text-sm text-gray-500">@{student.username}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => openEditModal(student)}
                                className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg"
                            >
                                <PencilIcon className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="space-y-2 text-sm">
                            <div className="flex items-center gap-2 text-gray-600">
                                <EnvelopeIcon className="w-4 h-4" />
                                <span>{student.email}</span>
                            </div>
                            {student.phone && (
                                <div className="flex items-center gap-2 text-gray-600">
                                    <PhoneIcon className="w-4 h-4" />
                                    <span>{student.phone}</span>
                                </div>
                            )}
                        </div>

                        {student.profile && (
                            <div className="mt-4 pt-4 border-t border-gray-100">
                                <div className="flex gap-2">
                                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                                        Grade {student.profile.grade_level}
                                    </span>
                                    <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                                        {student.profile.board}
                                    </span>
                                </div>
                            </div>
                        )}

                        {student.parent ? (
                            <div className="mt-4 pt-4 border-t border-gray-100">
                                <p className="text-xs text-gray-500 mb-2">Parent/Guardian</p>
                                <div className="space-y-1">
                                    <div className="flex items-center gap-2">
                                        <UserIcon className="w-4 h-4 text-gray-400" />
                                        <span className="text-sm font-medium text-gray-700">{student.parent.name}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <EnvelopeIcon className="w-4 h-4 text-gray-400" />
                                        <span className="text-xs text-gray-600">{student.parent.email}</span>
                                    </div>
                                    {student.parent.phone && (
                                        <div className="flex items-center gap-2">
                                            <PhoneIcon className="w-4 h-4 text-gray-400" />
                                            <span className="text-xs text-gray-600">{student.parent.phone}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <div className="mt-4 pt-4 border-t border-gray-100">
                                <p className="text-xs text-yellow-600 bg-yellow-50 px-2 py-1 rounded">
                                    No parent linked yet
                                </p>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {filteredStudents.length === 0 && (
                <div className="text-center py-12 bg-gray-50 rounded-xl">
                    <p className="text-gray-500">No students found</p>
                </div>
            )}

            {/* Edit Modal */}
            {showEditModal && editingStudent && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold text-gray-900">Edit Student</h2>
                            <button onClick={() => setShowEditModal(false)} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                                    <input
                                        type="text"
                                        value={editForm.first_name}
                                        onChange={(e) => setEditForm({ ...editForm, first_name: e.target.value })}
                                        className="input-field"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                                    <input
                                        type="text"
                                        value={editForm.last_name}
                                        onChange={(e) => setEditForm({ ...editForm, last_name: e.target.value })}
                                        className="input-field"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                                <input
                                    type="tel"
                                    value={editForm.phone}
                                    onChange={(e) => setEditForm({ ...editForm, phone: e.target.value })}
                                    className="input-field"
                                    placeholder="+91 xxx xxx xxxx"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Grade</label>
                                    <select
                                        value={editForm.profile.grade_level}
                                        onChange={(e) => setEditForm({
                                            ...editForm,
                                            profile: { ...editForm.profile, grade_level: e.target.value }
                                        })}
                                        className="input-field"
                                    >
                                        <option value="">Select</option>
                                        {['9', '10', '11', '12'].map(g => (
                                            <option key={g} value={g}>{g}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Board</label>
                                    <select
                                        value={editForm.profile.board}
                                        onChange={(e) => setEditForm({
                                            ...editForm,
                                            profile: { ...editForm.profile, board: e.target.value }
                                        })}
                                        className="input-field"
                                    >
                                        <option value="">Select</option>
                                        {['CBSE', 'ICSE', 'State Board', 'IB'].map(b => (
                                            <option key={b} value={b}>{b}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="flex gap-3 mt-6">
                            <button
                                onClick={() => setShowEditModal(false)}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button onClick={handleSave} className="flex-1 btn-primary">
                                Save Changes
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
