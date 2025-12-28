'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    MagnifyingGlassIcon,
    UserMinusIcon,
    AcademicCapIcon,
    XMarkIcon,
    PhoneIcon,
    EnvelopeIcon,
    UserGroupIcon,
    ChevronDownIcon,
    ChevronUpIcon
} from '@heroicons/react/24/outline'

interface Parent {
    id: string
    name: string
    email: string
    phone: string
    relationship_type: string
}

interface Student {
    id: string
    email: string
    username: string
    first_name: string
    last_name: string
    phone?: string
    created_at: string
    is_active: boolean
    profile?: {
        grade_level: string
        board: string
        stream?: string
        target_exams: string[]
        subjects: string[]
    }
    parents?: Parent[]
    classrooms?: { id: string; name: string }[]
}

export default function StudentsPage() {
    const { data: session } = useSession()
    const [students, setStudents] = useState<Student[]>([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const [availableLicenses, setAvailableLicenses] = useState(0)
    const [selectedStudent, setSelectedStudent] = useState<Student | null>(null)
    const [expandedId, setExpandedId] = useState<string | null>(null)

    useEffect(() => {
        const fetchStudents = async () => {
            try {
                const res = await fetch('http://localhost:8000/api/admin/students', {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (res.ok) {
                    const data = await res.json()
                    setStudents(data.students)
                }

                // Fetch license info
                const dashRes = await fetch('http://localhost:8000/api/admin/dashboard', {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (dashRes.ok) {
                    const dashData = await dashRes.json()
                    setAvailableLicenses(dashData.stats.available_licenses)
                }
            } catch (error) {
                console.error('Failed to fetch students:', error)
            } finally {
                setLoading(false)
            }
        }

        if (session) {
            fetchStudents()
        }
    }, [session])

    const fetchStudentDetails = async (studentId: string) => {
        try {
            const res = await fetch(`http://localhost:8000/api/admin/students/${studentId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setSelectedStudent(data.student)
            }
        } catch (error) {
            console.error('Failed to fetch student details:', error)
        }
    }

    const removeStudent = async (studentId: string) => {
        if (!confirm('Are you sure you want to remove this student? This will free up 1 license.')) return

        try {
            const res = await fetch(`http://localhost:8000/api/admin/students/${studentId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setStudents(students.filter(s => s.id !== studentId))
                setAvailableLicenses(data.available_licenses)
                setSelectedStudent(null)
            }
        } catch (error) {
            console.error('Failed to remove student:', error)
        }
    }

    const filteredStudents = students.filter(s =>
        s.email.toLowerCase().includes(search.toLowerCase()) ||
        s.first_name?.toLowerCase().includes(search.toLowerCase()) ||
        s.last_name?.toLowerCase().includes(search.toLowerCase()) ||
        s.phone?.includes(search)
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
            {/* Student Detail Modal */}
            {selectedStudent && (
                <div className="fixed inset-0 bg-black/50 z-[100] flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl p-6 w-full max-w-2xl shadow-2xl max-h-[90vh] overflow-y-auto">
                        <div className="flex items-start justify-between mb-6">
                            <div className="flex items-center gap-4">
                                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center">
                                    <span className="text-white font-bold text-2xl">
                                        {selectedStudent.first_name?.[0] || selectedStudent.username[0]}
                                    </span>
                                </div>
                                <div>
                                    <h2 className="text-xl font-bold text-gray-900">
                                        {selectedStudent.first_name} {selectedStudent.last_name}
                                    </h2>
                                    <p className="text-gray-500">@{selectedStudent.username}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setSelectedStudent(null)}
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
                                <p className="font-medium text-gray-900">{selectedStudent.email}</p>
                            </div>
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="flex items-center gap-2 text-gray-500 text-sm mb-1">
                                    <PhoneIcon className="w-4 h-4" />
                                    Phone
                                </div>
                                <p className="font-medium text-gray-900">
                                    {selectedStudent.phone || 'Not provided'}
                                </p>
                            </div>
                        </div>

                        {/* Academic Info */}
                        {selectedStudent.profile && (
                            <div className="mb-6">
                                <h3 className="font-semibold text-gray-900 mb-3">📚 Academic Information</h3>
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="p-3 bg-blue-50 rounded-lg">
                                        <p className="text-xs text-blue-600 mb-1">Grade</p>
                                        <p className="font-semibold text-gray-900">{selectedStudent.profile.grade_level || '--'}</p>
                                    </div>
                                    <div className="p-3 bg-purple-50 rounded-lg">
                                        <p className="text-xs text-purple-600 mb-1">Board</p>
                                        <p className="font-semibold text-gray-900">{selectedStudent.profile.board || '--'}</p>
                                    </div>
                                    <div className="p-3 bg-orange-50 rounded-lg">
                                        <p className="text-xs text-orange-600 mb-1">Stream</p>
                                        <p className="font-semibold text-gray-900">{selectedStudent.profile.stream || '--'}</p>
                                    </div>
                                </div>

                                {selectedStudent.profile.target_exams?.length > 0 && (
                                    <div className="mt-3">
                                        <p className="text-sm text-gray-600 mb-2">Target Exams</p>
                                        <div className="flex flex-wrap gap-2">
                                            {selectedStudent.profile.target_exams.map(exam => (
                                                <span key={exam} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                                                    {exam}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {selectedStudent.profile.subjects?.length > 0 && (
                                    <div className="mt-3">
                                        <p className="text-sm text-gray-600 mb-2">Subjects</p>
                                        <div className="flex flex-wrap gap-2">
                                            {selectedStudent.profile.subjects.map(sub => (
                                                <span key={sub} className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                                                    {sub}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Parent/Guardian Info */}
                        <div className="mb-6">
                            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                                <UserGroupIcon className="w-5 h-5 text-orange-600" />
                                Parents/Guardians
                            </h3>
                            {selectedStudent.parents && selectedStudent.parents.length > 0 ? (
                                <div className="space-y-2">
                                    {selectedStudent.parents.map(parent => (
                                        <div key={parent.id} className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                                            <div className="flex items-start justify-between">
                                                <div>
                                                    <p className="font-medium text-gray-900">{parent.name}</p>
                                                    <p className="text-sm text-orange-600 capitalize">{parent.relationship_type}</p>
                                                </div>
                                                <span className="text-xs bg-orange-200 text-orange-700 px-2 py-1 rounded-full">Linked</span>
                                            </div>
                                            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                                                <div className="flex items-center gap-2 text-gray-600">
                                                    <EnvelopeIcon className="w-4 h-4" />
                                                    {parent.email}
                                                </div>
                                                <div className="flex items-center gap-2 text-gray-600">
                                                    <PhoneIcon className="w-4 h-4" />
                                                    {parent.phone || 'No phone'}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-gray-500 text-sm italic p-4 bg-gray-50 rounded-lg">
                                    No parents/guardians linked to this student
                                </p>
                            )}
                        </div>

                        {/* Classrooms */}
                        {selectedStudent.classrooms && selectedStudent.classrooms.length > 0 && (
                            <div className="mb-6">
                                <h3 className="font-semibold text-gray-900 mb-3">🏫 Enrolled Classrooms</h3>
                                <div className="flex flex-wrap gap-2">
                                    {selectedStudent.classrooms.map(c => (
                                        <span key={c.id} className="px-3 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm">
                                            {c.name}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Footer */}
                        <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                            <p className="text-sm text-gray-500">
                                Joined: {new Date(selectedStudent.created_at).toLocaleDateString()}
                            </p>
                            <button
                                onClick={() => removeStudent(selectedStudent.id)}
                                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm"
                            >
                                Remove Student
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">👨‍🎓 Students</h1>
                    <p className="text-gray-600">
                        {students.length} students • {availableLicenses} licenses available
                    </p>
                </div>
                <a href="/admin/billing" className="btn-primary">
                    Buy More Licenses
                </a>
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

            {/* Students List */}
            {filteredStudents.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-xl">
                    <AcademicCapIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">No students found</p>
                    <p className="text-sm text-gray-400 mt-1">
                        Share your access token with students to let them register
                    </p>
                </div>
            ) : (
                <div className="space-y-3">
                    {filteredStudents.map((student) => (
                        <div
                            key={student.id}
                            className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
                        >
                            <div
                                className="p-4 cursor-pointer"
                                onClick={() => fetchStudentDetails(student.id)}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center">
                                            <span className="text-white font-semibold">
                                                {student.first_name?.[0] || student.username[0]}
                                            </span>
                                        </div>
                                        <div>
                                            <div className="font-semibold text-gray-900">
                                                {student.first_name} {student.last_name}
                                            </div>
                                            <div className="text-sm text-gray-500 flex items-center gap-3">
                                                <span>{student.email}</span>
                                                {student.phone && (
                                                    <span className="flex items-center gap-1">
                                                        <PhoneIcon className="w-3 h-3" />
                                                        {student.phone}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        {student.profile && (
                                            <div className="text-right hidden md:block">
                                                <p className="text-sm font-medium text-gray-900">
                                                    Grade {student.profile.grade_level} • {student.profile.board}
                                                </p>
                                                <p className="text-xs text-gray-500">
                                                    {student.profile.target_exams?.join(', ') || 'No exams'}
                                                </p>
                                            </div>
                                        )}
                                        <span className="text-xs text-gray-400">
                                            {new Date(student.created_at).toLocaleDateString()}
                                        </span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); removeStudent(student.id); }}
                                            className="text-red-600 hover:text-red-700 p-2 hover:bg-red-50 rounded-lg"
                                            title="Remove student"
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
        </div>
    )
}
