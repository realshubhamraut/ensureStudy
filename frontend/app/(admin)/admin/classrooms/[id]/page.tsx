'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import {
    AcademicCapIcon,
    UsersIcon,
    PlusIcon,
    TrashIcon,
    ArrowLeftIcon,
    MagnifyingGlassIcon
} from '@heroicons/react/24/outline'

interface Teacher {
    id: string
    name: string
    email: string
    subject?: string
}

interface Student {
    id: string
    name: string
    email: string
    parent_linked: boolean
}

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    teachers: Teacher[]
    students: Student[]
}

// Mock data
const mockClassrooms: { [key: string]: Classroom } = {
    '1': {
        id: '1', name: 'Class 10-A', grade: '10', section: 'A',
        teachers: [
            { id: 't1', name: 'Mr. Sharma', email: 'sharma@school.com', subject: 'Physics' },
            { id: 't2', name: 'Mrs. Gupta', email: 'gupta@school.com', subject: 'Mathematics' },
            { id: 't3', name: 'Mr. Kumar', email: 'kumar@school.com', subject: 'Chemistry' }
        ],
        students: [
            { id: 's1', name: 'Rahul Verma', email: 'rahul@student.com', parent_linked: true },
            { id: 's2', name: 'Priya Singh', email: 'priya@student.com', parent_linked: true },
            { id: 's3', name: 'Amit Patel', email: 'amit@student.com', parent_linked: false },
            { id: 's4', name: 'Neha Sharma', email: 'neha@student.com', parent_linked: true },
            { id: 's5', name: 'Vikram Joshi', email: 'vikram@student.com', parent_linked: false }
        ]
    },
    '2': {
        id: '2', name: 'Class 10-B', grade: '10', section: 'B',
        teachers: [
            { id: 't4', name: 'Mrs. Mehta', email: 'mehta@school.com', subject: 'English' },
            { id: 't5', name: 'Mr. Das', email: 'das@school.com', subject: 'Biology' }
        ],
        students: [
            { id: 's6', name: 'Riya Kapoor', email: 'riya@student.com', parent_linked: true },
            { id: 's7', name: 'Arjun Malhotra', email: 'arjun@student.com', parent_linked: false }
        ]
    }
}

export default function ClassroomDetailPage() {
    const params = useParams()
    const router = useRouter()
    const classroomId = params.id as string

    const [classroom, setClassroom] = useState<Classroom | null>(null)
    const [loading, setLoading] = useState(true)
    const [activeTab, setActiveTab] = useState<'teachers' | 'students'>('students')
    const [search, setSearch] = useState('')

    useEffect(() => {
        // Simulate API fetch
        setTimeout(() => {
            setClassroom(mockClassrooms[classroomId] || null)
            setLoading(false)
        }, 300)
    }, [classroomId])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    if (!classroom) {
        return (
            <div className="text-center py-12">
                <p className="text-gray-500">Classroom not found</p>
                <Link href="/admin/classrooms" className="text-primary-600 hover:underline mt-2 inline-block">
                    Back to Classrooms
                </Link>
            </div>
        )
    }

    const filteredTeachers = classroom.teachers.filter(t =>
        t.name.toLowerCase().includes(search.toLowerCase()) ||
        t.email.toLowerCase().includes(search.toLowerCase())
    )

    const filteredStudents = classroom.students.filter(s =>
        s.name.toLowerCase().includes(search.toLowerCase()) ||
        s.email.toLowerCase().includes(search.toLowerCase())
    )

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-4">
                <Link
                    href="/admin/classrooms"
                    className="p-2 hover:bg-gray-100 rounded-lg"
                >
                    <ArrowLeftIcon className="w-5 h-5" />
                </Link>
                <div className="flex-1">
                    <h1 className="text-2xl font-bold text-gray-900">{classroom.name}</h1>
                    <p className="text-gray-600">Grade {classroom.grade} • Section {classroom.section}</p>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4">
                <div className="card flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-purple-100">
                        <UsersIcon className="w-6 h-6 text-purple-600" />
                    </div>
                    <div>
                        <p className="text-2xl font-bold text-gray-900">{classroom.teachers.length}</p>
                        <p className="text-sm text-gray-500">Teachers</p>
                    </div>
                </div>
                <div className="card flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-green-100">
                        <AcademicCapIcon className="w-6 h-6 text-green-600" />
                    </div>
                    <div>
                        <p className="text-2xl font-bold text-gray-900">{classroom.students.length}</p>
                        <p className="text-sm text-gray-500">Students</p>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-200">
                <div className="flex gap-4">
                    <button
                        onClick={() => setActiveTab('students')}
                        className={`pb-3 px-1 font-medium border-b-2 transition-colors ${activeTab === 'students'
                                ? 'border-primary-600 text-primary-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        Students ({classroom.students.length})
                    </button>
                    <button
                        onClick={() => setActiveTab('teachers')}
                        className={`pb-3 px-1 font-medium border-b-2 transition-colors ${activeTab === 'teachers'
                                ? 'border-primary-600 text-primary-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        Teachers ({classroom.teachers.length})
                    </button>
                </div>
            </div>

            {/* Search */}
            <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                    type="text"
                    placeholder={`Search ${activeTab}...`}
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="input-field pl-10"
                />
            </div>

            {/* Content */}
            {activeTab === 'students' ? (
                <div className="bg-white rounded-xl shadow overflow-hidden">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Student</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parent Linked</th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {filteredStudents.map((student) => (
                                <tr key={student.id} className="hover:bg-gray-50">
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                                                <span className="text-green-600 font-semibold">{student.name[0]}</span>
                                            </div>
                                            <span className="font-medium text-gray-900">{student.name}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-gray-500">{student.email}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-1 text-xs rounded-full ${student.parent_linked
                                                ? 'bg-green-100 text-green-700'
                                                : 'bg-yellow-100 text-yellow-700'
                                            }`}>
                                            {student.parent_linked ? 'Yes' : 'No'}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <button className="text-red-600 hover:text-red-700 p-2">
                                            <TrashIcon className="w-4 h-4" />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="bg-white rounded-xl shadow overflow-hidden">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Teacher</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Subject</th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {filteredTeachers.map((teacher) => (
                                <tr key={teacher.id} className="hover:bg-gray-50">
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
                                                <span className="text-purple-600 font-semibold">{teacher.name[0]}</span>
                                            </div>
                                            <span className="font-medium text-gray-900">{teacher.name}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-gray-500">{teacher.email}</td>
                                    <td className="px-6 py-4 text-gray-500">{teacher.subject || '-'}</td>
                                    <td className="px-6 py-4 text-right">
                                        <button className="text-red-600 hover:text-red-700 p-2">
                                            <TrashIcon className="w-4 h-4" />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Add Button */}
            <button className="btn-primary flex items-center gap-2">
                <PlusIcon className="w-5 h-5" />
                Add {activeTab === 'students' ? 'Student' : 'Teacher'} to Classroom
            </button>
        </div>
    )
}
