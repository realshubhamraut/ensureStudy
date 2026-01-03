'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
    AcademicCapIcon,
    PlusIcon,
    FolderIcon,
    DocumentTextIcon,
    VideoCameraIcon,
    MusicalNoteIcon,
    PhotoIcon,
    ClockIcon,
    BookOpenIcon,
    XMarkIcon
} from '@heroicons/react/24/outline'

interface Material {
    id: string
    name: string
    type: string
    uploaded_at: string
}

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    subject: string
    join_code: string
    teacher?: { id: string; name: string; email?: string }
    materials?: Material[]
    due_assignments?: number
    has_syllabus?: boolean
    syllabus_content?: string
    syllabus_url?: string
}

export default function StudentClassroomsPage() {
    const [classrooms, setClassrooms] = useState<Classroom[]>([])
    const [loading, setLoading] = useState(true)
    const [showJoinModal, setShowJoinModal] = useState(false)
    const [joinCode, setJoinCode] = useState('')
    const [joining, setJoining] = useState(false)
    const [syllabusModal, setSyllabusModal] = useState<Classroom | null>(null)

    useEffect(() => {
        fetchClassrooms()
    }, [])

    const fetchClassrooms = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/my-classrooms`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setClassrooms(data.classrooms)
            }
        } catch (error) {
            console.error('Failed to fetch classrooms:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleJoin = async () => {
        if (!joinCode.trim()) return

        setJoining(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/join`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code: joinCode.trim() })
            })

            const data = await res.json()

            if (res.ok) {
                alert(`Joined ${data.classroom.name}!`)
                setShowJoinModal(false)
                setJoinCode('')
                fetchClassrooms()
            } else {
                alert(data.error || 'Failed to join')
            }
        } catch (error) {
            alert('Failed to join classroom')
        } finally {
            setJoining(false)
        }
    }

    const getFileIcon = (type: string) => {
        if (type.includes('video')) return <VideoCameraIcon className="w-4 h-4" />
        if (type.includes('audio')) return <MusicalNoteIcon className="w-4 h-4" />
        if (type.includes('image')) return <PhotoIcon className="w-4 h-4" />
        return <DocumentTextIcon className="w-4 h-4" />
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
                    <h1 className="text-2xl font-bold text-gray-900">My Classrooms</h1>
                    <p className="text-gray-600">View your classes and study materials</p>
                </div>
                <button
                    onClick={() => setShowJoinModal(true)}
                    className="btn-primary flex items-center gap-2"
                >
                    <PlusIcon className="w-5 h-5" />
                    Join Classroom
                </button>
            </div>

            {/* Classrooms Grid */}
            {classrooms.length === 0 ? (
                <div className="text-center py-16 bg-gray-50 rounded-xl">
                    <FolderIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No classrooms yet</h3>
                    <p className="text-gray-500 mb-4">Ask your teacher for a classroom code</p>
                    <button
                        onClick={() => setShowJoinModal(true)}
                        className="btn-primary"
                    >
                        Join Classroom
                    </button>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {classrooms.map((classroom) => (
                        <Link
                            key={classroom.id}
                            href={`/classrooms/${classroom.id}`}
                            className="card-hover group"
                        >
                            <div className="flex items-start gap-4 mb-4">
                                <div className="relative p-3 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500">
                                    <AcademicCapIcon className="w-6 h-6 text-white" />
                                    {/* Due Assignments Badge */}
                                    {classroom.due_assignments && classroom.due_assignments > 0 && (
                                        <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
                                            {classroom.due_assignments}
                                        </span>
                                    )}
                                </div>
                                <div className="flex-1">
                                    <h3 className="font-semibold text-gray-900 group-hover:text-green-600 transition-colors">
                                        {classroom.name}
                                    </h3>
                                    <p className="text-sm text-gray-500">
                                        {classroom.grade && `Grade ${classroom.grade}`}
                                        {classroom.section && ` • ${classroom.section}`}
                                    </p>
                                    {classroom.due_assignments && classroom.due_assignments > 0 && (
                                        <p className="text-xs text-red-600 mt-1 flex items-center gap-1">
                                            <ClockIcon className="w-3 h-3" />
                                            {classroom.due_assignments} due
                                        </p>
                                    )}
                                </div>
                            </div>

                            {classroom.teacher && (
                                <div className="flex items-center justify-between mb-3">
                                    <p className="text-sm text-gray-500">
                                        Teacher: {classroom.teacher.name}
                                    </p>
                                    {classroom.has_syllabus && (
                                        <button
                                            onClick={(e) => {
                                                e.preventDefault()
                                                e.stopPropagation()
                                                setSyllabusModal(classroom)
                                            }}
                                            className="text-xs text-primary-600 hover:text-primary-700 flex items-center gap-1 bg-primary-50 px-2 py-1 rounded-full"
                                        >
                                            <BookOpenIcon className="w-3 h-3" />
                                            Syllabus
                                        </button>
                                    )}
                                </div>
                            )}

                            {classroom.subject && (
                                <span className="inline-block px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                                    {classroom.subject}
                                </span>
                            )}

                            {/* Materials Preview */}
                            {classroom.materials && classroom.materials.length > 0 && (
                                <div className="mt-4 pt-4 border-t border-gray-100">
                                    <p className="text-xs text-gray-500 mb-2">Recent Materials</p>
                                    <div className="space-y-1">
                                        {classroom.materials.slice(0, 2).map((m) => (
                                            <div key={m.id} className="flex items-center gap-2 text-sm text-gray-600">
                                                {getFileIcon(m.type)}
                                                <span className="truncate">{m.name}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </Link>
                    ))}
                </div>
            )}

            {/* Join Modal */}
            {showJoinModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <h2 className="text-xl font-bold text-gray-900 mb-4">Join Classroom</h2>
                        <p className="text-gray-600 text-sm mb-4">
                            Enter the classroom code provided by your teacher
                        </p>

                        <input
                            type="text"
                            value={joinCode}
                            onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
                            placeholder="e.g., ABC123"
                            className="input-field text-center text-2xl font-mono tracking-widest uppercase mb-4"
                            maxLength={6}
                        />

                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowJoinModal(false)}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleJoin}
                                disabled={joining || !joinCode.trim()}
                                className="flex-1 btn-primary"
                            >
                                {joining ? 'Joining...' : 'Join'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Syllabus Preview Modal */}
            {syllabusModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-gray-100">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                                    <BookOpenIcon className="w-6 h-6 text-primary-600" />
                                    Syllabus
                                </h2>
                                <p className="text-sm text-gray-500 mt-1">
                                    {syllabusModal.name} {syllabusModal.teacher && `• ${syllabusModal.teacher.name}`}
                                </p>
                            </div>
                            <button
                                onClick={() => setSyllabusModal(null)}
                                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                            >
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6">
                            {syllabusModal.syllabus_content ? (
                                <div className="prose prose-sm max-w-none">
                                    <pre className="whitespace-pre-wrap font-sans text-gray-700 leading-relaxed">
                                        {syllabusModal.syllabus_content}
                                    </pre>
                                </div>
                            ) : syllabusModal.syllabus_url ? (
                                <div className="text-center py-8">
                                    <DocumentTextIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                                    <p className="text-gray-600 mb-4">View the syllabus document</p>
                                    <a
                                        href={syllabusModal.syllabus_url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="btn-primary"
                                    >
                                        Open Document
                                    </a>
                                </div>
                            ) : (
                                <p className="text-gray-500 text-center py-8">No syllabus content available</p>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="p-4 border-t border-gray-100 flex justify-end">
                            <button
                                onClick={() => setSyllabusModal(null)}
                                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
