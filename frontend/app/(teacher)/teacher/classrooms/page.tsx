'use client'

import { getApiBaseUrl, getAiServiceUrl } from '@/utils/api'

import { useState, useEffect, useRef } from 'react'
import { useSession } from 'next-auth/react'
import Link from 'next/link'
import {
    PlusIcon,
    ClipboardDocumentIcon,
    AcademicCapIcon,
    UsersIcon,
    ArrowPathIcon,
    XMarkIcon,
    DocumentTextIcon,
    CheckCircleIcon,
    ExclamationTriangleIcon,
    TrashIcon
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
    const [creationStep, setCreationStep] = useState<'info' | 'syllabus' | 'extracting' | 'done'>('info')
    const [syllabusFile, setSyllabusFile] = useState<File | null>(null)
    const [extractedTopics, setExtractedTopics] = useState<{ chapters: any[], topicCount: number }>({ chapters: [], topicCount: 0 })
    const [extractionError, setExtractionError] = useState<string>('')
    const fileInputRef = useRef<HTMLInputElement>(null)
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

    const handleNextStep = () => {
        if (creationStep === 'info') {
            if (!formData.name || !formData.subject) {
                alert('Please enter classroom name and subject')
                return
            }
            setCreationStep('syllabus')
        }
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file && file.type === 'application/pdf') {
            setSyllabusFile(file)
            setExtractionError('')
        } else {
            alert('Please select a PDF file')
        }
    }

    const handleCreate = async () => {
        if (!syllabusFile) {
            alert('Please upload a syllabus PDF to continue')
            return
        }

        setCreating(true)
        setCreationStep('extracting')
        setExtractionError('')

        try {
            // Step 1: Create the classroom
            const res = await fetch(`${apiUrl}/api/classroom`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })

            if (!res.ok) {
                throw new Error('Failed to create classroom')
            }

            const data = await res.json()
            const newClassroom = data.classroom

            // Step 2: Upload syllabus and extract topics
            const formDataSyllabus = new FormData()
            formDataSyllabus.append('file', syllabusFile)
            formDataSyllabus.append('classroom_id', newClassroom.id)
            formDataSyllabus.append('subject_name', formData.subject)

            const extractRes = await fetch(`${getAiServiceUrl()}/api/classroom-syllabus/extract`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.accessToken}`
                },
                body: formDataSyllabus
            })

            if (extractRes.ok) {
                const extractData = await extractRes.json()
                const chapters = extractData.chapters || []

                // Step 3: Save extracted hierarchy to database
                if (chapters.length > 0) {
                    const saveRes = await fetch(`${getAiServiceUrl()}/api/classroom-syllabus/save`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${session?.accessToken}`
                        },
                        body: JSON.stringify({
                            classroom_id: newClassroom.id,
                            chapters: chapters
                        })
                    })

                    if (saveRes.ok) {
                        const saveData = await saveRes.json()
                        setExtractedTopics({
                            chapters: chapters,
                            topicCount: saveData.topics_created || chapters.reduce((sum: number, ch: any) => sum + (ch.topics?.length || 0), 0)
                        })
                    }
                }
            }

            setClassrooms([...classrooms, newClassroom])
            setCreationStep('done')

        } catch (error) {
            console.error('Creation error:', error)
            setExtractionError(error instanceof Error ? error.message : 'Failed to create classroom')
            setCreationStep('syllabus')
        } finally {
            setCreating(false)
        }
    }

    const resetModal = () => {
        setShowCreateModal(false)
        setCreationStep('info')
        setFormData({ name: '', grade: '', section: '', subject: '' })
        setSyllabusFile(null)
        setExtractedTopics({ chapters: [], topicCount: 0 })
        setExtractionError('')
    }

    const copyCode = (code: string) => {
        navigator.clipboard.writeText(code)
        alert('Code copied!')
    }

    const handleDelete = async (e: React.MouseEvent, classroomId: string) => {
        e.preventDefault()
        e.stopPropagation()

        if (!confirm('Are you sure you want to delete this classroom? This action cannot be undone.')) {
            return
        }

        try {
            const res = await fetch(`${apiUrl}/api/classroom/${classroomId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${session?.accessToken}`
                }
            })

            if (res.ok) {
                setClassrooms(classrooms.filter(c => c.id !== classroomId))
            } else {
                const data = await res.json()
                alert(data.error || 'Failed to delete classroom')
            }
        } catch (error) {
            console.error('Delete error:', error)
            alert('Failed to delete classroom')
        }
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
                                <div className="flex items-center gap-2">
                                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${classroom.is_active
                                        ? 'bg-green-100 text-green-700'
                                        : 'bg-gray-100 text-gray-700'
                                        }`}>
                                        {classroom.is_active ? 'Active' : 'Inactive'}
                                    </span>
                                    <button
                                        onClick={(e) => handleDelete(e, classroom.id)}
                                        className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                                        title="Delete classroom"
                                    >
                                        <TrashIcon className="w-4 h-4" />
                                    </button>
                                </div>
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

            {/* Create Modal - Multi-step */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full p-6 max-h-[90vh] overflow-y-auto">
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Create Classroom</h2>
                                <div className="flex items-center gap-2 mt-2">
                                    <div className={`w-8 h-1 rounded ${creationStep === 'info' ? 'bg-purple-600' : 'bg-purple-200'}`} />
                                    <div className={`w-8 h-1 rounded ${creationStep === 'syllabus' || creationStep === 'extracting' ? 'bg-purple-600' : 'bg-purple-200'}`} />
                                    <div className={`w-8 h-1 rounded ${creationStep === 'done' ? 'bg-green-500' : 'bg-purple-200'}`} />
                                </div>
                            </div>
                            <button onClick={resetModal} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        {/* Step 1: Classroom Info */}
                        {creationStep === 'info' && (
                            <>
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
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Subject *</label>
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
                                    <button onClick={resetModal} className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
                                        Cancel
                                    </button>
                                    <button onClick={handleNextStep} className="flex-1 btn-primary">
                                        Next: Upload Syllabus
                                    </button>
                                </div>
                            </>
                        )}

                        {/* Step 2: Syllabus Upload */}
                        {creationStep === 'syllabus' && (
                            <>
                                <div className="text-center py-6">
                                    <DocumentTextIcon className="w-16 h-16 text-purple-500 mx-auto mb-4" />
                                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Syllabus PDF</h3>
                                    <p className="text-gray-500 text-sm mb-6">
                                        Upload your syllabus document to automatically extract chapters and topics for assessments
                                    </p>

                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept=".pdf"
                                        onChange={handleFileSelect}
                                        className="hidden"
                                    />

                                    {!syllabusFile ? (
                                        <button
                                            onClick={() => fileInputRef.current?.click()}
                                            className="w-full py-8 border-2 border-dashed border-purple-300 rounded-xl hover:border-purple-500 hover:bg-purple-50 transition-colors"
                                        >
                                            <PlusIcon className="w-8 h-8 text-purple-500 mx-auto mb-2" />
                                            <span className="text-purple-600 font-medium">Click to select PDF file</span>
                                        </button>
                                    ) : (
                                        <div className="p-4 bg-green-50 border border-green-200 rounded-xl">
                                            <CheckCircleIcon className="w-8 h-8 text-green-500 mx-auto mb-2" />
                                            <p className="font-medium text-green-700">{syllabusFile.name}</p>
                                            <p className="text-sm text-green-600">{(syllabusFile.size / 1024).toFixed(1)} KB</p>
                                            <button
                                                onClick={() => { setSyllabusFile(null); fileInputRef.current?.click() }}
                                                className="mt-2 text-sm text-purple-600 hover:underline"
                                            >
                                                Change file
                                            </button>
                                        </div>
                                    )}

                                    {extractionError && (
                                        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
                                            <ExclamationTriangleIcon className="w-5 h-5" />
                                            <span className="text-sm">{extractionError}</span>
                                        </div>
                                    )}
                                </div>

                                <div className="flex gap-3 mt-6">
                                    <button onClick={() => setCreationStep('info')} className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
                                        Back
                                    </button>
                                    <button
                                        onClick={handleCreate}
                                        disabled={!syllabusFile || creating}
                                        className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                    >
                                        {creating ? <ArrowPathIcon className="w-5 h-5 animate-spin" /> : <PlusIcon className="w-5 h-5" />}
                                        Create & Extract Topics
                                    </button>
                                </div>
                            </>
                        )}

                        {/* Step 3: Extracting */}
                        {creationStep === 'extracting' && (
                            <div className="text-center py-12">
                                <ArrowPathIcon className="w-16 h-16 text-purple-500 mx-auto mb-4 animate-spin" />
                                <h3 className="text-lg font-semibold text-gray-900 mb-2">Extracting Topics...</h3>
                                <p className="text-gray-500 text-sm">
                                    AI is analyzing your syllabus and extracting chapters and topics
                                </p>
                            </div>
                        )}

                        {/* Step 4: Done */}
                        {creationStep === 'done' && (
                            <>
                                <div className="text-center py-6">
                                    <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
                                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Classroom Created!</h3>
                                    <p className="text-gray-500 text-sm mb-4">
                                        {extractedTopics.topicCount > 0
                                            ? `${extractedTopics.chapters.length} chapters and ${extractedTopics.topicCount} topics extracted`
                                            : 'Classroom created successfully'}
                                    </p>

                                    {extractedTopics.chapters.length > 0 && (
                                        <div className="mt-4 text-left bg-gray-50 rounded-xl p-4 max-h-48 overflow-y-auto">
                                            <p className="text-sm font-medium text-gray-700 mb-2">Extracted Chapters:</p>
                                            <ul className="space-y-1">
                                                {extractedTopics.chapters.map((ch: any, i: number) => (
                                                    <li key={i} className="text-sm text-gray-600 flex items-center gap-2">
                                                        <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                                        {ch.name} ({ch.topics?.length || 0} topics)
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>

                                <button
                                    onClick={resetModal}
                                    className="w-full btn-primary mt-4"
                                >
                                    Done
                                </button>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
