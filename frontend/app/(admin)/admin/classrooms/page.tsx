'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
    AcademicCapIcon,
    PlusIcon,
    PencilIcon,
    TrashIcon,
    XMarkIcon,
    MagnifyingGlassIcon,
    ChevronRightIcon
} from '@heroicons/react/24/outline'

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    teacher_count: number
    student_count: number
}

export default function ClassroomsPage() {
    const { data: session } = useSession()
    const [classrooms, setClassrooms] = useState<Classroom[]>([
        { id: '1', name: 'Class 10-A', grade: '10', section: 'A', teacher_count: 3, student_count: 35 },
        { id: '2', name: 'Class 10-B', grade: '10', section: 'B', teacher_count: 3, student_count: 32 },
        { id: '3', name: 'Class 11-Science', grade: '11', section: 'Science', teacher_count: 5, student_count: 40 },
        { id: '4', name: 'Class 11-Commerce', grade: '11', section: 'Commerce', teacher_count: 4, student_count: 38 },
        { id: '5', name: 'Class 12-Science', grade: '12', section: 'Science', teacher_count: 5, student_count: 42 },
    ])
    const [loading, setLoading] = useState(false)
    const [search, setSearch] = useState('')
    const [showModal, setShowModal] = useState(false)
    const [editingId, setEditingId] = useState<string | null>(null)
    const [formData, setFormData] = useState({
        name: '',
        grade: '',
        section: ''
    })

    const filteredClassrooms = classrooms.filter(c =>
        c.name.toLowerCase().includes(search.toLowerCase()) ||
        c.grade.includes(search) ||
        c.section.toLowerCase().includes(search.toLowerCase())
    )

    const openAddModal = () => {
        setEditingId(null)
        setFormData({ name: '', grade: '', section: '' })
        setShowModal(true)
    }

    const openEditModal = (e: React.MouseEvent, classroom: Classroom) => {
        e.preventDefault()
        e.stopPropagation()
        setEditingId(classroom.id)
        setFormData({ name: classroom.name, grade: classroom.grade, section: classroom.section })
        setShowModal(true)
    }

    const handleSave = () => {
        if (editingId) {
            setClassrooms(classrooms.map(c =>
                c.id === editingId
                    ? { ...c, name: formData.name, grade: formData.grade, section: formData.section }
                    : c
            ))
        } else {
            const newClassroom: Classroom = {
                id: Date.now().toString(),
                name: formData.name || `Class ${formData.grade}-${formData.section}`,
                grade: formData.grade,
                section: formData.section,
                teacher_count: 0,
                student_count: 0
            }
            setClassrooms([...classrooms, newClassroom])
        }
        setShowModal(false)
    }

    const handleDelete = (e: React.MouseEvent, id: string) => {
        e.preventDefault()
        e.stopPropagation()
        if (confirm('Are you sure you want to delete this classroom?')) {
            setClassrooms(classrooms.filter(c => c.id !== id))
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Classrooms</h1>
                    <p className="text-gray-600">{classrooms.length} classrooms in your organization</p>
                </div>
                <button onClick={openAddModal} className="btn-primary flex items-center gap-2">
                    <PlusIcon className="w-5 h-5" />
                    Add Classroom
                </button>
            </div>

            {/* Search */}
            <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                    type="text"
                    placeholder="Search classrooms..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="input-field pl-10"
                />
            </div>

            {/* Classrooms Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredClassrooms.map((classroom) => (
                    <Link
                        key={classroom.id}
                        href={`/admin/classrooms/${classroom.id}`}
                        className="card-hover group cursor-pointer"
                    >
                        <div className="flex items-start justify-between mb-4">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                                <AcademicCapIcon className="w-6 h-6 text-white" />
                            </div>
                            <div className="flex gap-1">
                                <button
                                    onClick={(e) => openEditModal(e, classroom)}
                                    className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg"
                                >
                                    <PencilIcon className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={(e) => handleDelete(e, classroom.id)}
                                    className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg"
                                >
                                    <TrashIcon className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="font-semibold text-gray-900 text-lg group-hover:text-primary-600 transition-colors">
                                    {classroom.name}
                                </h3>
                                <p className="text-sm text-gray-500 mt-1">Grade {classroom.grade} • Section {classroom.section}</p>
                            </div>
                            <ChevronRightIcon className="w-5 h-5 text-gray-400 group-hover:text-primary-600 transition-colors" />
                        </div>

                        <div className="mt-4 pt-4 border-t border-gray-100 flex justify-between text-sm">
                            <span className="text-gray-500">
                                <strong className="text-gray-900">{classroom.teacher_count}</strong> Teachers
                            </span>
                            <span className="text-gray-500">
                                <strong className="text-gray-900">{classroom.student_count}</strong> Students
                            </span>
                        </div>
                    </Link>
                ))}
            </div>

            {/* Add/Edit Modal */}
            {showModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold text-gray-900">
                                {editingId ? 'Edit Classroom' : 'Add Classroom'}
                            </h2>
                            <button onClick={() => setShowModal(false)} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                                <input
                                    type="text"
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    className="input-field"
                                    placeholder="e.g., Class 10-A"
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
                        </div>

                        <div className="flex gap-3 mt-6">
                            <button
                                onClick={() => setShowModal(false)}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleSave}
                                className="flex-1 btn-primary"
                            >
                                {editingId ? 'Save Changes' : 'Add Classroom'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
