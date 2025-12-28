'use client'

import { useState } from 'react'
import {
    DocumentTextIcon,
    PlusIcon,
    ClockIcon,
    XMarkIcon,
    SparklesIcon,
    ArrowPathIcon,
    UserGroupIcon,
    CheckIcon
} from '@heroicons/react/24/outline'

interface Question {
    id: string
    question: string
    options: string[]
    correct_answer: number
}

interface Assessment {
    id: string
    name: string
    subject: string
    questions: number
    due_date: string
    status: string
}

export default function TeacherAssessmentsPage() {
    const [assessments, setAssessments] = useState<Assessment[]>([
        { id: '1', name: 'Physics Unit Test 1', subject: 'Physics', questions: 20, due_date: '2024-01-15', status: 'active' },
        { id: '2', name: 'Chemistry Mid-term', subject: 'Chemistry', questions: 50, due_date: '2024-01-20', status: 'draft' },
    ])

    const [showCreateModal, setShowCreateModal] = useState(false)
    const [showAssignModal, setShowAssignModal] = useState(false)
    const [selectedAssessment, setSelectedAssessment] = useState<Assessment | null>(null)
    const [generating, setGenerating] = useState(false)
    const [generatedQuestions, setGeneratedQuestions] = useState<Question[]>([])
    const [selectedStudents, setSelectedStudents] = useState<string[]>([])
    const [dueDate, setDueDate] = useState('')
    const [formData, setFormData] = useState({
        name: '',
        subject: '',
        topic: '',
        numQuestions: 5,
        difficulty: 'medium'
    })

    // Mock students list
    const students = [
        { id: '1', name: 'John Doe', class: 'Class 10A' },
        { id: '2', name: 'Jane Smith', class: 'Class 10A' },
        { id: '3', name: 'Mike Johnson', class: 'Class 10B' },
        { id: '4', name: 'Sarah Wilson', class: 'Class 10B' },
        { id: '5', name: 'Alex Brown', class: 'Class 10A' },
    ]

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'active': return 'bg-green-100 text-green-700'
            case 'draft': return 'bg-yellow-100 text-yellow-700'
            case 'completed': return 'bg-gray-100 text-gray-700'
            default: return 'bg-gray-100 text-gray-700'
        }
    }

    const generateQuiz = async () => {
        if (!formData.subject || !formData.topic) {
            alert('Please enter subject and topic')
            return
        }

        setGenerating(true)

        // Simulated AI-generated questions
        // In production, this would call the AI service
        await new Promise(resolve => setTimeout(resolve, 2000))

        const mockQuestions: Question[] = Array(formData.numQuestions).fill(null).map((_, i) => ({
            id: `q${i + 1}`,
            question: `${formData.topic} Question ${i + 1}: What is the correct answer for this ${formData.difficulty} difficulty ${formData.subject} question?`,
            options: [
                'Option A - This is the first choice',
                'Option B - This is the second choice',
                'Option C - This is the third choice',
                'Option D - This is the fourth choice'
            ],
            correct_answer: Math.floor(Math.random() * 4)
        }))

        setGeneratedQuestions(mockQuestions)
        setGenerating(false)
    }

    const saveAssessment = () => {
        if (!formData.name) {
            alert('Please enter an assessment name')
            return
        }

        const newAssessment: Assessment = {
            id: Date.now().toString(),
            name: formData.name,
            subject: formData.subject,
            questions: generatedQuestions.length,
            due_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            status: 'draft'
        }

        setAssessments([newAssessment, ...assessments])
        setShowCreateModal(false)
        setGeneratedQuestions([])
        setFormData({ name: '', subject: '', topic: '', numQuestions: 5, difficulty: 'medium' })
        alert('Assessment saved as draft!')
    }

    // Open assign modal for a quiz
    const handleAssignQuiz = (assessment: Assessment) => {
        setSelectedAssessment(assessment)
        setSelectedStudents([])
        setDueDate(assessment.due_date)
        setShowAssignModal(true)
    }

    // Toggle student selection
    const toggleStudent = (studentId: string) => {
        setSelectedStudents(prev =>
            prev.includes(studentId)
                ? prev.filter(id => id !== studentId)
                : [...prev, studentId]
        )
    }

    // Select all students
    const selectAllStudents = () => {
        if (selectedStudents.length === students.length) {
            setSelectedStudents([])
        } else {
            setSelectedStudents(students.map(s => s.id))
        }
    }

    // Confirm assignment
    const confirmAssign = () => {
        if (selectedStudents.length === 0) {
            alert('Please select at least one student')
            return
        }
        if (!dueDate) {
            alert('Please set a due date')
            return
        }

        // Update assessment status to active
        setAssessments(prev => prev.map(a =>
            a.id === selectedAssessment?.id
                ? { ...a, status: 'active', due_date: dueDate }
                : a
        ))

        setShowAssignModal(false)
        alert(`Quiz assigned to ${selectedStudents.length} student(s)!`)
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Assessments</h1>
                    <p className="text-gray-600">Create AI-generated quizzes for your students</p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="btn-primary flex items-center gap-2"
                >
                    <PlusIcon className="w-5 h-5" />
                    Create Quiz
                </button>
            </div>

            {/* Assessments List */}
            {assessments.length === 0 ? (
                <div className="text-center py-16 bg-gray-50 rounded-xl">
                    <DocumentTextIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No assessments yet</h3>
                    <p className="text-gray-500 mb-4">Create your first quiz using AI</p>
                    <button onClick={() => setShowCreateModal(true)} className="btn-primary">
                        Create Quiz
                    </button>
                </div>
            ) : (
                <div className="space-y-4">
                    {assessments.map((assessment) => (
                        <div key={assessment.id} className="card-hover flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className="p-3 rounded-xl bg-purple-100">
                                    <DocumentTextIcon className="w-6 h-6 text-purple-600" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-900">{assessment.name}</h3>
                                    <p className="text-sm text-gray-500">{assessment.subject} • {assessment.questions} questions</p>
                                </div>
                            </div>

                            <div className="flex items-center gap-3">
                                <div className="text-right">
                                    <div className="flex items-center gap-1 text-sm text-gray-500">
                                        <ClockIcon className="w-4 h-4" />
                                        Due: {new Date(assessment.due_date).toLocaleDateString()}
                                    </div>
                                </div>
                                <span className={`px-3 py-1 rounded-full text-xs font-medium capitalize ${getStatusColor(assessment.status)}`}>
                                    {assessment.status}
                                </span>
                                <button
                                    onClick={() => handleAssignQuiz(assessment)}
                                    className="px-3 py-1.5 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-1"
                                >
                                    <UserGroupIcon className="w-4 h-4" />
                                    Assign
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Create Quiz Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="p-6 border-b border-gray-200 flex items-center justify-between sticky top-0 bg-white">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Create Quiz with AI</h2>
                                <p className="text-sm text-gray-500">Generate questions automatically</p>
                            </div>
                            <button onClick={() => setShowCreateModal(false)} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="p-6 space-y-6">
                            {/* Quiz Settings */}
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Assessment Name</label>
                                    <input
                                        type="text"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        className="input-field"
                                        placeholder="e.g., Physics Chapter 5 Quiz"
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
                                        <select
                                            value={formData.subject}
                                            onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                            className="input-field"
                                        >
                                            <option value="">Select Subject</option>
                                            {['Physics', 'Chemistry', 'Mathematics', 'Biology', 'English', 'History'].map(s => (
                                                <option key={s} value={s}>{s}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Difficulty</label>
                                        <select
                                            value={formData.difficulty}
                                            onChange={(e) => setFormData({ ...formData, difficulty: e.target.value })}
                                            className="input-field"
                                        >
                                            <option value="easy">Easy</option>
                                            <option value="medium">Medium</option>
                                            <option value="hard">Hard</option>
                                        </select>
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Topic</label>
                                    <input
                                        type="text"
                                        value={formData.topic}
                                        onChange={(e) => setFormData({ ...formData, topic: e.target.value })}
                                        className="input-field"
                                        placeholder="e.g., Newton's Laws of Motion"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Number of Questions</label>
                                    <input
                                        type="number"
                                        min={1}
                                        max={20}
                                        value={formData.numQuestions}
                                        onChange={(e) => setFormData({ ...formData, numQuestions: parseInt(e.target.value) || 5 })}
                                        className="input-field w-24"
                                    />
                                </div>

                                <button
                                    onClick={generateQuiz}
                                    disabled={generating}
                                    className="btn-primary flex items-center gap-2 w-full justify-center"
                                >
                                    {generating ? (
                                        <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                    ) : (
                                        <SparklesIcon className="w-5 h-5" />
                                    )}
                                    {generating ? 'Generating Questions...' : 'Generate Quiz with AI'}
                                </button>
                            </div>

                            {/* Generated Questions Preview */}
                            {generatedQuestions.length > 0 && (
                                <div className="border-t pt-6">
                                    <h3 className="font-semibold text-gray-900 mb-4">Generated Questions ({generatedQuestions.length})</h3>
                                    <div className="space-y-4 max-h-64 overflow-y-auto">
                                        {generatedQuestions.map((q, index) => (
                                            <div key={q.id} className="p-4 bg-gray-50 rounded-lg">
                                                <p className="font-medium text-gray-900 mb-2">
                                                    {index + 1}. {q.question}
                                                </p>
                                                <div className="grid grid-cols-2 gap-2">
                                                    {q.options.map((opt, i) => (
                                                        <div
                                                            key={i}
                                                            className={`text-sm p-2 rounded ${i === q.correct_answer
                                                                ? 'bg-green-100 text-green-700'
                                                                : 'bg-white text-gray-600'
                                                                }`}
                                                        >
                                                            {String.fromCharCode(65 + i)}. {opt}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    <button
                                        onClick={saveAssessment}
                                        className="mt-4 btn-primary w-full"
                                    >
                                        Save Assessment
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Assign Quiz Modal */}
            {showAssignModal && selectedAssessment && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
                        <div className="p-6 border-b border-gray-200 flex items-center justify-between sticky top-0 bg-white">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Assign Quiz</h2>
                                <p className="text-sm text-gray-500">{selectedAssessment.name}</p>
                            </div>
                            <button onClick={() => setShowAssignModal(false)} className="text-gray-400 hover:text-gray-600">
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="p-6 space-y-6">
                            {/* Due Date */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Due Date</label>
                                <input
                                    type="date"
                                    value={dueDate}
                                    onChange={(e) => setDueDate(e.target.value)}
                                    className="input-field"
                                    min={new Date().toISOString().split('T')[0]}
                                />
                            </div>

                            {/* Student Selection */}
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <label className="block text-sm font-medium text-gray-700">
                                        Select Students ({selectedStudents.length} selected)
                                    </label>
                                    <button
                                        onClick={selectAllStudents}
                                        className="text-sm text-primary-600 hover:underline"
                                    >
                                        {selectedStudents.length === students.length ? 'Deselect All' : 'Select All'}
                                    </button>
                                </div>

                                <div className="border rounded-lg divide-y max-h-60 overflow-y-auto">
                                    {students.map((student) => (
                                        <div
                                            key={student.id}
                                            onClick={() => toggleStudent(student.id)}
                                            className={`p-3 flex items-center justify-between cursor-pointer hover:bg-gray-50 ${selectedStudents.includes(student.id) ? 'bg-primary-50' : ''
                                                }`}
                                        >
                                            <div>
                                                <p className="font-medium text-gray-900">{student.name}</p>
                                                <p className="text-sm text-gray-500">{student.class}</p>
                                            </div>
                                            <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${selectedStudents.includes(student.id)
                                                    ? 'bg-primary-600 border-primary-600'
                                                    : 'border-gray-300'
                                                }`}>
                                                {selectedStudents.includes(student.id) && (
                                                    <CheckIcon className="w-3 h-3 text-white" />
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Actions */}
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setShowAssignModal(false)}
                                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmAssign}
                                    className="flex-1 btn-primary flex items-center justify-center gap-2"
                                >
                                    <UserGroupIcon className="w-4 h-4" />
                                    Assign to {selectedStudents.length} Student(s)
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
