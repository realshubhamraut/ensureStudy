'use client'

import { useState, useRef } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import {
    CloudArrowUpIcon,
    DocumentTextIcon,
    ArrowPathIcon,
    CheckCircleIcon,
    XMarkIcon,
    AcademicCapIcon,
    BookOpenIcon,
    ChevronDownIcon,
    ChevronRightIcon,
    PencilIcon,
    TrashIcon,
    PlusIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface ExtractedTopic {
    name: string
    description: string
    difficulty: string
    estimated_hours: number
    key_concepts: string[]
}

interface ExtractedChapter {
    name: string
    description: string
    color: string
    order: number
    topics: ExtractedTopic[]
}

interface ExtractedHierarchy {
    subject_name: string
    description: string
    chapters: ExtractedChapter[]
}

interface Props {
    classroomId: string
    classroomName: string
    onSuccess: (hierarchy: any) => void
    onClose: () => void
}

// ============================================================================
// Constants
// ============================================================================

const CHAPTER_COLORS = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
    '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#14B8A6'
]

const DIFFICULTIES = ['easy', 'medium', 'hard']

// ============================================================================
// Main Component
// ============================================================================

export default function TeacherSyllabusModal({ classroomId, classroomName, onSuccess, onClose }: Props) {
    // State
    const [file, setFile] = useState<File | null>(null)
    const [step, setStep] = useState<'upload' | 'extracting' | 'preview' | 'saving' | 'done'>('upload')
    const [error, setError] = useState('')
    const [hierarchy, setHierarchy] = useState<ExtractedHierarchy | null>(null)
    const [expandedChapters, setExpandedChapters] = useState<Set<number>>(new Set([0]))
    const fileRef = useRef<HTMLInputElement>(null)

    // File handling
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0]
        if (f && (f.type === 'application/pdf' || f.name.endsWith('.pdf'))) {
            setFile(f)
            setError('')
        } else {
            setError('Please select a PDF file')
        }
    }

    // Extract hierarchy from syllabus
    const extractHierarchy = async () => {
        if (!file) {
            setError('Please select a file')
            return
        }

        setStep('extracting')
        setError('')

        try {
            const formData = new FormData()
            formData.append('file', file)
            formData.append('classroom_id', classroomId)

            const res = await fetch(`${getAiServiceUrl()}/api/classroom-syllabus/extract`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: formData
            })

            if (res.ok) {
                const data = await res.json()
                setHierarchy(data.hierarchy || data)
                // Expand first chapter
                setExpandedChapters(new Set([0]))
                setStep('preview')
            } else {
                const err = await res.json()
                setError(err.detail || 'Failed to extract hierarchy')
                setStep('upload')
            }
        } catch (e) {
            console.error('Extraction error:', e)
            setError('Failed to process syllabus. Please try again.')
            setStep('upload')
        }
    }

    // Save hierarchy to classroom
    const saveHierarchy = async () => {
        if (!hierarchy) return

        setStep('saving')
        setError('')

        try {
            const res = await fetch(`${getAiServiceUrl()}/api/classroom-syllabus/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({
                    classroom_id: classroomId,
                    chapters: hierarchy.chapters
                })
            })

            if (res.ok) {
                const data = await res.json()
                setStep('done')
                setTimeout(() => onSuccess(data), 1500)
            } else {
                const err = await res.json()
                setError(err.detail || 'Failed to save hierarchy')
                setStep('preview')
            }
        } catch (e) {
            console.error('Save error:', e)
            setError('Failed to save. Please try again.')
            setStep('preview')
        }
    }

    // Toggle chapter expansion
    const toggleChapter = (index: number) => {
        setExpandedChapters(prev => {
            const newSet = new Set(prev)
            if (newSet.has(index)) {
                newSet.delete(index)
            } else {
                newSet.add(index)
            }
            return newSet
        })
    }

    // Edit chapter name
    const updateChapterName = (index: number, name: string) => {
        if (!hierarchy) return
        const updated = { ...hierarchy }
        updated.chapters[index].name = name
        setHierarchy(updated)
    }

    // Edit topic
    const updateTopicName = (chapterIndex: number, topicIndex: number, name: string) => {
        if (!hierarchy) return
        const updated = { ...hierarchy }
        updated.chapters[chapterIndex].topics[topicIndex].name = name
        setHierarchy(updated)
    }

    // Delete topic
    const deleteTopic = (chapterIndex: number, topicIndex: number) => {
        if (!hierarchy) return
        const updated = { ...hierarchy }
        updated.chapters[chapterIndex].topics.splice(topicIndex, 1)
        setHierarchy(updated)
    }

    // Add topic
    const addTopic = (chapterIndex: number) => {
        if (!hierarchy) return
        const updated = { ...hierarchy }
        updated.chapters[chapterIndex].topics.push({
            name: 'New Topic',
            description: '',
            difficulty: 'medium',
            estimated_hours: 2,
            key_concepts: []
        })
        setHierarchy(updated)
    }

    // Calculate totals
    const totalTopics = hierarchy?.chapters.reduce((sum, ch) => sum + ch.topics.length, 0) || 0

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] flex flex-col">
                {/* Header */}
                <div className="p-6 border-b flex items-center justify-between flex-shrink-0">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-indigo-100">
                            <AcademicCapIcon className="w-6 h-6 text-indigo-600" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-gray-900">Upload Syllabus</h2>
                            <p className="text-sm text-gray-500">{classroomName}</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                        <XMarkIcon className="w-6 h-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto">
                    {/* Upload Step */}
                    {step === 'upload' && (
                        <div className="p-6 space-y-4">
                            {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}

                            {/* File Upload */}
                            <div
                                onClick={() => fileRef.current?.click()}
                                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${file ? 'border-indigo-300 bg-indigo-50' : 'border-gray-300 hover:border-gray-400'
                                    }`}
                            >
                                <input ref={fileRef} type="file" accept=".pdf" onChange={handleFileChange} className="hidden" />
                                {file ? (
                                    <div className="flex items-center justify-center gap-3">
                                        <DocumentTextIcon className="w-8 h-8 text-indigo-600" />
                                        <div className="text-left">
                                            <p className="font-medium text-gray-900">{file.name}</p>
                                            <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                                        <p className="text-gray-600">Click to upload syllabus PDF</p>
                                        <p className="text-sm text-gray-400">AI will extract chapters and topics</p>
                                    </>
                                )}
                            </div>

                            <button
                                onClick={extractHierarchy}
                                disabled={!file}
                                className="w-full py-3 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Extract Topics
                            </button>
                        </div>
                    )}

                    {/* Extracting Step */}
                    {step === 'extracting' && (
                        <div className="p-12 text-center">
                            <ArrowPathIcon className="w-12 h-12 text-indigo-500 animate-spin mx-auto mb-4" />
                            <p className="text-gray-600">Analyzing syllabus with AI...</p>
                            <p className="text-sm text-gray-400 mt-2">This may take a moment</p>
                        </div>
                    )}

                    {/* Preview Step */}
                    {step === 'preview' && hierarchy && (
                        <div className="p-6 space-y-4">
                            {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}

                            {/* Summary */}
                            <div className="flex items-center justify-between text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
                                <span>{hierarchy.chapters.length} chapters</span>
                                <span>{totalTopics} topics</span>
                            </div>

                            {/* Chapters */}
                            <div className="space-y-3">
                                {hierarchy.chapters.map((chapter, chapterIdx) => {
                                    const isExpanded = expandedChapters.has(chapterIdx)

                                    return (
                                        <div
                                            key={chapterIdx}
                                            className="border rounded-lg overflow-hidden"
                                            style={{ borderColor: chapter.color + '40' }}
                                        >
                                            {/* Chapter Header */}
                                            <div
                                                className="flex items-center gap-3 p-3 cursor-pointer hover:bg-gray-50"
                                                style={{ backgroundColor: chapter.color + '08' }}
                                            >
                                                <div
                                                    className="w-1 h-10 rounded-full"
                                                    style={{ backgroundColor: chapter.color }}
                                                />
                                                <button
                                                    onClick={() => toggleChapter(chapterIdx)}
                                                    className="p-1"
                                                >
                                                    {isExpanded ? (
                                                        <ChevronDownIcon className="w-4 h-4 text-gray-400" />
                                                    ) : (
                                                        <ChevronRightIcon className="w-4 h-4 text-gray-400" />
                                                    )}
                                                </button>
                                                <input
                                                    type="text"
                                                    value={chapter.name}
                                                    onChange={(e) => updateChapterName(chapterIdx, e.target.value)}
                                                    className="flex-1 font-medium text-gray-900 bg-transparent border-none focus:ring-0 p-0"
                                                />
                                                <span className="text-xs text-gray-500">
                                                    {chapter.topics.length} topics
                                                </span>
                                            </div>

                                            {/* Topics */}
                                            {isExpanded && (
                                                <div className="border-t divide-y" style={{ borderColor: chapter.color + '20' }}>
                                                    {chapter.topics.map((topic, topicIdx) => (
                                                        <div key={topicIdx} className="flex items-center gap-2 p-2 pl-10 group">
                                                            <BookOpenIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
                                                            <input
                                                                type="text"
                                                                value={topic.name}
                                                                onChange={(e) => updateTopicName(chapterIdx, topicIdx, e.target.value)}
                                                                className="flex-1 text-sm text-gray-700 bg-transparent border-none focus:ring-0 p-0"
                                                            />
                                                            <span className={`text-xs px-1.5 py-0.5 rounded ${topic.difficulty === 'easy' ? 'bg-green-100 text-green-600' :
                                                                topic.difficulty === 'hard' ? 'bg-red-100 text-red-600' :
                                                                    'bg-yellow-100 text-yellow-600'
                                                                }`}>
                                                                {topic.difficulty}
                                                            </span>
                                                            <button
                                                                onClick={() => deleteTopic(chapterIdx, topicIdx)}
                                                                className="p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                                                            >
                                                                <TrashIcon className="w-4 h-4" />
                                                            </button>
                                                        </div>
                                                    ))}
                                                    {/* Add topic button */}
                                                    <button
                                                        onClick={() => addTopic(chapterIdx)}
                                                        className="w-full flex items-center gap-2 p-2 pl-10 text-sm text-indigo-600 hover:bg-indigo-50"
                                                    >
                                                        <PlusIcon className="w-4 h-4" />
                                                        Add Topic
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    )
                                })}
                            </div>

                            {/* Actions */}
                            <div className="flex gap-3 pt-4">
                                <button
                                    onClick={() => setStep('upload')}
                                    className="flex-1 py-3 rounded-lg border border-gray-300 text-gray-700 font-medium hover:bg-gray-50"
                                >
                                    Back
                                </button>
                                <button
                                    onClick={saveHierarchy}
                                    className="flex-1 py-3 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700"
                                >
                                    Save to Classroom
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Saving Step */}
                    {step === 'saving' && (
                        <div className="p-12 text-center">
                            <ArrowPathIcon className="w-12 h-12 text-indigo-500 animate-spin mx-auto mb-4" />
                            <p className="text-gray-600">Saving to classroom...</p>
                        </div>
                    )}

                    {/* Done Step */}
                    {step === 'done' && (
                        <div className="p-12 text-center">
                            <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
                            <p className="text-xl font-bold text-gray-900">Syllabus Saved!</p>
                            <p className="text-gray-500 mt-2">{totalTopics} topics added to {classroomName}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
