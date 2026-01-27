'use client'

import { useState, useRef, useEffect } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import {
    CloudArrowUpIcon,
    DocumentTextIcon,
    ArrowPathIcon,
    CheckCircleIcon,
    XMarkIcon,
    Cog6ToothIcon
} from '@heroicons/react/24/outline'

interface Props {
    onSuccess: (curriculum: any) => void
    onClose: () => void
    curriculumId?: string  // If provided, we're reconfiguring an existing curriculum
    subjectName?: string   // Pre-filled subject name for reconfiguration
}

interface Topic {
    id: string
    name: string
    description?: string
    estimated_hours?: number
}

export default function SyllabusUploadModal({ onSuccess, onClose, curriculumId, subjectName: initialSubjectName }: Props) {
    const [file, setFile] = useState<File | null>(null)
    const [subjectName, setSubjectName] = useState(initialSubjectName || '')
    const [hoursPerDay, setHoursPerDay] = useState(2)
    const [deadlineDays, setDeadlineDays] = useState(14)
    const [step, setStep] = useState<'upload' | 'preview' | 'generating' | 'done'>('upload')
    const [previewTopics, setPreviewTopics] = useState<string[]>([])
    const [existingTopics, setExistingTopics] = useState<Topic[]>([])
    const [error, setError] = useState('')
    const [loadingTopics, setLoadingTopics] = useState(false)
    const fileRef = useRef<HTMLInputElement>(null)

    // Check if we're in reconfigure mode
    const isReconfiguring = !!curriculumId

    // Fetch existing topics when reconfiguring
    useEffect(() => {
        if (isReconfiguring && curriculumId) {
            fetchExistingTopics()
        }
    }, [curriculumId])

    const fetchExistingTopics = async () => {
        setLoadingTopics(true)
        setError('')
        try {
            const res = await fetch(`${getAiServiceUrl()}/api/curriculum/${curriculumId}`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                const curriculum = data.curriculum || data
                const topics = curriculum.topics || []
                setExistingTopics(topics)
                setPreviewTopics(topics.map((t: Topic) => t.name))
                // Set existing schedule settings
                if (curriculum.hours_per_day) setHoursPerDay(curriculum.hours_per_day)
                if (curriculum.deadline_days) setDeadlineDays(curriculum.deadline_days)
                setStep('preview')  // Skip directly to preview
            } else {
                setError('Failed to load existing topics')
            }
        } catch (e) {
            setError('Failed to fetch curriculum data')
        }
        setLoadingTopics(false)
    }

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0]
        if (f && f.type === 'application/pdf') {
            setFile(f)
            setError('')
        } else {
            setError('Please select a PDF file')
        }
    }

    const previewSyllabus = async () => {
        if (!file || !subjectName) {
            setError('Please select a file and enter subject name')
            return
        }

        setStep('generating')
        setError('')

        try {
            const formData = new FormData()
            formData.append('file', file)
            formData.append('subject_name', subjectName)

            const res = await fetch(`${getAiServiceUrl()}/api/curriculum/syllabus/extract-topics`, {
                method: 'POST',
                body: formData
            })

            if (res.ok) {
                const data = await res.json()
                setPreviewTopics(data.topics || [])
                setStep('preview')
            } else {
                const err = await res.json()
                setError(err.detail || 'Failed to extract topics')
                setStep('upload')
            }
        } catch (e) {
            setError('Failed to process syllabus')
            setStep('upload')
        }
    }

    const generateCurriculum = async () => {
        if (!isReconfiguring && !file) return

        setStep('generating')
        setError('')

        try {
            const userId = localStorage.getItem('userId') || 'demo-user'

            if (isReconfiguring) {
                // Reconfigure existing curriculum - just update schedule with new params
                const res = await fetch(`${getAiServiceUrl()}/api/curriculum/${curriculumId}/reconfigure`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    },
                    body: JSON.stringify({
                        user_id: userId,
                        hours_per_day: hoursPerDay,
                        deadline_days: deadlineDays
                    })
                })

                if (res.ok) {
                    const data = await res.json()
                    setStep('done')
                    setTimeout(() => onSuccess(data.curriculum), 1500)
                } else {
                    const err = await res.json()
                    setError(err.detail || 'Failed to reconfigure curriculum')
                    setStep('preview')
                }
            } else {
                // New curriculum - upload file and generate
                const formData = new FormData()
                formData.append('file', file!)
                formData.append('user_id', userId)
                formData.append('subject_name', subjectName)
                formData.append('hours_per_day', hoursPerDay.toString())
                formData.append('deadline_days', deadlineDays.toString())

                const res = await fetch(`${getAiServiceUrl()}/api/curriculum/syllabus/upload`, {
                    method: 'POST',
                    body: formData
                })

                if (res.ok) {
                    const data = await res.json()
                    setStep('done')
                    setTimeout(() => onSuccess(data.curriculum), 1500)
                } else {
                    const err = await res.json()
                    setError(err.detail || 'Failed to generate curriculum')
                    setStep('preview')
                }
            }
        } catch (e) {
            setError('Failed to generate curriculum')
            setStep('preview')
        }
    }

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl max-w-lg w-full">
                {/* Header */}
                <div className="p-6 border-b flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${isReconfiguring ? 'bg-gray-100' : 'bg-primary-100'}`}>
                            {isReconfiguring ? (
                                <Cog6ToothIcon className="w-6 h-6 text-gray-600" />
                            ) : (
                                <CloudArrowUpIcon className="w-6 h-6 text-primary-600" />
                            )}
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-gray-900">
                                {isReconfiguring ? 'Reconfigure Schedule' : 'Upload Syllabus'}
                            </h2>
                            {isReconfiguring && (
                                <p className="text-sm text-gray-500">Adjust schedule for {initialSubjectName}</p>
                            )}
                        </div>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                        <XMarkIcon className="w-6 h-6" />
                    </button>
                </div>

                {/* Loading Topics (for reconfigure mode) */}
                {loadingTopics && (
                    <div className="p-12 text-center">
                        <ArrowPathIcon className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
                        <p className="text-gray-600">Loading existing topics...</p>
                    </div>
                )}

                {/* Upload Step (only for new curriculum) */}
                {step === 'upload' && !isReconfiguring && !loadingTopics && (
                    <div className="p-6 space-y-4">
                        {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}

                        {/* File Upload */}
                        <div
                            onClick={() => fileRef.current?.click()}
                            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${file ? 'border-primary-300 bg-primary-50' : 'border-gray-300 hover:border-gray-400'
                                }`}
                        >
                            <input ref={fileRef} type="file" accept=".pdf" onChange={handleFileChange} className="hidden" />
                            {file ? (
                                <div className="flex items-center justify-center gap-3">
                                    <DocumentTextIcon className="w-8 h-8 text-primary-600" />
                                    <div className="text-left">
                                        <p className="font-medium text-gray-900">{file.name}</p>
                                        <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                                    <p className="text-gray-600">Click to upload syllabus PDF</p>
                                    <p className="text-sm text-gray-400">Max 10MB</p>
                                </>
                            )}
                        </div>

                        {/* Subject Name */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Subject Name</label>
                            <input
                                type="text"
                                value={subjectName}
                                onChange={e => setSubjectName(e.target.value)}
                                placeholder="e.g., Mathematics, Physics"
                                className="input-field"
                            />
                        </div>

                        <button
                            onClick={previewSyllabus}
                            disabled={!file || !subjectName}
                            className="btn-primary w-full disabled:opacity-50"
                        >
                            Extract Topics
                        </button>
                    </div>
                )}

                {/* Preview Step */}
                {step === 'preview' && !loadingTopics && (
                    <div className="p-6 space-y-4">
                        {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}

                        <div>
                            <h3 className="font-medium text-gray-900 mb-2">
                                {isReconfiguring ? 'Current Topics' : 'Extracted Topics'} ({previewTopics.length})
                            </h3>
                            <div className="max-h-40 overflow-y-auto space-y-1">
                                {previewTopics.map((t, i) => (
                                    <div key={i} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                                        <span className="w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-xs flex items-center justify-center">
                                            {i + 1}
                                        </span>
                                        <span className="text-sm">{t}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Hours/Day</label>
                                <select value={hoursPerDay} onChange={e => setHoursPerDay(Number(e.target.value))} className="input-field">
                                    {[1, 2, 3, 4, 5, 6].map(h => <option key={h} value={h}>{h} hours</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Target Days</label>
                                <select value={deadlineDays} onChange={e => setDeadlineDays(Number(e.target.value))} className="input-field">
                                    {[7, 14, 21, 30, 45, 60].map(d => <option key={d} value={d}>{d} days</option>)}
                                </select>
                            </div>
                        </div>

                        <div className="flex gap-3">
                            {!isReconfiguring && (
                                <button onClick={() => setStep('upload')} className="flex-1 btn-secondary">Back</button>
                            )}
                            <button onClick={generateCurriculum} className={`${isReconfiguring ? 'w-full' : 'flex-1'} btn-primary`}>
                                {isReconfiguring ? 'Update Schedule' : 'Generate Curriculum'}
                            </button>
                        </div>
                    </div>
                )}

                {/* Generating Step */}
                {step === 'generating' && (
                    <div className="p-12 text-center">
                        <ArrowPathIcon className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
                        <p className="text-gray-600">
                            {isReconfiguring ? 'Updating schedule...' : 'Processing your syllabus...'}
                        </p>
                    </div>
                )}

                {/* Done Step */}
                {step === 'done' && (
                    <div className="p-12 text-center">
                        <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
                        <p className="text-xl font-bold text-gray-900">
                            {isReconfiguring ? 'Schedule Updated!' : 'Curriculum Generated!'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}
