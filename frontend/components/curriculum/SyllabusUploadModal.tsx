'use client'

import { useState, useRef } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import {
    CloudArrowUpIcon,
    DocumentTextIcon,
    ArrowPathIcon,
    CheckCircleIcon,
    XMarkIcon
} from '@heroicons/react/24/outline'

interface Props {
    onSuccess: (curriculum: any) => void
    onClose: () => void
}

export default function SyllabusUploadModal({ onSuccess, onClose }: Props) {
    const [file, setFile] = useState<File | null>(null)
    const [subjectName, setSubjectName] = useState('')
    const [hoursPerDay, setHoursPerDay] = useState(2)
    const [deadlineDays, setDeadlineDays] = useState(14)
    const [step, setStep] = useState<'upload' | 'preview' | 'generating' | 'done'>('upload')
    const [previewTopics, setPreviewTopics] = useState<string[]>([])
    const [error, setError] = useState('')
    const fileRef = useRef<HTMLInputElement>(null)

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
        if (!file) return

        setStep('generating')
        setError('')

        try {
            const userId = localStorage.getItem('userId') || 'demo-user'
            const formData = new FormData()
            formData.append('file', file)
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
                        <div className="p-2 bg-primary-100 rounded-lg">
                            <CloudArrowUpIcon className="w-6 h-6 text-primary-600" />
                        </div>
                        <h2 className="text-xl font-bold text-gray-900">Upload Syllabus</h2>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                        <XMarkIcon className="w-6 h-6" />
                    </button>
                </div>

                {/* Upload Step */}
                {step === 'upload' && (
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
                {step === 'preview' && (
                    <div className="p-6 space-y-4">
                        {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>}

                        <div>
                            <h3 className="font-medium text-gray-900 mb-2">Extracted Topics ({previewTopics.length})</h3>
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
                            <button onClick={() => setStep('upload')} className="flex-1 btn-secondary">Back</button>
                            <button onClick={generateCurriculum} className="flex-1 btn-primary">Generate Curriculum</button>
                        </div>
                    </div>
                )}

                {/* Generating Step */}
                {step === 'generating' && (
                    <div className="p-12 text-center">
                        <ArrowPathIcon className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
                        <p className="text-gray-600">Processing your syllabus...</p>
                    </div>
                )}

                {/* Done Step */}
                {step === 'done' && (
                    <div className="p-12 text-center">
                        <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
                        <p className="text-xl font-bold text-gray-900">Curriculum Generated!</p>
                    </div>
                )}
            </div>
        </div>
    )
}
