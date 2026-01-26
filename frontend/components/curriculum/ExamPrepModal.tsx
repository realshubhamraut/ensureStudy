'use client'

import { useState } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import {
    CalendarIcon,
    ClockIcon,
    AcademicCapIcon,
    ExclamationTriangleIcon,
    CheckCircleIcon,
    BookOpenIcon,
    PlayIcon
} from '@heroicons/react/24/outline'

interface ExamPrepPlan {
    exam_id: string
    exam_name: string
    exam_date: string
    days_until_exam: number
    weak_topics: { name: string; mastery: number }[]
    strong_topics: { name: string; mastery: number }[]
    prep_days: {
        day: number
        date: string
        focus_topics: string[]
        activities: { type: string; description: string; duration_min: number }[]
        is_review_day: boolean
        is_exam_day: boolean
    }[]
    practice_tests: { day: number; type: string; description: string }[]
}

interface Props {
    curriculumId: string
    subjectName: string
    onClose: () => void
}

export default function ExamPrepModal({ curriculumId, subjectName, onClose }: Props) {
    const [step, setStep] = useState<'form' | 'loading' | 'result'>('form')
    const [examName, setExamName] = useState('')
    const [examDate, setExamDate] = useState('')
    const [hoursPerDay, setHoursPerDay] = useState(3)
    const [plan, setPlan] = useState<ExamPrepPlan | null>(null)
    const [error, setError] = useState('')

    const createPlan = async () => {
        if (!examName || !examDate) {
            setError('Please fill all fields')
            return
        }

        setStep('loading')
        setError('')

        try {
            const userId = localStorage.getItem('userId') || 'demo-user'
            const res = await fetch(`${getAiServiceUrl()}/api/curriculum/exam-prep/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    exam_name: examName,
                    exam_date: examDate,
                    curriculum_id: curriculumId,
                    user_id: userId,
                    hours_per_day: hoursPerDay,
                    include_resources: true
                })
            })

            if (res.ok) {
                const data = await res.json()
                setPlan(data.exam_prep)
                setStep('result')
            } else {
                const err = await res.json()
                setError(err.detail || 'Failed to create plan')
                setStep('form')
            }
        } catch (e) {
            setError('Failed to create exam prep plan')
            setStep('form')
        }
    }

    const getMinDate = () => {
        const tomorrow = new Date()
        tomorrow.setDate(tomorrow.getDate() + 1)
        return tomorrow.toISOString().split('T')[0]
    }

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="p-6 border-b flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-red-100 rounded-lg">
                            <AcademicCapIcon className="w-6 h-6 text-red-600" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-gray-900">Exam Prep Mode</h2>
                            <p className="text-sm text-gray-500">{subjectName}</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button>
                </div>

                {/* Form */}
                {step === 'form' && (
                    <div className="p-6 space-y-4">
                        {error && (
                            <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>
                        )}

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Exam Name</label>
                            <input
                                type="text"
                                value={examName}
                                onChange={e => setExamName(e.target.value)}
                                placeholder="e.g., Final Exam, Midterm, Unit Test"
                                className="input-field"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Exam Date</label>
                            <input
                                type="date"
                                value={examDate}
                                onChange={e => setExamDate(e.target.value)}
                                min={getMinDate()}
                                className="input-field"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Study Hours Per Day</label>
                            <select value={hoursPerDay} onChange={e => setHoursPerDay(Number(e.target.value))} className="input-field">
                                {[1, 2, 3, 4, 5, 6, 7, 8].map(h => (
                                    <option key={h} value={h}>{h} hours</option>
                                ))}
                            </select>
                        </div>

                        <div className="bg-yellow-50 p-4 rounded-lg">
                            <div className="flex items-start gap-2">
                                <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 mt-0.5" />
                                <div className="text-sm text-yellow-800">
                                    <p className="font-medium">Intensive Mode</p>
                                    <p>This will create an intensified study schedule focusing on your weak topics.</p>
                                </div>
                            </div>
                        </div>

                        <button onClick={createPlan} className="btn-primary w-full">
                            Create Exam Prep Plan
                        </button>
                    </div>
                )}

                {/* Loading */}
                {step === 'loading' && (
                    <div className="p-12 text-center">
                        <div className="spinner mx-auto mb-4"></div>
                        <p className="text-gray-600">Creating your personalized exam prep plan...</p>
                    </div>
                )}

                {/* Result */}
                {step === 'result' && plan && (
                    <div className="p-6 space-y-6">
                        {/* Summary */}
                        <div className="grid grid-cols-3 gap-4">
                            <div className="text-center p-4 bg-red-50 rounded-lg">
                                <p className="text-2xl font-bold text-red-600">{plan.days_until_exam}</p>
                                <p className="text-xs text-gray-500">Days Left</p>
                            </div>
                            <div className="text-center p-4 bg-yellow-50 rounded-lg">
                                <p className="text-2xl font-bold text-yellow-600">{plan.weak_topics.length}</p>
                                <p className="text-xs text-gray-500">Weak Topics</p>
                            </div>
                            <div className="text-center p-4 bg-green-50 rounded-lg">
                                <p className="text-2xl font-bold text-green-600">{plan.practice_tests.length}</p>
                                <p className="text-xs text-gray-500">Practice Tests</p>
                            </div>
                        </div>

                        {/* Weak Topics */}
                        {plan.weak_topics.length > 0 && (
                            <div>
                                <h3 className="font-medium text-gray-900 mb-2">Focus Areas (Weak Topics)</h3>
                                <div className="flex flex-wrap gap-2">
                                    {plan.weak_topics.map((t, i) => (
                                        <span key={i} className="px-3 py-1 bg-red-100 text-red-700 rounded-lg text-sm">
                                            {t.name} ({Math.round(t.mastery * 100)}%)
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Schedule Preview */}
                        <div>
                            <h3 className="font-medium text-gray-900 mb-2">Study Schedule</h3>
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                                {plan.prep_days.slice(0, 7).map((day, i) => (
                                    <div key={i} className={`p-3 rounded-lg border ${day.is_exam_day ? 'bg-red-50 border-red-200' :
                                            day.is_review_day ? 'bg-blue-50 border-blue-200' :
                                                'bg-gray-50 border-gray-200'
                                        }`}>
                                        <div className="flex items-center justify-between">
                                            <span className="font-medium">Day {day.day}</span>
                                            <span className="text-xs text-gray-500">{day.date}</span>
                                        </div>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                            {day.focus_topics.map((t, j) => (
                                                <span key={j} className="text-xs px-2 py-0.5 bg-white rounded">{t}</span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <button onClick={onClose} className="btn-primary w-full">
                            Start Studying
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}
