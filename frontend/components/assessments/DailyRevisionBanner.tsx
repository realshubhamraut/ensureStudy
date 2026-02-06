'use client'

import { useState, useEffect, useCallback } from 'react'
import { CalendarDaysIcon, SparklesIcon, PlayIcon, CheckCircleIcon, ClockIcon } from '@heroicons/react/24/outline'
import { getApiBaseUrl } from '@/utils/api'
import clsx from 'clsx'

interface DailyRevisionData {
    assessment: {
        id: string
        title: string
        num_questions: number
        completed?: boolean
        score?: number
        revision_date?: string
    } | null
    topics_covered?: string[]
    message?: string
}

export default function DailyRevisionBanner() {
    const [data, setData] = useState<DailyRevisionData | null>(null)
    const [status, setStatus] = useState<'initializing' | 'loading' | 'generating' | 'ready' | 'no_topics' | 'error'>('initializing')
    const [errorMessage, setErrorMessage] = useState<string | null>(null)

    const today = new Date().toISOString().split('T')[0]

    // Generate assessment function
    const generateAssessment = useCallback(async () => {
        setStatus('generating')
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/assessments/generate-daily-revision`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ date: today })
            })

            if (res.ok) {
                const result = await res.json()
                if (result.assessment) {
                    setData(result)
                    setStatus('ready')
                } else {
                    setStatus('no_topics')
                    setErrorMessage(result.message || 'No topics scheduled for revision today')
                }
            } else {
                const err = await res.json()
                setStatus('error')
                setErrorMessage(err.error || 'Failed to generate assessment')
            }
        } catch (err) {
            console.error('Failed to generate revision assessment:', err)
            setStatus('error')
            setErrorMessage('Failed to connect to server')
        }
    }, [today])

    // Auto-trigger on mount
    useEffect(() => {
        const fetchAndGenerate = async () => {
            setStatus('loading')

            try {
                // First check if assessment already exists for today
                const res = await fetch(`${getApiBaseUrl()}/api/assessments/daily-revision?date=${today}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })

                if (res.ok) {
                    const result = await res.json()

                    if (result.assessment) {
                        // Assessment already exists
                        setData(result)
                        setStatus('ready')
                    } else {
                        // No assessment exists - auto-generate one
                        await generateAssessment()
                    }
                } else {
                    // Error fetching - try to generate anyway
                    await generateAssessment()
                }
            } catch (err) {
                console.error('Failed to check/generate revision assessment:', err)
                setStatus('error')
                setErrorMessage('Failed to connect to server')
            }
        }

        fetchAndGenerate()
    }, [today, generateAssessment])

    // Initializing - return nothing (prevents flash)
    if (status === 'initializing') {
        return null
    }

    // Loading state - show spinner
    if (status === 'loading' || status === 'generating') {
        return (
            <div className="relative bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 rounded-xl p-4 border border-indigo-100 shadow-sm mb-6">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl text-white shadow-lg">
                        <ClockIcon className="w-6 h-6 animate-spin" />
                    </div>

                    <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                            <SparklesIcon className="w-4 h-4 text-yellow-500" />
                            {status === 'loading' ? 'Checking revision schedule...' : 'Generating your assessment...'}
                        </h3>
                        <p className="text-sm text-gray-600">
                            {status === 'generating'
                                ? 'Creating AI-powered questions based on your revision topics...'
                                : 'Looking for topics scheduled for revision today...'}
                        </p>
                    </div>

                    <div className="w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin"></div>
                </div>
            </div>
        )
    }

    // No topics or error - don't show banner
    if (status === 'no_topics' || status === 'error') {
        return null
    }

    // No assessment data
    if (!data?.assessment) {
        return null
    }

    const assessment = data.assessment

    // Assessment exists - show it
    return (
        <div className="relative bg-gradient-to-r from-emerald-50 via-teal-50 to-cyan-50 rounded-xl p-4 border border-emerald-100 shadow-sm mb-6">
            <div className="flex items-center gap-4">
                <div className={clsx(
                    "p-3 rounded-xl text-white shadow-lg",
                    assessment.completed
                        ? "bg-gradient-to-br from-green-500 to-emerald-600"
                        : "bg-gradient-to-br from-teal-500 to-cyan-600"
                )}>
                    {assessment.completed ? (
                        <CheckCircleIcon className="w-6 h-6" />
                    ) : (
                        <CalendarDaysIcon className="w-6 h-6" />
                    )}
                </div>

                <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                        <SparklesIcon className="w-4 h-4 text-yellow-500" />
                        {assessment.title}
                        {assessment.completed && (
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                                Completed
                            </span>
                        )}
                    </h3>
                    <p className="text-sm text-gray-600">
                        {assessment.num_questions} AI-generated questions
                        {data.topics_covered && data.topics_covered.length > 0 && (
                            <> • {data.topics_covered.slice(0, 3).join(', ')}{data.topics_covered.length > 3 ? ` +${data.topics_covered.length - 3} more` : ''}</>
                        )}
                    </p>
                </div>

                {assessment.completed ? (
                    <div className="text-right">
                        <div className="text-2xl font-bold text-emerald-600">{Math.round(assessment.score || 0)}%</div>
                        <div className="text-xs text-gray-500">Score</div>
                    </div>
                ) : (
                    <a
                        href={`/assessments/take/${assessment.id}`}
                        className="px-4 py-2 bg-gradient-to-r from-teal-600 to-cyan-600 text-white rounded-lg font-medium text-sm flex items-center gap-2 hover:shadow-md transition-all"
                    >
                        <PlayIcon className="w-4 h-4" />
                        Start Assessment
                    </a>
                )}
            </div>
        </div>
    )
}
