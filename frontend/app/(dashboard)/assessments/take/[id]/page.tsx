'use client'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useSession } from 'next-auth/react'
import { ArrowLeftIcon, ArrowRightIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

import AssessmentTimer from '@/components/assessments/AssessmentTimer'
import QuestionNavigator from '@/components/assessments/QuestionNavigator'
import QuestionCard from '@/components/assessments/QuestionCard'
import { getApiBaseUrl } from '@/utils/api'

interface Option {
    id: string
    text: string
    explanation?: string
    is_correct?: boolean
}

interface Question {
    id: string
    question_type: 'mcq' | 'descriptive'
    question_text: string
    options?: Option[]
    marks: number
    difficulty: string
}

interface Assessment {
    id: string
    title: string
    topic?: string
    time_limit_minutes: number
    questions: Question[]
    total_marks: number
}

interface SubmissionResult {
    score: number
    total_marks: number
    percentage: number
    correct_count: number
    total_questions: number
    answers_with_feedback: {
        question_id: string
        selected_answer: string
        correct_answer: string
        is_correct: boolean
    }[]
}

export default function TakeAssessmentPage() {
    const params = useParams()
    const router = useRouter()
    const { data: session } = useSession()

    const assessmentId = params.id as string

    const [assessment, setAssessment] = useState<Assessment | null>(null)
    const [currentQuestion, setCurrentQuestion] = useState(0)
    const [answers, setAnswers] = useState<Map<number, string>>(new Map())
    const [flaggedQuestions, setFlaggedQuestions] = useState<Set<number>>(new Set())
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [result, setResult] = useState<SubmissionResult | null>(null)
    const [showFeedback, setShowFeedback] = useState(false)
    const [showNavigator, setShowNavigator] = useState(false)

    // Fetch assessment on mount
    useEffect(() => {
        fetchAssessment()
    }, [assessmentId])

    const fetchAssessment = async () => {
        setIsLoading(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/assessments/${assessmentId}/start`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (!res.ok) {
                throw new Error('Failed to load assessment')
            }

            const data = await res.json()
            setAssessment(data.assessment)
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Something went wrong')
        } finally {
            setIsLoading(false)
        }
    }

    const handleSelectAnswer = (answer: string) => {
        if (showFeedback) return // Don't allow changes after submission

        setAnswers(prev => {
            const newAnswers = new Map(prev)
            newAnswers.set(currentQuestion, answer)
            return newAnswers
        })
    }

    const handleToggleFlag = () => {
        setFlaggedQuestions(prev => {
            const newFlagged = new Set(prev)
            if (newFlagged.has(currentQuestion)) {
                newFlagged.delete(currentQuestion)
            } else {
                newFlagged.add(currentQuestion)
            }
            return newFlagged
        })
    }

    const handlePrevious = () => {
        if (currentQuestion > 0) {
            setCurrentQuestion(currentQuestion - 1)
        }
    }

    const handleNext = () => {
        if (assessment && currentQuestion < assessment.questions.length - 1) {
            setCurrentQuestion(currentQuestion + 1)
        }
    }

    const handleSubmit = useCallback(async () => {
        if (!assessment || isSubmitting) return

        // Confirm if not all questions are answered
        const unansweredCount = assessment.questions.length - answers.size
        if (unansweredCount > 0) {
            const confirmed = confirm(
                `You have ${unansweredCount} unanswered question${unansweredCount > 1 ? 's' : ''}. Are you sure you want to submit?`
            )
            if (!confirmed) return
        }

        setIsSubmitting(true)

        try {
            // Prepare answers as object with index keys (backend expects this format)
            const answersPayload: Record<number, string | null> = {}
            assessment.questions.forEach((q, i) => {
                answersPayload[i] = answers.get(i) || null
            })

            const res = await fetch(`${getApiBaseUrl()}/api/assessments/${assessmentId}/submit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({ answers: answersPayload })
            })

            if (!res.ok) {
                throw new Error('Failed to submit assessment')
            }

            const data = await res.json()
            setResult(data.result)
            setShowFeedback(true)

            // Update assessment with correct answers for feedback
            if (data.assessment_with_answers) {
                setAssessment(data.assessment_with_answers)
            }

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to submit')
        } finally {
            setIsSubmitting(false)
        }
    }, [assessment, assessmentId, answers, isSubmitting])

    const handleTimeUp = useCallback(() => {
        handleSubmit()
    }, [handleSubmit])

    // Loading state
    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-gray-500">Loading assessment...</p>
                </div>
            </div>
        )
    }

    // Error state
    if (error || !assessment) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center max-w-md">
                    <p className="text-red-500 mb-4">{error || 'Assessment not found'}</p>
                    <button
                        onClick={() => router.push('/assessments')}
                        className="btn-secondary"
                    >
                        Back to Assessments
                    </button>
                </div>
            </div>
        )
    }

    // Results view
    if (result && showFeedback) {
        return (
            <div className="min-h-screen bg-gray-50">
                {/* Results Header */}
                <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-12">
                    <div className="max-w-4xl mx-auto px-4 text-center">
                        <h1 className="text-3xl font-bold mb-2">Assessment Complete!</h1>
                        <p className="text-indigo-100">{assessment.title}</p>

                        {/* Score Circle */}
                        <div className="mt-8 inline-flex items-center justify-center w-32 h-32 rounded-full bg-white/10 backdrop-blur">
                            <div className="text-center">
                                <span className="text-4xl font-bold">{result.percentage}%</span>
                                <p className="text-sm text-indigo-200">Score</p>
                            </div>
                        </div>

                        <div className="mt-6 flex justify-center gap-8">
                            <div>
                                <span className="text-2xl font-bold">{result.correct_count}</span>
                                <p className="text-sm text-indigo-200">Correct</p>
                            </div>
                            <div>
                                <span className="text-2xl font-bold">{result.total_questions - result.correct_count}</span>
                                <p className="text-sm text-indigo-200">Incorrect</p>
                            </div>
                            <div>
                                <span className="text-2xl font-bold">{result.score}/{result.total_marks}</span>
                                <p className="text-sm text-indigo-200">Marks</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Review Questions */}
                <div className="max-w-4xl mx-auto px-4 py-8">
                    <h2 className="text-xl font-bold text-gray-900 mb-6">Review Your Answers</h2>

                    <div className="space-y-6">
                        {assessment.questions.map((question, index) => (
                            <QuestionCard
                                key={question.id}
                                question={question}
                                questionNumber={index + 1}
                                selectedAnswer={answers.get(index) || null}
                                onSelectAnswer={() => { }}
                                isFlagged={flaggedQuestions.has(index)}
                                onToggleFlag={() => { }}
                                showFeedback={true}
                            />
                        ))}
                    </div>

                    <div className="mt-8 text-center">
                        <button
                            onClick={() => router.push('/assessments')}
                            className="btn-primary"
                        >
                            Back to Assessments
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    const currentQ = assessment.questions[currentQuestion]
    const answeredQuestions = new Set(answers.keys())

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b border-gray-200 sticky top-0 z-20">
                <div className="max-w-7xl mx-auto px-4 py-3">
                    <div className="flex items-center justify-between">
                        {/* Left - Title */}
                        <div className="flex items-center gap-4">
                            <button
                                onClick={() => router.push('/assessments')}
                                className="p-2 hover:bg-gray-100 rounded-lg text-gray-500"
                            >
                                <ArrowLeftIcon className="w-5 h-5" />
                            </button>
                            <div>
                                <h1 className="font-semibold text-gray-900">{assessment.title}</h1>
                                <p className="text-sm text-gray-500">
                                    Question {currentQuestion + 1} of {assessment.questions.length}
                                </p>
                            </div>
                        </div>

                        {/* Right - Timer & Navigator Toggle */}
                        <div className="flex items-center gap-4">
                            <AssessmentTimer
                                initialMinutes={assessment.time_limit_minutes}
                                onTimeUp={handleTimeUp}
                            />

                            <button
                                onClick={() => setShowNavigator(!showNavigator)}
                                className={clsx(
                                    "hidden md:flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                                    showNavigator
                                        ? "bg-indigo-100 text-indigo-700"
                                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                                )}
                            >
                                <span className="grid grid-cols-2 gap-0.5">
                                    <span className="w-1.5 h-1.5 bg-current rounded-sm" />
                                    <span className="w-1.5 h-1.5 bg-current rounded-sm" />
                                    <span className="w-1.5 h-1.5 bg-current rounded-sm" />
                                    <span className="w-1.5 h-1.5 bg-current rounded-sm" />
                                </span>
                                Navigator
                            </button>
                        </div>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="h-1 bg-gray-100">
                    <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all"
                        style={{ width: `${((currentQuestion + 1) / assessment.questions.length) * 100}%` }}
                    />
                </div>
            </header>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-4 py-6">
                <div className="flex gap-6">
                    {/* Question Area */}
                    <div className="flex-1">
                        <QuestionCard
                            question={currentQ}
                            questionNumber={currentQuestion + 1}
                            selectedAnswer={answers.get(currentQuestion) || null}
                            onSelectAnswer={handleSelectAnswer}
                            isFlagged={flaggedQuestions.has(currentQuestion)}
                            onToggleFlag={handleToggleFlag}
                            showFeedback={showFeedback}
                        />

                        {/* Navigation Buttons */}
                        <div className="mt-6 flex items-center justify-between">
                            <button
                                onClick={handlePrevious}
                                disabled={currentQuestion === 0}
                                className={clsx(
                                    "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
                                    currentQuestion === 0
                                        ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                                )}
                            >
                                <ArrowLeftIcon className="w-4 h-4" />
                                Previous
                            </button>

                            <div className="flex items-center gap-3">
                                {currentQuestion < assessment.questions.length - 1 ? (
                                    <button
                                        onClick={handleNext}
                                        className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium 
                                            bg-indigo-500 text-white hover:bg-indigo-600 transition-colors"
                                    >
                                        Next
                                        <ArrowRightIcon className="w-4 h-4" />
                                    </button>
                                ) : (
                                    <button
                                        onClick={handleSubmit}
                                        disabled={isSubmitting}
                                        className="flex items-center gap-2 px-6 py-2 rounded-lg font-medium 
                                            bg-green-500 text-white hover:bg-green-600 transition-colors
                                            disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {isSubmitting ? (
                                            <>
                                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                                Submitting...
                                            </>
                                        ) : (
                                            <>
                                                <PaperAirplaneIcon className="w-4 h-4" />
                                                Submit Assessment
                                            </>
                                        )}
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Question Navigator Sidebar */}
                    {showNavigator && (
                        <div className="hidden md:block w-64 flex-shrink-0">
                            <div className="sticky top-24">
                                <QuestionNavigator
                                    totalQuestions={assessment.questions.length}
                                    currentQuestion={currentQuestion}
                                    answeredQuestions={answeredQuestions}
                                    flaggedQuestions={flaggedQuestions}
                                    onQuestionSelect={setCurrentQuestion}
                                />

                                {/* Submit Button in Navigator */}
                                <button
                                    onClick={handleSubmit}
                                    disabled={isSubmitting}
                                    className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-3 
                                        rounded-xl font-medium bg-green-500 text-white hover:bg-green-600 
                                        transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <PaperAirplaneIcon className="w-5 h-5" />
                                    Submit
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
