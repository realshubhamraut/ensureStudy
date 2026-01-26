'use client'

import { useState, useEffect } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import {
    EyeIcon,
    SpeakerWaveIcon,
    BookOpenIcon,
    HandRaisedIcon,
    CheckCircleIcon,
    ArrowRightIcon
} from '@heroicons/react/24/outline'

interface Question {
    id: string
    text: string
    options: { id: string; text: string; style: string }[]
}

interface Props {
    onComplete: (primary: string, secondary: string | null) => void
    onSkip?: () => void
}

const styleIcons: Record<string, any> = {
    visual: EyeIcon,
    auditory: SpeakerWaveIcon,
    reading: BookOpenIcon,
    kinesthetic: HandRaisedIcon
}

const styleColors: Record<string, string> = {
    visual: 'from-blue-500 to-cyan-500',
    auditory: 'from-purple-500 to-pink-500',
    reading: 'from-green-500 to-emerald-500',
    kinesthetic: 'from-orange-500 to-red-500'
}

export default function LearningStyleQuiz({ onComplete, onSkip }: Props) {
    const [quiz, setQuiz] = useState<{ title: string; description: string; questions: Question[] } | null>(null)
    const [currentQuestion, setCurrentQuestion] = useState(0)
    const [responses, setResponses] = useState<Record<string, string>>({})
    const [result, setResult] = useState<{ primary: string; secondary: string | null; description: string; tips: string[] } | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchQuiz()
    }, [])

    const fetchQuiz = async () => {
        try {
            const res = await fetch(`${getAiServiceUrl()}/api/curriculum/learning-style/quiz`)
            if (res.ok) {
                const data = await res.json()
                setQuiz(data.quiz)
            }
        } catch (error) {
            console.error('Failed to fetch quiz:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleAnswer = (questionId: string, style: string) => {
        const newResponses = { ...responses, [questionId]: style }
        setResponses(newResponses)

        if (quiz && currentQuestion < quiz.questions.length - 1) {
            setTimeout(() => setCurrentQuestion(currentQuestion + 1), 300)
        }
    }

    const submitQuiz = async () => {
        try {
            const userId = localStorage.getItem('userId') || 'demo-user'
            const res = await fetch(`${getAiServiceUrl()}/api/curriculum/learning-style/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, responses })
            })

            if (res.ok) {
                const data = await res.json()
                setResult({
                    primary: data.primary_style,
                    secondary: data.secondary_style,
                    description: data.description,
                    tips: data.study_tips
                })
                onComplete(data.primary_style, data.secondary_style)
            }
        } catch (error) {
            console.error('Failed to submit quiz:', error)
        }
    }

    if (loading) {
        return <div className="flex items-center justify-center h-64"><div className="spinner"></div></div>
    }

    if (!quiz) {
        return <div className="text-center text-gray-500">Failed to load quiz</div>
    }

    if (result) {
        const Icon = styleIcons[result.primary] || EyeIcon
        return (
            <div className="text-center space-y-6">
                <div className={`w-20 h-20 mx-auto rounded-full bg-gradient-to-br ${styleColors[result.primary]} flex items-center justify-center`}>
                    <Icon className="w-10 h-10 text-white" />
                </div>
                <div>
                    <h3 className="text-2xl font-bold text-gray-900 capitalize">{result.primary} Learner</h3>
                    {result.secondary && (
                        <p className="text-gray-500 mt-1">with <span className="capitalize">{result.secondary}</span> tendencies</p>
                    )}
                </div>
                <p className="text-gray-600 max-w-md mx-auto">{result.description}</p>
                <div className="bg-gray-50 rounded-xl p-4 text-left max-w-md mx-auto">
                    <h4 className="font-medium text-gray-900 mb-2">Study Tips for You:</h4>
                    <ul className="space-y-1">
                        {result.tips.map((tip, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                                <CheckCircleIcon className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                                {tip}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        )
    }

    const question = quiz.questions[currentQuestion]
    const progress = ((currentQuestion + 1) / quiz.questions.length) * 100
    const allAnswered = Object.keys(responses).length === quiz.questions.length

    return (
        <div className="space-y-6">
            <div>
                <h3 className="text-xl font-bold text-gray-900">{quiz.title}</h3>
                <p className="text-gray-500 text-sm mt-1">{quiz.description}</p>
            </div>

            {/* Progress */}
            <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-500">
                    <span>Question {currentQuestion + 1} of {quiz.questions.length}</span>
                    <span>{Math.round(progress)}%</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full">
                    <div className="h-full bg-primary-500 rounded-full transition-all" style={{ width: `${progress}%` }} />
                </div>
            </div>

            {/* Question */}
            <div className="py-4">
                <p className="text-lg font-medium text-gray-900 mb-4">{question.text}</p>
                <div className="space-y-3">
                    {question.options.map(option => (
                        <button
                            key={option.id}
                            onClick={() => handleAnswer(question.id, option.style)}
                            className={`w-full text-left p-4 rounded-xl border-2 transition-all ${responses[question.id] === option.style
                                    ? 'border-primary-500 bg-primary-50'
                                    : 'border-gray-200 hover:border-gray-300'
                                }`}
                        >
                            {option.text}
                        </button>
                    ))}
                </div>
            </div>

            {/* Navigation */}
            <div className="flex justify-between">
                <button
                    onClick={() => setCurrentQuestion(Math.max(0, currentQuestion - 1))}
                    disabled={currentQuestion === 0}
                    className="text-gray-500 hover:text-gray-700 disabled:opacity-50"
                >
                    ← Previous
                </button>
                {allAnswered ? (
                    <button onClick={submitQuiz} className="btn-primary flex items-center gap-2">
                        See Results <ArrowRightIcon className="w-4 h-4" />
                    </button>
                ) : (
                    <button
                        onClick={() => setCurrentQuestion(Math.min(quiz.questions.length - 1, currentQuestion + 1))}
                        disabled={!responses[question.id]}
                        className="text-primary-600 hover:text-primary-700 disabled:opacity-50"
                    >
                        Next →
                    </button>
                )}
            </div>

            {onSkip && (
                <button onClick={onSkip} className="w-full text-center text-sm text-gray-400 hover:text-gray-600">
                    Skip for now
                </button>
            )}
        </div>
    )
}
