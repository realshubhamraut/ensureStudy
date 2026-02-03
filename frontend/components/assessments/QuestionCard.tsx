'use client'

import { useState } from 'react'
import clsx from 'clsx'
import { CheckCircleIcon, XCircleIcon, FlagIcon } from '@heroicons/react/24/solid'
import { FlagIcon as FlagOutlineIcon, LightBulbIcon } from '@heroicons/react/24/outline'

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

interface QuestionCardProps {
    question: Question
    questionNumber: number
    selectedAnswer: string | null
    onSelectAnswer: (answer: string) => void
    isFlagged: boolean
    onToggleFlag: () => void
    showFeedback: boolean
}

export default function QuestionCard({
    question,
    questionNumber,
    selectedAnswer,
    onSelectAnswer,
    isFlagged,
    onToggleFlag,
    showFeedback
}: QuestionCardProps) {
    const [showExplanation, setShowExplanation] = useState(false)

    const getCorrectAnswer = () => {
        return question.options?.find(opt => opt.is_correct)?.id || null
    }

    const isCorrect = showFeedback && selectedAnswer === getCorrectAnswer()
    const isIncorrect = showFeedback && selectedAnswer && selectedAnswer !== getCorrectAnswer()

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
            {/* Question Header */}
            <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-3">
                    <span className="w-10 h-10 rounded-full bg-indigo-100 text-indigo-700 
                        font-bold flex items-center justify-center text-lg">
                        {questionNumber}
                    </span>
                    <div>
                        <span className={clsx(
                            "text-xs font-medium px-2 py-1 rounded-full",
                            question.difficulty === 'easy' && "bg-green-100 text-green-700",
                            question.difficulty === 'medium' && "bg-yellow-100 text-yellow-700",
                            question.difficulty === 'hard' && "bg-red-100 text-red-700"
                        )}>
                            {question.difficulty.charAt(0).toUpperCase() + question.difficulty.slice(1)}
                        </span>
                        <span className="ml-2 text-xs text-gray-500">
                            {question.marks} mark{question.marks > 1 ? 's' : ''}
                        </span>
                    </div>
                </div>

                <button
                    onClick={onToggleFlag}
                    className={clsx(
                        "p-2 rounded-lg transition-colors",
                        isFlagged
                            ? "bg-orange-100 text-orange-600"
                            : "hover:bg-gray-100 text-gray-400"
                    )}
                    title={isFlagged ? "Unflag question" : "Flag for review"}
                >
                    {isFlagged ? (
                        <FlagIcon className="w-5 h-5" />
                    ) : (
                        <FlagOutlineIcon className="w-5 h-5" />
                    )}
                </button>
            </div>

            {/* Question Text */}
            <p className="text-lg text-gray-800 leading-relaxed mb-6">
                {question.question_text}
            </p>

            {/* Options */}
            {question.question_type === 'mcq' && question.options && (
                <div className="space-y-3">
                    {question.options.map((option) => {
                        const isSelected = selectedAnswer === option.id
                        const isOptionCorrect = showFeedback && option.is_correct
                        const isOptionIncorrect = showFeedback && isSelected && !option.is_correct

                        return (
                            <button
                                key={option.id}
                                onClick={() => !showFeedback && onSelectAnswer(option.id)}
                                disabled={showFeedback}
                                className={clsx(
                                    "w-full flex items-start gap-4 p-4 rounded-xl border-2 transition-all text-left",
                                    !showFeedback && isSelected && "border-indigo-500 bg-indigo-50",
                                    !showFeedback && !isSelected && "border-gray-200 hover:border-gray-300 hover:bg-gray-50",
                                    isOptionCorrect && "border-green-500 bg-green-50",
                                    isOptionIncorrect && "border-red-500 bg-red-50",
                                    showFeedback && !isOptionCorrect && !isOptionIncorrect && "border-gray-200 opacity-60"
                                )}
                            >
                                {/* Option Letter Badge */}
                                <span className={clsx(
                                    "w-8 h-8 rounded-lg flex items-center justify-center font-bold flex-shrink-0",
                                    !showFeedback && isSelected && "bg-indigo-500 text-white",
                                    !showFeedback && !isSelected && "bg-gray-100 text-gray-600",
                                    isOptionCorrect && "bg-green-500 text-white",
                                    isOptionIncorrect && "bg-red-500 text-white"
                                )}>
                                    {option.id}
                                </span>

                                <div className="flex-1">
                                    <p className={clsx(
                                        "text-gray-800",
                                        isOptionCorrect && "text-green-800 font-medium",
                                        isOptionIncorrect && "text-red-800"
                                    )}>
                                        {option.text}
                                    </p>

                                    {/* Per-Option Explanation */}
                                    {showFeedback && option.explanation && (isOptionCorrect || isOptionIncorrect) && (
                                        <p className={clsx(
                                            "mt-2 text-sm",
                                            isOptionCorrect && "text-green-600",
                                            isOptionIncorrect && "text-red-600"
                                        )}>
                                            💡 {option.explanation}
                                        </p>
                                    )}
                                </div>

                                {/* Status Icon */}
                                {showFeedback && (
                                    <div className="flex-shrink-0">
                                        {isOptionCorrect && (
                                            <CheckCircleIcon className="w-6 h-6 text-green-500" />
                                        )}
                                        {isOptionIncorrect && (
                                            <XCircleIcon className="w-6 h-6 text-red-500" />
                                        )}
                                    </div>
                                )}
                            </button>
                        )
                    })}
                </div>
            )}

            {/* Show All Explanations Button */}
            {showFeedback && question.options && (
                <div className="mt-4">
                    <button
                        onClick={() => setShowExplanation(!showExplanation)}
                        className="flex items-center gap-2 text-sm text-indigo-600 hover:text-indigo-700"
                    >
                        <LightBulbIcon className="w-4 h-4" />
                        {showExplanation ? "Hide all explanations" : "Show all explanations"}
                    </button>

                    {showExplanation && (
                        <div className="mt-3 p-4 bg-indigo-50 rounded-xl border border-indigo-100">
                            <p className="text-sm font-semibold text-indigo-800 mb-2">
                                Why each option is correct/incorrect:
                            </p>
                            <ul className="space-y-2">
                                {question.options.map((opt) => (
                                    <li key={opt.id} className="text-sm">
                                        <span className={clsx(
                                            "font-medium",
                                            opt.is_correct ? "text-green-700" : "text-gray-700"
                                        )}>
                                            {opt.id}) {opt.text}
                                        </span>
                                        {opt.explanation && (
                                            <span className="text-gray-600">
                                                {" — "}{opt.explanation}
                                            </span>
                                        )}
                                        {opt.is_correct && (
                                            <span className="ml-1 text-green-600 font-medium">✓ Correct</span>
                                        )}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {/* Feedback Banner */}
            {showFeedback && (
                <div className={clsx(
                    "mt-6 p-4 rounded-xl flex items-center gap-3",
                    isCorrect && "bg-green-100 text-green-800",
                    isIncorrect && "bg-red-100 text-red-800"
                )}>
                    {isCorrect ? (
                        <>
                            <CheckCircleIcon className="w-6 h-6 text-green-600" />
                            <span className="font-medium">Correct! Well done.</span>
                        </>
                    ) : (
                        <>
                            <XCircleIcon className="w-6 h-6 text-red-600" />
                            <span className="font-medium">
                                Incorrect. The correct answer is {getCorrectAnswer()}.
                            </span>
                        </>
                    )}
                </div>
            )}
        </div>
    )
}
