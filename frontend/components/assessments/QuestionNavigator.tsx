'use client'

import clsx from 'clsx'

interface QuestionNavigatorProps {
    totalQuestions: number
    currentQuestion: number
    answeredQuestions: Set<number>
    flaggedQuestions: Set<number>
    onQuestionSelect: (index: number) => void
}

export default function QuestionNavigator({
    totalQuestions,
    currentQuestion,
    answeredQuestions,
    flaggedQuestions,
    onQuestionSelect
}: QuestionNavigatorProps) {
    return (
        <div className="bg-white rounded-2xl shadow-lg p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Questions</h3>

            <div className="grid grid-cols-5 gap-2">
                {Array.from({ length: totalQuestions }, (_, i) => {
                    const isAnswered = answeredQuestions.has(i)
                    const isFlagged = flaggedQuestions.has(i)
                    const isCurrent = currentQuestion === i

                    return (
                        <button
                            key={i}
                            onClick={() => onQuestionSelect(i)}
                            className={clsx(
                                "w-9 h-9 rounded-lg text-sm font-medium transition-all",
                                "flex items-center justify-center relative",
                                isCurrent && "ring-2 ring-indigo-500 ring-offset-2",
                                isAnswered && !isCurrent && "bg-green-500 text-white",
                                isFlagged && !isAnswered && "bg-yellow-400 text-yellow-900",
                                !isAnswered && !isFlagged && !isCurrent && "bg-gray-100 text-gray-600 hover:bg-gray-200",
                                isCurrent && !isAnswered && "bg-indigo-500 text-white"
                            )}
                        >
                            {i + 1}
                            {isFlagged && (
                                <span className="absolute -top-1 -right-1 w-2 h-2 bg-orange-500 rounded-full" />
                            )}
                        </button>
                    )
                })}
            </div>

            {/* Legend */}
            <div className="mt-4 pt-3 border-t border-gray-100 space-y-1">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span className="w-3 h-3 rounded bg-green-500" />
                    <span>Answered</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span className="w-3 h-3 rounded bg-yellow-400" />
                    <span>Flagged for review</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span className="w-3 h-3 rounded bg-gray-100 border border-gray-300" />
                    <span>Not answered</span>
                </div>
            </div>
        </div>
    )
}
