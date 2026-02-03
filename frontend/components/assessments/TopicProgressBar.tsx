'use client'

/**
 * TopicProgressBar - Shows question progress for a topic
 * 
 * Displays:
 * - Progress bar showing attempted/total questions
 * - Percentage text
 * - Auto-generation indicator when 80%+ complete
 */

interface TopicProgressBarProps {
    topicId: string
    topicName: string
    questionsAttempted: number
    totalQuestions: number
    shouldGenerate?: boolean
    compact?: boolean
}

export default function TopicProgressBar({
    topicId,
    topicName,
    questionsAttempted,
    totalQuestions,
    shouldGenerate = false,
    compact = false
}: TopicProgressBarProps) {
    const percentage = totalQuestions > 0
        ? Math.round((questionsAttempted / totalQuestions) * 100)
        : 0

    const getProgressColor = () => {
        if (percentage >= 80) return 'bg-green-500'
        if (percentage >= 50) return 'bg-yellow-500'
        if (percentage >= 25) return 'bg-orange-500'
        return 'bg-gray-300'
    }

    const getTextColor = () => {
        if (percentage >= 80) return 'text-green-700'
        if (percentage >= 50) return 'text-yellow-700'
        return 'text-gray-600'
    }

    if (compact) {
        return (
            <div className="flex items-center gap-2 text-xs">
                <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                        className={`h-full ${getProgressColor()} transition-all duration-300`}
                        style={{ width: `${percentage}%` }}
                    />
                </div>
                <span className={`font-medium ${getTextColor()}`}>
                    {questionsAttempted}/{totalQuestions}
                </span>
                {shouldGenerate && (
                    <span className="text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">
                        🔄
                    </span>
                )}
            </div>
        )
    }

    return (
        <div className="px-3 py-2 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-700 truncate" title={topicName}>
                    {topicName}
                </span>
                <span className={`text-xs font-medium ${getTextColor()}`}>
                    {questionsAttempted}/{totalQuestions} ({percentage}%)
                </span>
            </div>

            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                    className={`h-full ${getProgressColor()} transition-all duration-300`}
                    style={{ width: `${percentage}%` }}
                />
            </div>

            {shouldGenerate && (
                <div className="mt-1.5 flex items-center gap-1.5 text-xs text-blue-600">
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    <span>Auto-generating new questions...</span>
                </div>
            )}

            {totalQuestions === 0 && (
                <div className="mt-1.5 text-xs text-gray-500 italic">
                    No questions available yet
                </div>
            )}
        </div>
    )
}
