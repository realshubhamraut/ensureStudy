'use client'

import { useState, useEffect } from 'react'
import { SparklesIcon, ArrowPathIcon, CheckCircleIcon } from '@heroicons/react/24/outline'
import { getApiBaseUrl } from '@/utils/api'

/**
 * LearningAgentStatus - Shows the Type 5 Learning Agent status
 * 
 * Displays:
 * - Active status badge
 * - Learning iterations count
 * - Calibrated difficulty
 * - Manual generation trigger button
 */

interface LearningAgentStatus {
    active: boolean
    learning_iterations: number
    calibrated_difficulty: number
    last_learning_at: string | null
}

interface TopicProgress {
    topic_id: string
    topic_name: string
    total_questions: number
    questions_attempted: number
    attempt_percentage: number
    should_generate: boolean
    learning_agent: {
        iterations: number
        difficulty: number
    }
    mastery_percentage?: number  // From StudentTopicScore - actual confidence
}

interface LearningAgentStatusProps {
    classroomId?: string
    selectedTopics?: string[]
    onGenerationComplete?: () => void
}

export default function LearningAgentStatus({
    classroomId,
    selectedTopics = [],
    onGenerationComplete
}: LearningAgentStatusProps) {
    const [progress, setProgress] = useState<TopicProgress[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [isGenerating, setIsGenerating] = useState(false)
    const [generatingTopic, setGeneratingTopic] = useState<string | null>(null)

    useEffect(() => {
        if (!classroomId) {
            setProgress([])
            return
        }

        const fetchProgress = async () => {
            setIsLoading(true)
            try {
                const token = typeof window !== 'undefined' ? localStorage.getItem('accessToken') : null

                // Fetch question progress
                const res = await fetch(`${getApiBaseUrl()}/api/questions/progress/classroom/${classroomId}`, {
                    headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                })

                // Also fetch topic mastery for confidence scores
                const masteryRes = await fetch(`${getApiBaseUrl()}/api/progress/topic-mastery?classroom_id=${classroomId}`, {
                    headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                })

                // Build mastery map
                const masteryMap: Record<string, number> = {}
                if (masteryRes.ok) {
                    const masteryData = await masteryRes.json()
                    for (const t of (masteryData.topics || [])) {
                        masteryMap[t.topic_id] = t.mastery_level || 0
                    }
                }

                if (res.ok) {
                    const data = await res.json()
                    // Merge mastery percentages into progress data
                    const progressWithMastery = (data.topics || []).map((p: TopicProgress) => ({
                        ...p,
                        mastery_percentage: masteryMap[p.topic_id] || 0
                    }))
                    setProgress(progressWithMastery)
                }
            } catch (err) {
                console.error('Failed to fetch progress:', err)
            }
            setIsLoading(false)
        }

        fetchProgress()
    }, [classroomId])

    const handleTriggerGeneration = async (topicId: string) => {
        setIsGenerating(true)
        setGeneratingTopic(topicId)

        try {
            const token = typeof window !== 'undefined' ? localStorage.getItem('accessToken') : null
            const res = await fetch(`${getApiBaseUrl()}/api/questions/trigger-generation/${topicId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                }
            })

            if (res.ok) {
                const data = await res.json()
                console.log('Generation result:', data)
                onGenerationComplete?.()

                // Refresh progress
                setTimeout(() => {
                    const fetchProgress = async () => {
                        const res = await fetch(`${getApiBaseUrl()}/api/questions/progress/classroom/${classroomId}`, {
                            headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                        })
                        if (res.ok) {
                            const data = await res.json()
                            setProgress(data.topics || [])
                        }
                    }
                    fetchProgress()
                }, 1000)
            }
        } catch (err) {
            console.error('Failed to trigger generation:', err)
        }

        setIsGenerating(false)
        setGeneratingTopic(null)
    }

    // Filter to selected topics if any
    const displayedProgress = selectedTopics.length > 0
        ? progress.filter(p => selectedTopics.includes(p.topic_id))
        : progress

    const topicsAtThreshold = displayedProgress.filter(p => p.should_generate)
    const avgLearningIterations = displayedProgress.length > 0
        ? Math.round(displayedProgress.reduce((sum, p) => sum + p.learning_agent.iterations, 0) / displayedProgress.length)
        : 0

    return (
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl p-4 border border-purple-100">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-purple-100 rounded-lg">
                        <SparklesIcon className="w-4 h-4 text-purple-600" />
                    </div>
                    <span className="font-semibold text-gray-800">Learning Agent</span>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                        Active
                    </span>
                </div>
                {avgLearningIterations > 0 && (
                    <span className="text-xs text-gray-500">
                        🧠 {avgLearningIterations} learning cycles
                    </span>
                )}
            </div>

            {/* Loading state */}
            {isLoading && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                    <ArrowPathIcon className="w-4 h-4 animate-spin" />
                    Loading progress...
                </div>
            )}

            {/* Topics at threshold */}
            {topicsAtThreshold.length > 0 && (
                <div className="mb-3 p-2 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="text-xs font-medium text-blue-800 mb-1">
                        📊 {topicsAtThreshold.length} topic(s) at 80%+ threshold
                    </div>
                    <div className="text-xs text-blue-600">
                        New questions will be auto-generated when you create this assessment.
                    </div>
                </div>
            )}

            {/* Progress summary */}
            {displayedProgress.length > 0 && (
                <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-600 mb-1">
                        Topic Progress & Mastery
                    </div>
                    <div className="max-h-32 overflow-y-auto space-y-1.5">
                        {displayedProgress.slice(0, 5).map(topic => {
                            const mastery = topic.mastery_percentage || 0
                            const masteryColor = mastery >= 70 ? 'bg-green-500' :
                                mastery >= 50 ? 'bg-yellow-500' :
                                    mastery > 0 ? 'bg-red-400' : 'bg-gray-300'
                            return (
                                <div
                                    key={topic.topic_id}
                                    className="flex items-center gap-2 text-xs bg-white/50 rounded-lg px-2 py-1.5"
                                >
                                    <div className="flex-1 truncate" title={topic.topic_name}>
                                        {topic.topic_name}
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        {/* Mastery bar (confidence) */}
                                        <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden" title={`Mastery: ${Math.round(mastery)}%`}>
                                            <div
                                                className={`h-full transition-all ${masteryColor}`}
                                                style={{ width: `${mastery}%` }}
                                            />
                                        </div>
                                        <span className={`w-10 font-medium ${mastery >= 70 ? 'text-green-600' :
                                                mastery >= 50 ? 'text-yellow-600' :
                                                    mastery > 0 ? 'text-red-500' : 'text-gray-400'
                                            }`}>
                                            {mastery > 0 ? `${Math.round(mastery)}%` : 'New'}
                                        </span>
                                        {topic.should_generate && (
                                            <button
                                                onClick={() => handleTriggerGeneration(topic.topic_id)}
                                                disabled={isGenerating}
                                                className="text-blue-600 hover:text-blue-800 disabled:opacity-50"
                                                title="Generate more questions"
                                            >
                                                {generatingTopic === topic.topic_id ? (
                                                    <ArrowPathIcon className="w-3.5 h-3.5 animate-spin" />
                                                ) : (
                                                    <SparklesIcon className="w-3.5 h-3.5" />
                                                )}
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                    {displayedProgress.length > 5 && (
                        <div className="text-xs text-gray-500 text-center">
                            +{displayedProgress.length - 5} more topics
                        </div>
                    )}
                </div>
            )}

            {/* Empty state */}
            {!isLoading && displayedProgress.length === 0 && classroomId && (
                <div className="text-xs text-gray-500 text-center py-2">
                    No topic progress data available
                </div>
            )}

            {/* Info footer */}
            <div className="mt-3 pt-2 border-t border-purple-100 text-[10px] text-gray-500">
                💡 The Learning Agent automatically generates new questions when you've completed 80% of available questions for a topic.
            </div>
        </div>
    )
}
