'use client'

import { useState, useEffect } from 'react'
import { CheckIcon, XMarkIcon, TrophyIcon, ClockIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Challenge {
    id: string
    assessment_id: string
    sender_id: string
    sender_name: string
    recipient_name: string
    status: 'pending' | 'accepted' | 'declined' | 'completed'
    sender_score: number | null
    recipient_score: number | null
    challenge_message: string | null
    sent_at: string
    assessment?: {
        title: string
        num_questions: number
        time_limit_minutes: number
        difficulty: string
    }
}

interface ReceivedChallengesProps {
    onAccepted?: (assessment: any) => void
}

export default function ReceivedChallenges({ onAccepted }: ReceivedChallengesProps) {
    const [challenges, setChallenges] = useState<Challenge[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [actionLoading, setActionLoading] = useState<string | null>(null)

    useEffect(() => {
        fetchChallenges()
    }, [])

    const fetchChallenges = async () => {
        try {
            const res = await fetch('/api/assessments/challenges/received')
            if (res.ok) {
                const data = await res.json()
                setChallenges(data.challenges || [])
            }
        } catch (err) {
            console.error('Failed to fetch challenges:', err)
        }
        setIsLoading(false)
    }

    const handleAccept = async (challengeId: string) => {
        setActionLoading(challengeId)
        try {
            const res = await fetch(`/api/assessments/challenges/${challengeId}/accept`, {
                method: 'POST'
            })

            if (res.ok) {
                const data = await res.json()
                fetchChallenges()
                if (onAccepted && data.assessment) {
                    onAccepted(data.assessment)
                }
            }
        } catch (err) {
            console.error('Failed to accept challenge:', err)
        }
        setActionLoading(null)
    }

    const handleDecline = async (challengeId: string) => {
        setActionLoading(challengeId)
        try {
            await fetch(`/api/assessments/challenges/${challengeId}/decline`, {
                method: 'POST'
            })
            fetchChallenges()
        } catch (err) {
            console.error('Failed to decline challenge:', err)
        }
        setActionLoading(null)
    }

    const pendingChallenges = challenges.filter(c => c.status === 'pending')
    const otherChallenges = challenges.filter(c => c.status !== 'pending')

    if (isLoading) {
        return (
            <div className="text-center py-8 text-gray-500">
                Loading challenges...
            </div>
        )
    }

    if (challenges.length === 0) {
        return (
            <div className="text-center py-12">
                <TrophyIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500">No challenges yet</p>
                <p className="text-sm text-gray-400">When someone challenges you, it will appear here</p>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Pending Challenges */}
            {pendingChallenges.length > 0 && (
                <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full animate-pulse" />
                        Pending Challenges ({pendingChallenges.length})
                    </h3>
                    <div className="space-y-3">
                        {pendingChallenges.map(challenge => (
                            <div
                                key={challenge.id}
                                className="bg-gradient-to-r from-orange-50 to-red-50 border border-orange-200 rounded-xl p-4"
                            >
                                <div className="flex items-start justify-between">
                                    <div>
                                        <div className="font-semibold text-gray-900">
                                            {challenge.sender_name} challenged you!
                                        </div>
                                        {challenge.assessment && (
                                            <div className="text-sm text-gray-600 mt-1">
                                                {challenge.assessment.title}
                                                <span className="mx-2">•</span>
                                                {challenge.assessment.num_questions} questions
                                                <span className="mx-2">•</span>
                                                {challenge.assessment.time_limit_minutes} min
                                            </div>
                                        )}
                                        {challenge.challenge_message && (
                                            <div className="mt-2 text-sm italic text-gray-600 bg-white/50 px-3 py-2 rounded-lg">
                                                "{challenge.challenge_message}"
                                            </div>
                                        )}
                                        <div className="text-xs text-gray-400 mt-2 flex items-center gap-1">
                                            <ClockIcon className="w-3 h-3" />
                                            {new Date(challenge.sent_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => handleDecline(challenge.id)}
                                            disabled={actionLoading === challenge.id}
                                            className="p-2 text-gray-500 hover:bg-gray-100 rounded-lg transition-colors"
                                            title="Decline"
                                        >
                                            <XMarkIcon className="w-5 h-5" />
                                        </button>
                                        <button
                                            onClick={() => handleAccept(challenge.id)}
                                            disabled={actionLoading === challenge.id}
                                            className={clsx(
                                                'px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-1',
                                                actionLoading === challenge.id
                                                    ? 'bg-gray-200 text-gray-500'
                                                    : 'bg-orange-500 text-white hover:bg-orange-600'
                                            )}
                                        >
                                            {actionLoading === challenge.id ? (
                                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                            ) : (
                                                <>
                                                    <CheckIcon className="w-4 h-4" />
                                                    Accept
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Past Challenges */}
            {otherChallenges.length > 0 && (
                <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">
                        Challenge History
                    </h3>
                    <div className="space-y-2">
                        {otherChallenges.map(challenge => (
                            <div
                                key={challenge.id}
                                className={clsx(
                                    'border rounded-lg p-3',
                                    challenge.status === 'completed' ? 'bg-green-50 border-green-200' :
                                        challenge.status === 'accepted' ? 'bg-blue-50 border-blue-200' :
                                            'bg-gray-50 border-gray-200'
                                )}
                            >
                                <div className="flex items-center justify-between">
                                    <div>
                                        <span className="font-medium">{challenge.sender_name}</span>
                                        <span className="text-gray-500 mx-1">vs</span>
                                        <span className="font-medium">You</span>
                                        {challenge.assessment && (
                                            <span className="text-sm text-gray-500 ml-2">
                                                • {challenge.assessment.title}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {challenge.status === 'completed' && (
                                            <div className="flex items-center gap-2 text-sm">
                                                <span className={clsx(
                                                    'font-bold',
                                                    (challenge.sender_score || 0) > (challenge.recipient_score || 0)
                                                        ? 'text-green-600' : 'text-gray-600'
                                                )}>
                                                    {challenge.sender_score?.toFixed(0)}%
                                                </span>
                                                <span className="text-gray-400">vs</span>
                                                <span className={clsx(
                                                    'font-bold',
                                                    (challenge.recipient_score || 0) > (challenge.sender_score || 0)
                                                        ? 'text-green-600' : 'text-gray-600'
                                                )}>
                                                    {challenge.recipient_score?.toFixed(0)}%
                                                </span>
                                                {(challenge.recipient_score || 0) > (challenge.sender_score || 0) && (
                                                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                                                        You won!
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                        {challenge.status === 'accepted' && (
                                            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                                                In Progress
                                            </span>
                                        )}
                                        {challenge.status === 'declined' && (
                                            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
                                                Declined
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
