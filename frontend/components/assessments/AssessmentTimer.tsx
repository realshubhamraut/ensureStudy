'use client'

import { useEffect, useState } from 'react'
import clsx from 'clsx'

interface AssessmentTimerProps {
    initialMinutes: number
    onTimeUp: () => void
    isPaused?: boolean
}

export default function AssessmentTimer({
    initialMinutes,
    onTimeUp,
    isPaused = false
}: AssessmentTimerProps) {
    const [secondsRemaining, setSecondsRemaining] = useState(initialMinutes * 60)

    useEffect(() => {
        if (isPaused || secondsRemaining <= 0) return

        const interval = setInterval(() => {
            setSecondsRemaining(prev => {
                if (prev <= 1) {
                    clearInterval(interval)
                    onTimeUp()
                    return 0
                }
                return prev - 1
            })
        }, 1000)

        return () => clearInterval(interval)
    }, [isPaused, secondsRemaining, onTimeUp])

    const minutes = Math.floor(secondsRemaining / 60)
    const seconds = secondsRemaining % 60

    const isWarning = secondsRemaining <= 300 // 5 minutes
    const isCritical = secondsRemaining <= 60 // 1 minute

    return (
        <div className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-xl font-mono text-lg font-semibold transition-all",
            isCritical && "bg-red-100 text-red-700 animate-pulse",
            isWarning && !isCritical && "bg-orange-100 text-orange-700",
            !isWarning && "bg-gray-100 text-gray-700"
        )}>
            <svg
                className={clsx("w-5 h-5", isCritical && "animate-bounce")}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
            >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>
                {String(minutes).padStart(2, '0')}:{String(seconds).padStart(2, '0')}
            </span>
        </div>
    )
}
