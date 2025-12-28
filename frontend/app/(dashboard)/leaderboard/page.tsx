'use client'

import { useSession } from 'next-auth/react'
import {
    TrophyIcon,
    FireIcon,
    StarIcon,
    UserCircleIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface LeaderboardEntry {
    rank: number
    username: string
    points: number
    level: number
    streak: number
    isCurrentUser?: boolean
}

const mockLeaderboard: LeaderboardEntry[] = [
    { rank: 1, username: 'math_master', points: 12500, level: 25, streak: 30 },
    { rank: 2, username: 'science_pro', points: 11200, level: 23, streak: 21 },
    { rank: 3, username: 'history_buff', points: 10800, level: 22, streak: 14 },
    { rank: 4, username: 'study_queen', points: 9500, level: 20, streak: 28 },
    { rank: 5, username: 'learning_ninja', points: 8900, level: 19, streak: 12 },
    { rank: 6, username: 'brain_power', points: 8200, level: 18, streak: 9 },
    { rank: 7, username: 'quiz_champion', points: 7800, level: 17, streak: 15 },
    { rank: 8, username: 'you', points: 6500, level: 15, streak: 7, isCurrentUser: true },
    { rank: 9, username: 'smart_cookie', points: 6200, level: 14, streak: 5 },
    { rank: 10, username: 'study_buddy', points: 5800, level: 13, streak: 3 },
]

export default function LeaderboardPage() {
    const { data: session } = useSession()
    const currentUser = mockLeaderboard.find(e => e.isCurrentUser)

    return (
        <div className="space-y-6">
            {/* Current User Position */}
            <div className="card bg-gradient-to-r from-primary-600 to-secondary-600 text-white">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-white/80">Your Current Rank</p>
                        <p className="text-4xl font-bold mt-1">#{currentUser?.rank || '-'}</p>
                    </div>
                    <div className="text-right">
                        <p className="text-white/80">Total Points</p>
                        <p className="text-4xl font-bold mt-1">{currentUser?.points.toLocaleString() || '-'}</p>
                    </div>
                </div>
                <div className="flex gap-6 mt-6 pt-4 border-t border-white/20">
                    <div className="flex items-center gap-2">
                        <StarIcon className="w-5 h-5" />
                        <span>Level {currentUser?.level || '-'}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <FireIcon className="w-5 h-5" />
                        <span>{currentUser?.streak || 0} day streak</span>
                    </div>
                </div>
            </div>

            {/* Top 3 Podium */}
            <div className="flex items-end justify-center gap-4 py-8">
                {/* 2nd Place */}
                <div className="text-center">
                    <div className="w-20 h-20 mx-auto bg-gray-200 rounded-full flex items-center justify-center mb-2">
                        <UserCircleIcon className="w-12 h-12 text-gray-400" />
                    </div>
                    <p className="font-bold text-gray-900">{mockLeaderboard[1]?.username}</p>
                    <p className="text-sm text-gray-500">{mockLeaderboard[1]?.points.toLocaleString()} pts</p>
                    <div className="w-24 h-20 bg-gray-200 mt-2 rounded-t-lg flex items-center justify-center">
                        <span className="text-3xl font-bold text-gray-400">2</span>
                    </div>
                </div>

                {/* 1st Place */}
                <div className="text-center">
                    <div className="relative">
                        <div className="w-24 h-24 mx-auto bg-yellow-100 rounded-full flex items-center justify-center mb-2 ring-4 ring-yellow-400">
                            <UserCircleIcon className="w-16 h-16 text-yellow-500" />
                        </div>
                        <TrophyIcon className="w-8 h-8 text-yellow-500 absolute -top-2 -right-2" />
                    </div>
                    <p className="font-bold text-gray-900">{mockLeaderboard[0]?.username}</p>
                    <p className="text-sm text-gray-500">{mockLeaderboard[0]?.points.toLocaleString()} pts</p>
                    <div className="w-28 h-28 bg-yellow-400 mt-2 rounded-t-lg flex items-center justify-center">
                        <span className="text-4xl font-bold text-white">1</span>
                    </div>
                </div>

                {/* 3rd Place */}
                <div className="text-center">
                    <div className="w-20 h-20 mx-auto bg-orange-100 rounded-full flex items-center justify-center mb-2">
                        <UserCircleIcon className="w-12 h-12 text-orange-400" />
                    </div>
                    <p className="font-bold text-gray-900">{mockLeaderboard[2]?.username}</p>
                    <p className="text-sm text-gray-500">{mockLeaderboard[2]?.points.toLocaleString()} pts</p>
                    <div className="w-20 h-16 bg-orange-300 mt-2 rounded-t-lg flex items-center justify-center">
                        <span className="text-2xl font-bold text-white">3</span>
                    </div>
                </div>
            </div>

            {/* Full Leaderboard */}
            <div className="card">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Global Rankings</h2>
                <div className="divide-y divide-gray-100">
                    {mockLeaderboard.map((entry) => (
                        <div
                            key={entry.rank}
                            className={clsx(
                                'flex items-center gap-4 py-4',
                                entry.isCurrentUser && 'bg-primary-50 -mx-6 px-6 rounded-lg'
                            )}
                        >
                            <div className={clsx(
                                'w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm',
                                entry.rank === 1 ? 'bg-yellow-400 text-white' :
                                    entry.rank === 2 ? 'bg-gray-300 text-white' :
                                        entry.rank === 3 ? 'bg-orange-300 text-white' :
                                            'bg-gray-100 text-gray-600'
                            )}>
                                {entry.rank}
                            </div>

                            <div className="w-10 h-10 bg-gradient-to-br from-primary-400 to-secondary-400 rounded-full flex items-center justify-center text-white font-bold">
                                {entry.username[0].toUpperCase()}
                            </div>

                            <div className="flex-1">
                                <p className={clsx(
                                    'font-medium',
                                    entry.isCurrentUser ? 'text-primary-700' : 'text-gray-900'
                                )}>
                                    {entry.username}
                                    {entry.isCurrentUser && ' (You)'}
                                </p>
                                <div className="flex items-center gap-4 text-sm text-gray-500">
                                    <span className="flex items-center gap-1">
                                        <StarIcon className="w-4 h-4" />
                                        Level {entry.level}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <FireIcon className="w-4 h-4" />
                                        {entry.streak} days
                                    </span>
                                </div>
                            </div>

                            <p className="text-xl font-bold text-gray-900">
                                {entry.points.toLocaleString()}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
