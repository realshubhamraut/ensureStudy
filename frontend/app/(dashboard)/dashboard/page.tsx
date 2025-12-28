'use client'

import { useSession } from 'next-auth/react'
import Link from 'next/link'
import {
    ChatBubbleLeftRightIcon,
    BookOpenIcon,
    ClipboardDocumentListIcon,
    TrophyIcon,
    FireIcon,
    AcademicCapIcon,
    ArrowTrendingUpIcon
} from '@heroicons/react/24/outline'

export default function DashboardPage() {
    const { data: session } = useSession()

    return (
        <div className="space-y-8">
            {/* Welcome Section */}
            <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-2xl p-8 text-white">
                <h1 className="text-3xl font-bold mb-2">
                    Welcome back, {session?.user?.username || 'Student'}! 👋
                </h1>
                <p className="text-white/80 text-lg">
                    Ready to continue your learning journey? Let&apos;s make today productive.
                </p>
                <div className="flex gap-4 mt-6">
                    <Link href="/chat" className="px-6 py-3 bg-white text-primary-600 rounded-lg font-medium hover:bg-gray-100 transition-colors">
                        Start Learning
                    </Link>
                    <Link href="/assessments" className="px-6 py-3 bg-white/20 text-white rounded-lg font-medium hover:bg-white/30 transition-colors">
                        Take Assessment
                    </Link>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                    icon={<FireIcon className="w-6 h-6" />}
                    label="Study Streak"
                    value="7 days"
                    change="+2 from last week"
                    color="orange"
                />
                <StatCard
                    icon={<AcademicCapIcon className="w-6 h-6" />}
                    label="Topics Mastered"
                    value="12"
                    change="3 this week"
                    color="green"
                />
                <StatCard
                    icon={<ClipboardDocumentListIcon className="w-6 h-6" />}
                    label="Assessments"
                    value="24"
                    change="Avg: 82%"
                    color="blue"
                />
                <StatCard
                    icon={<TrophyIcon className="w-6 h-6" />}
                    label="Global Rank"
                    value="#42"
                    change="↑ 5 positions"
                    color="purple"
                />
            </div>

            {/* Quick Actions */}
            <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <QuickActionCard
                        href="/chat"
                        icon={<ChatBubbleLeftRightIcon className="w-8 h-8" />}
                        title="Ask AI Tutor"
                        description="Get instant help with any topic"
                        gradient="from-blue-500 to-cyan-500"
                    />
                    <QuickActionCard
                        href="/notes"
                        icon={<BookOpenIcon className="w-8 h-8" />}
                        title="Study Notes"
                        description="Review AI-generated notes"
                        gradient="from-purple-500 to-pink-500"
                    />
                    <QuickActionCard
                        href="/assessments"
                        icon={<ClipboardDocumentListIcon className="w-8 h-8" />}
                        title="Practice Quiz"
                        description="Test your knowledge"
                        gradient="from-green-500 to-emerald-500"
                    />
                </div>
            </div>

            {/* Weak Topics Alert */}
            <div className="card border-l-4 border-warning-500">
                <div className="flex items-start gap-4">
                    <div className="p-3 bg-warning-100 rounded-lg">
                        <ArrowTrendingUpIcon className="w-6 h-6 text-warning-600" />
                    </div>
                    <div>
                        <h3 className="font-bold text-gray-900">Topics Needing Attention</h3>
                        <p className="text-gray-600 mt-1">
                            Based on your recent assessments, you might want to review these topics:
                        </p>
                        <div className="flex flex-wrap gap-2 mt-3">
                            <span className="px-3 py-1 bg-warning-100 text-warning-700 rounded-full text-sm font-medium">
                                Photosynthesis
                            </span>
                            <span className="px-3 py-1 bg-warning-100 text-warning-700 rounded-full text-sm font-medium">
                                Quadratic Equations
                            </span>
                            <span className="px-3 py-1 bg-warning-100 text-warning-700 rounded-full text-sm font-medium">
                                World War II
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Activity */}
            <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Activity</h2>
                <div className="card divide-y divide-gray-100">
                    <ActivityItem
                        icon={<ChatBubbleLeftRightIcon className="w-5 h-5" />}
                        title="Asked about mitochondria function"
                        time="2 hours ago"
                        color="blue"
                    />
                    <ActivityItem
                        icon={<ClipboardDocumentListIcon className="w-5 h-5" />}
                        title="Completed Biology Quiz - 85%"
                        time="Yesterday"
                        color="green"
                    />
                    <ActivityItem
                        icon={<BookOpenIcon className="w-5 h-5" />}
                        title="Reviewed Calculus notes"
                        time="2 days ago"
                        color="purple"
                    />
                </div>
            </div>
        </div>
    )
}

function StatCard({
    icon,
    label,
    value,
    change,
    color
}: {
    icon: React.ReactNode
    label: string
    value: string
    change: string
    color: 'orange' | 'green' | 'blue' | 'purple'
}) {
    const colors = {
        orange: 'bg-orange-100 text-orange-600',
        green: 'bg-green-100 text-green-600',
        blue: 'bg-blue-100 text-blue-600',
        purple: 'bg-purple-100 text-purple-600',
    }

    return (
        <div className="card">
            <div className={`inline-flex p-3 rounded-lg ${colors[color]} mb-4`}>
                {icon}
            </div>
            <p className="text-gray-500 text-sm">{label}</p>
            <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
            <p className="text-sm text-gray-500 mt-1">{change}</p>
        </div>
    )
}

function QuickActionCard({
    href,
    icon,
    title,
    description,
    gradient
}: {
    href: string
    icon: React.ReactNode
    title: string
    description: string
    gradient: string
}) {
    return (
        <Link href={href} className="card-hover group flex items-start gap-4">
            <div className={`p-4 rounded-xl bg-gradient-to-br ${gradient} text-white group-hover:scale-110 transition-transform`}>
                {icon}
            </div>
            <div>
                <h3 className="font-bold text-gray-900">{title}</h3>
                <p className="text-gray-600 text-sm">{description}</p>
            </div>
        </Link>
    )
}

function ActivityItem({
    icon,
    title,
    time,
    color
}: {
    icon: React.ReactNode
    title: string
    time: string
    color: 'blue' | 'green' | 'purple'
}) {
    const colors = {
        blue: 'bg-blue-100 text-blue-600',
        green: 'bg-green-100 text-green-600',
        purple: 'bg-purple-100 text-purple-600',
    }

    return (
        <div className="flex items-center gap-4 py-4">
            <div className={`p-2 rounded-lg ${colors[color]}`}>
                {icon}
            </div>
            <div className="flex-1">
                <p className="font-medium text-gray-900">{title}</p>
            </div>
            <p className="text-sm text-gray-500">{time}</p>
        </div>
    )
}
