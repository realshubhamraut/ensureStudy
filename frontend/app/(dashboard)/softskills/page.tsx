'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
    AcademicCapIcon,
    ChatBubbleLeftRightIcon,
    MicrophoneIcon,
    VideoCameraIcon,
    ArrowRightIcon,
    SparklesIcon,
    UserGroupIcon,
    ChartBarIcon
} from '@heroicons/react/24/outline'

export default function SoftSkillsPage() {
    const [selectedMode, setSelectedMode] = useState<'interview' | 'communication' | null>(null)

    const modes = [
        {
            id: 'interview' as const,
            title: 'Mock Interview',
            subtitle: 'Subject Knowledge Assessment',
            description: 'Practice academic interviews with an AI avatar. Get evaluated on your understanding of topics in Math, Physics, and Chemistry.',
            icon: AcademicCapIcon,
            features: [
                'AI avatar asks subject questions verbally',
                'Answer via voice in real-time',
                'Get instant concept-by-concept scoring',
                'Identify weak topics for improvement'
            ],
            color: 'from-blue-500 to-indigo-600',
            hoverColor: 'hover:from-blue-600 hover:to-indigo-700',
            href: '/softskills/mock-interview'
        },
        {
            id: 'communication' as const,
            title: 'Soft Skills',
            subtitle: 'Communication Assessment',
            description: 'Evaluate your speaking skills, body language, and presentation abilities with real-time AI analysis.',
            icon: ChatBubbleLeftRightIcon,
            features: [
                'Fluency & grammar analysis',
                'Eye contact & posture tracking',
                'Confidence level assessment',
                'Detailed improvement suggestions'
            ],
            color: 'from-purple-500 to-pink-600',
            hoverColor: 'hover:from-purple-600 hover:to-pink-700',
            href: '/softskills/communication'
        }
    ]

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
            {/* Header */}
            <div className="bg-white border-b border-gray-200 px-6 py-8">
                <div className="max-w-5xl mx-auto">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <UserGroupIcon className="w-7 h-7 text-white" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900">Soft Skills AI</h1>
                            <p className="text-gray-500 mt-1">Practice interviews and improve communication skills with AI-powered feedback</p>
                        </div>
                    </div>

                    {/* Stats Preview */}
                    <div className="flex items-center gap-6 mt-6">
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                            <MicrophoneIcon className="w-4 h-4 text-indigo-500" />
                            <span>Voice-based interaction</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                            <VideoCameraIcon className="w-4 h-4 text-purple-500" />
                            <span>Video analysis</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                            <ChartBarIcon className="w-4 h-4 text-pink-500" />
                            <span>Real-time scoring</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mode Selection Cards */}
            <div className="max-w-5xl mx-auto px-6 py-10">
                <div className="grid md:grid-cols-2 gap-8">
                    {modes.map((mode) => (
                        <Link
                            key={mode.id}
                            href={mode.href}
                            className="group"
                        >
                            <div
                                className={`relative overflow-hidden rounded-3xl bg-gradient-to-br ${mode.color} ${mode.hoverColor} p-8 text-white shadow-xl transition-all duration-300 transform hover:scale-[1.02] hover:shadow-2xl cursor-pointer`}
                                onMouseEnter={() => setSelectedMode(mode.id)}
                                onMouseLeave={() => setSelectedMode(null)}
                            >
                                {/* Background Pattern */}
                                <div className="absolute inset-0 opacity-10">
                                    <div className="absolute top-0 right-0 w-64 h-64 rounded-full bg-white/20 -translate-y-1/2 translate-x-1/2" />
                                    <div className="absolute bottom-0 left-0 w-48 h-48 rounded-full bg-white/20 translate-y-1/2 -translate-x-1/2" />
                                </div>

                                {/* Icon */}
                                <div className="relative">
                                    <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                        <mode.icon className="w-8 h-8 text-white" />
                                    </div>

                                    {/* Title */}
                                    <div className="mb-4">
                                        <h2 className="text-2xl font-bold mb-1">{mode.title}</h2>
                                        <p className="text-white/80 text-sm font-medium">{mode.subtitle}</p>
                                    </div>

                                    {/* Description */}
                                    <p className="text-white/90 text-sm leading-relaxed mb-6">
                                        {mode.description}
                                    </p>

                                    {/* Features */}
                                    <ul className="space-y-2 mb-6">
                                        {mode.features.map((feature, idx) => (
                                            <li key={idx} className="flex items-center gap-2 text-sm text-white/90">
                                                <SparklesIcon className="w-4 h-4 text-white/70 flex-shrink-0" />
                                                <span>{feature}</span>
                                            </li>
                                        ))}
                                    </ul>

                                    {/* CTA */}
                                    <div className="flex items-center gap-2 text-white font-semibold group-hover:gap-4 transition-all">
                                        <span>Get Started</span>
                                        <ArrowRightIcon className="w-5 h-5" />
                                    </div>
                                </div>

                                {/* Hover Indicator */}
                                <div className={`absolute bottom-0 left-0 right-0 h-1 bg-white/50 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left`} />
                            </div>
                        </Link>
                    ))}
                </div>

                {/* Info Section */}
                <div className="mt-12 bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">How It Works</h3>
                    <div className="grid md:grid-cols-3 gap-6">
                        <div className="flex items-start gap-3">
                            <div className="w-10 h-10 rounded-xl bg-indigo-100 flex items-center justify-center flex-shrink-0">
                                <span className="text-indigo-600 font-bold">1</span>
                            </div>
                            <div>
                                <p className="font-medium text-gray-900">Choose Mode</p>
                                <p className="text-sm text-gray-500">Select Mock Interview for subjects or Soft Skills for communication</p>
                            </div>
                        </div>
                        <div className="flex items-start gap-3">
                            <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center flex-shrink-0">
                                <span className="text-purple-600 font-bold">2</span>
                            </div>
                            <div>
                                <p className="font-medium text-gray-900">Interact with AI</p>
                                <p className="text-sm text-gray-500">Answer questions from our 3D avatar interviewer via voice</p>
                            </div>
                        </div>
                        <div className="flex items-start gap-3">
                            <div className="w-10 h-10 rounded-xl bg-pink-100 flex items-center justify-center flex-shrink-0">
                                <span className="text-pink-600 font-bold">3</span>
                            </div>
                            <div>
                                <p className="font-medium text-gray-900">Get Feedback</p>
                                <p className="text-sm text-gray-500">Receive detailed scores and improvement suggestions</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
