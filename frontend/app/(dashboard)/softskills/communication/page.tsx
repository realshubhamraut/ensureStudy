'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import {
    ArrowLeftIcon,
    ChatBubbleLeftRightIcon,
    MicrophoneIcon,
    VideoCameraIcon,
    PlayIcon,
    UserIcon,
    SparklesIcon,
    EyeIcon,
    HandRaisedIcon,
    ChartBarIcon
} from '@heroicons/react/24/outline'

const avatars = [
    { id: 'female', name: 'Sara', gender: 'Female' },
    { id: 'male', name: 'Alex', gender: 'Male' }
]

const skillsEvaluated = [
    { name: 'Fluency', description: 'Speech rate, pauses, and flow', icon: MicrophoneIcon, weight: '35%' },
    { name: 'Grammar', description: 'Sentence structure and correctness', icon: ChatBubbleLeftRightIcon, weight: '25%' },
    { name: 'Eye Contact', description: 'Looking at the camera/interviewer', icon: EyeIcon, weight: '20%' },
    { name: 'Hand Gestures', description: 'Natural and stable movements', icon: HandRaisedIcon, weight: '10%' },
    { name: 'Posture', description: 'Body stability and presence', icon: UserIcon, weight: '10%' }
]

export default function CommunicationPage() {
    const router = useRouter()
    const [selectedAvatar, setSelectedAvatar] = useState<string>('female')

    const startSession = () => {
        router.push(`/softskills/communication/session?avatar=${selectedAvatar}`)
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200 px-6 py-6">
                <div className="max-w-4xl mx-auto">
                    <Link href="/softskills" className="inline-flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4 transition-colors">
                        <ArrowLeftIcon className="w-4 h-4" />
                        <span className="text-sm">Back to Soft Skills</span>
                    </Link>
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center shadow-lg">
                            <ChatBubbleLeftRightIcon className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">Communication Skills</h1>
                            <p className="text-gray-500">Evaluate your speaking and body language</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-4xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Left Column */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Skills Evaluated */}
                        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                            <h2 className="text-lg font-semibold text-gray-900 mb-5">What We Evaluate</h2>
                            <div className="space-y-4">
                                {skillsEvaluated.map((skill) => (
                                    <div key={skill.name} className="flex items-center gap-4 p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors">
                                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0">
                                            <skill.icon className="w-5 h-5 text-white" />
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center justify-between">
                                                <p className="font-medium text-gray-900">{skill.name}</p>
                                                <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full font-medium">
                                                    {skill.weight}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-500">{skill.description}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Avatar Selection */}
                        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                            <h2 className="text-lg font-semibold text-gray-900 mb-5">Choose Your Interviewer</h2>
                            <div className="grid grid-cols-2 gap-4">
                                {avatars.map((avatar) => (
                                    <button
                                        key={avatar.id}
                                        onClick={() => setSelectedAvatar(avatar.id)}
                                        className={`p-5 rounded-xl border-2 transition-all flex flex-col items-center gap-3 ${selectedAvatar === avatar.id
                                            ? 'border-purple-500 bg-purple-50 shadow-md'
                                            : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        <div className={`w-20 h-20 rounded-full flex items-center justify-center ${selectedAvatar === avatar.id
                                            ? 'bg-gradient-to-br from-purple-400 to-pink-500'
                                            : 'bg-gradient-to-br from-gray-300 to-gray-400'
                                            }`}>
                                            <UserIcon className="w-10 h-10 text-white" />
                                        </div>
                                        <div className="text-center">
                                            <p className={`font-semibold ${selectedAvatar === avatar.id ? 'text-purple-700' : 'text-gray-700'}`}>
                                                {avatar.name}
                                            </p>
                                            <p className="text-xs text-gray-500">{avatar.gender} Avatar</p>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right Column */}
                    <div className="lg:col-span-1">
                        <div className="sticky top-6 space-y-6">
                            {/* Session Info */}
                            <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl p-6 text-white shadow-xl">
                                <h3 className="font-semibold mb-4">Session Info</h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex items-center justify-between">
                                        <span className="text-purple-200">Mode</span>
                                        <span className="font-medium">Soft Skills</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-purple-200">Interviewer</span>
                                        <span className="font-medium">{avatars.find(a => a.id === selectedAvatar)?.name}</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-purple-200">Duration</span>
                                        <span className="font-medium">3-5 minutes</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-purple-200">Analysis</span>
                                        <span className="font-medium">Real-time</span>
                                    </div>
                                </div>
                            </div>

                            {/* Requirements */}
                            <div className="bg-white rounded-2xl border border-gray-200 p-5">
                                <h3 className="font-semibold text-gray-900 mb-3">Requirements</h3>
                                <div className="space-y-2">
                                    <div className="flex items-center gap-2 text-sm text-gray-600">
                                        <MicrophoneIcon className="w-4 h-4 text-gray-400" />
                                        <span>Microphone access</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm text-gray-600">
                                        <VideoCameraIcon className="w-4 h-4 text-gray-400" />
                                        <span>Camera access (required)</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm text-gray-600">
                                        <SparklesIcon className="w-4 h-4 text-gray-400" />
                                        <span>Good lighting</span>
                                    </div>
                                </div>
                            </div>

                            {/* How It Works */}
                            <div className="bg-white rounded-2xl border border-gray-200 p-5">
                                <h3 className="font-semibold text-gray-900 mb-3">How It Works</h3>
                                <ol className="space-y-2 text-sm text-gray-600">
                                    <li className="flex items-start gap-2">
                                        <span className="w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-bold flex-shrink-0">1</span>
                                        <span>AI avatar will ask you questions</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-bold flex-shrink-0">2</span>
                                        <span>Respond naturally via voice</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-bold flex-shrink-0">3</span>
                                        <span>Get real-time feedback</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-bold flex-shrink-0">4</span>
                                        <span>View detailed scores</span>
                                    </li>
                                </ol>
                            </div>

                            {/* Start Button */}
                            <button
                                onClick={startSession}
                                className="w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg hover:shadow-xl hover:scale-[1.02] transition-all"
                            >
                                <PlayIcon className="w-5 h-5" />
                                <span>Start Evaluation</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
