'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import {
    ArrowLeftIcon,
    AcademicCapIcon,
    MicrophoneIcon,
    VideoCameraIcon,
    PlayIcon,
    UserIcon,
    BookOpenIcon,
    ChevronDownIcon,
    SparklesIcon,
    CheckCircleIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'

// Mock data - in production, fetch from API
const subjects = [
    { id: 'math', name: 'Mathematics', icon: '📐', chapters: ['Algebra', 'Calculus', 'Trigonometry', 'Statistics', 'Geometry'] },
    { id: 'physics', name: 'Physics', icon: '⚡', chapters: ['Mechanics', 'Thermodynamics', 'Optics', 'Electromagnetism', 'Modern Physics'] },
    { id: 'chemistry', name: 'Chemistry', icon: '🧪', chapters: ['Organic Chemistry', 'Inorganic Chemistry', 'Physical Chemistry', 'Biochemistry', 'Analytical Chemistry'] }
]

const avatars = [
    { id: 'female', name: 'Sara', gender: 'Female', image: '/avatars/female-avatar.png' },
    { id: 'male', name: 'Alex', gender: 'Male', image: '/avatars/male-avatar.png' }
]

export default function MockInterviewPage() {
    const router = useRouter()
    const [selectedSubject, setSelectedSubject] = useState<string | null>(null)
    const [selectedChapter, setSelectedChapter] = useState<string | null>(null)
    const [selectedAvatar, setSelectedAvatar] = useState<string>('female')
    const [isReady, setIsReady] = useState(false)
    const [showChapters, setShowChapters] = useState(false)

    // Check if ready to start
    useEffect(() => {
        setIsReady(selectedSubject !== null && selectedChapter !== null)
    }, [selectedSubject, selectedChapter])

    const selectedSubjectData = subjects.find(s => s.id === selectedSubject)

    const startInterview = () => {
        if (isReady) {
            router.push(`/softskills/mock-interview/session?subject=${selectedSubject}&chapter=${selectedChapter}&avatar=${selectedAvatar}`)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200 px-6 py-6">
                <div className="max-w-4xl mx-auto">
                    <Link href="/softskills" className="inline-flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4 transition-colors">
                        <ArrowLeftIcon className="w-4 h-4" />
                        <span className="text-sm">Back to Soft Skills</span>
                    </Link>
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                            <AcademicCapIcon className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">Mock Interview</h1>
                            <p className="text-gray-500">Test your subject knowledge with an AI interviewer</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-4xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Left Column - Setup */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Step 1: Subject Selection */}
                        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                            <div className="flex items-center gap-3 mb-5">
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                                    <span className="text-blue-600 font-bold text-sm">1</span>
                                </div>
                                <h2 className="text-lg font-semibold text-gray-900">Choose Subject</h2>
                                {selectedSubject && <CheckCircleIcon className="w-5 h-5 text-green-500 ml-auto" />}
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                {subjects.map((subject) => (
                                    <button
                                        key={subject.id}
                                        onClick={() => {
                                            setSelectedSubject(subject.id)
                                            setSelectedChapter(null)
                                            setShowChapters(true)
                                        }}
                                        className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${selectedSubject === subject.id
                                            ? 'border-blue-500 bg-blue-50 shadow-md'
                                            : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        <span className="text-3xl">{subject.icon}</span>
                                        <span className={`text-sm font-medium ${selectedSubject === subject.id ? 'text-blue-700' : 'text-gray-700'}`}>
                                            {subject.name}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Step 2: Chapter Selection */}
                        <div className={`bg-white rounded-2xl border border-gray-200 p-6 shadow-sm transition-opacity ${selectedSubject ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                            <div className="flex items-center gap-3 mb-5">
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                                    <span className="text-blue-600 font-bold text-sm">2</span>
                                </div>
                                <h2 className="text-lg font-semibold text-gray-900">Select Topic</h2>
                                {selectedChapter && <CheckCircleIcon className="w-5 h-5 text-green-500 ml-auto" />}
                            </div>
                            {selectedSubjectData && (
                                <div className="space-y-2">
                                    {selectedSubjectData.chapters.map((chapter) => (
                                        <button
                                            key={chapter}
                                            onClick={() => setSelectedChapter(chapter)}
                                            className={`w-full p-3 rounded-lg border transition-all flex items-center gap-3 ${selectedChapter === chapter
                                                ? 'border-blue-500 bg-blue-50'
                                                : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                                                }`}
                                        >
                                            <BookOpenIcon className={`w-5 h-5 ${selectedChapter === chapter ? 'text-blue-600' : 'text-gray-400'}`} />
                                            <span className={`text-sm font-medium ${selectedChapter === chapter ? 'text-blue-700' : 'text-gray-700'}`}>
                                                {chapter}
                                            </span>
                                            {selectedChapter === chapter && (
                                                <CheckCircleIcon className="w-4 h-4 text-blue-600 ml-auto" />
                                            )}
                                        </button>
                                    ))}
                                </div>
                            )}
                            {!selectedSubject && (
                                <p className="text-gray-400 text-sm text-center py-4">Select a subject first</p>
                            )}
                        </div>

                        {/* Step 3: Avatar Selection */}
                        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                            <div className="flex items-center gap-3 mb-5">
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                                    <span className="text-blue-600 font-bold text-sm">3</span>
                                </div>
                                <h2 className="text-lg font-semibold text-gray-900">Choose Interviewer</h2>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                {avatars.map((avatar) => (
                                    <button
                                        key={avatar.id}
                                        onClick={() => setSelectedAvatar(avatar.id)}
                                        className={`p-5 rounded-xl border-2 transition-all flex flex-col items-center gap-3 ${selectedAvatar === avatar.id
                                            ? 'border-blue-500 bg-blue-50 shadow-md'
                                            : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        <div className={`w-20 h-20 rounded-full flex items-center justify-center ${selectedAvatar === avatar.id
                                            ? 'bg-gradient-to-br from-blue-400 to-indigo-500'
                                            : 'bg-gradient-to-br from-gray-300 to-gray-400'
                                            }`}>
                                            <UserIcon className="w-10 h-10 text-white" />
                                        </div>
                                        <div className="text-center">
                                            <p className={`font-semibold ${selectedAvatar === avatar.id ? 'text-blue-700' : 'text-gray-700'}`}>
                                                {avatar.name}
                                            </p>
                                            <p className="text-xs text-gray-500">{avatar.gender} Avatar</p>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right Column - Preview & Start */}
                    <div className="lg:col-span-1">
                        <div className="sticky top-6 space-y-6">
                            {/* Preview Card */}
                            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl p-6 text-white shadow-xl">
                                <h3 className="font-semibold mb-4">Interview Preview</h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex items-center justify-between">
                                        <span className="text-blue-200">Subject</span>
                                        <span className="font-medium">{selectedSubjectData?.name || '—'}</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-blue-200">Topic</span>
                                        <span className="font-medium">{selectedChapter || '—'}</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-blue-200">Interviewer</span>
                                        <span className="font-medium">{avatars.find(a => a.id === selectedAvatar)?.name}</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-blue-200">Questions</span>
                                        <span className="font-medium">5-8 questions</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-blue-200">Duration</span>
                                        <span className="font-medium">~10-15 min</span>
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
                                        <span>Camera access (optional)</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm text-gray-600">
                                        <SparklesIcon className="w-4 h-4 text-gray-400" />
                                        <span>Quiet environment</span>
                                    </div>
                                </div>
                            </div>

                            {/* Start Button */}
                            <button
                                onClick={startInterview}
                                disabled={!isReady}
                                className={`w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${isReady
                                    ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg hover:shadow-xl hover:scale-[1.02]'
                                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                                    }`}
                            >
                                <PlayIcon className="w-5 h-5" />
                                <span>Start Interview</span>
                            </button>

                            {!isReady && (
                                <p className="text-xs text-gray-400 text-center">
                                    Select a subject and topic to continue
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
