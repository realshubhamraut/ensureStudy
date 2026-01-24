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
    SparklesIcon,
    CheckCircleIcon,
    ExclamationTriangleIcon,
    ArrowTrendingDownIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import { getApiBaseUrl } from '@/utils/api'

// Types
interface Classroom {
    id: string
    name: string
    subject?: string
    teacher_name?: string
}

interface Topic {
    id: string
    name: string
    description?: string
    subject_name?: string
    subject_icon?: string
    confidence_score: number
    subtopics_count: number
    subtopics_mastered: number
    is_weak: boolean
}

const avatars = [
    { id: 'female', name: 'Sara', gender: 'Female', image: '/avatars/female-avatar.png' },
    { id: 'male', name: 'Alex', gender: 'Male', image: '/avatars/male-avatar.png' }
]

// Confidence color helper
const getConfidenceColor = (score: number) => {
    if (score >= 75) return { bg: 'bg-green-100', text: 'text-green-700', ring: 'ring-green-500' }
    if (score >= 50) return { bg: 'bg-yellow-100', text: 'text-yellow-700', ring: 'ring-yellow-500' }
    return { bg: 'bg-red-100', text: 'text-red-700', ring: 'ring-red-500' }
}

const getConfidenceEmoji = (score: number) => {
    if (score >= 75) return '🟢'
    if (score >= 50) return '🟡'
    return '🔴'
}

export default function MockInterviewPage() {
    const router = useRouter()

    // State
    const [classrooms, setClassrooms] = useState<Classroom[]>([])
    const [selectedClassrooms, setSelectedClassrooms] = useState<string[]>([])
    const [allClassroomsSelected, setAllClassroomsSelected] = useState(true)
    const [topics, setTopics] = useState<Topic[]>([])
    const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null)
    const [selectedAvatar, setSelectedAvatar] = useState<string>('female')
    const [loading, setLoading] = useState({ classrooms: true, topics: false })
    const [isReady, setIsReady] = useState(false)

    // Check if ready to start
    useEffect(() => {
        setIsReady(selectedTopic !== null && (allClassroomsSelected || selectedClassrooms.length > 0))
    }, [selectedTopic, selectedClassrooms, allClassroomsSelected])

    // Fetch classrooms on mount
    useEffect(() => {
        const fetchClassrooms = async () => {
            try {
                const res = await fetch(`${getApiBaseUrl()}/api/classroom/my-classrooms`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (res.ok) {
                    const data = await res.json()
                    setClassrooms(data.classrooms || [])
                }
            } catch (error) {
                console.error('Failed to fetch classrooms:', error)
            } finally {
                setLoading(prev => ({ ...prev, classrooms: false }))
            }
        }
        fetchClassrooms()
    }, [])

    // Fetch topics when classroom selection changes
    useEffect(() => {
        const fetchTopics = async () => {
            setLoading(prev => ({ ...prev, topics: true }))
            setSelectedTopic(null)

            try {
                const classroomIds = allClassroomsSelected ? ['all'] : selectedClassrooms

                const res = await fetch(`${getApiBaseUrl()}/api/topics/by-classrooms`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    },
                    body: JSON.stringify({
                        classroom_ids: classroomIds,
                        sort_by_confidence: 'asc' // Weak topics first
                    })
                })

                if (res.ok) {
                    const data = await res.json()
                    setTopics(data.topics || [])
                }
            } catch (error) {
                console.error('Failed to fetch topics:', error)
            } finally {
                setLoading(prev => ({ ...prev, topics: false }))
            }
        }

        if (classrooms.length > 0 && (allClassroomsSelected || selectedClassrooms.length > 0)) {
            fetchTopics()
        }
    }, [classrooms, selectedClassrooms, allClassroomsSelected])

    // Toggle classroom selection
    const toggleClassroom = (classroomId: string) => {
        setAllClassroomsSelected(false)
        setSelectedClassrooms(prev =>
            prev.includes(classroomId)
                ? prev.filter(id => id !== classroomId)
                : [...prev, classroomId]
        )
    }

    // Select all classrooms
    const selectAllClassrooms = () => {
        setAllClassroomsSelected(true)
        setSelectedClassrooms([])
    }

    const startInterview = () => {
        if (isReady && selectedTopic) {
            const classroomParam = allClassroomsSelected ? 'all' : selectedClassrooms.join(',')
            router.push(`/softskills/mock-interview/session?topic=${selectedTopic.id}&topic_name=${encodeURIComponent(selectedTopic.name)}&classrooms=${classroomParam}&avatar=${selectedAvatar}`)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200 px-6 py-6">
                <div className="max-w-5xl mx-auto">
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

            <div className="max-w-5xl mx-auto px-6 py-8">
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Left Column - Setup */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Step 1: Classroom Selection */}
                        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                            <div className="flex items-center gap-3 mb-5">
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                                    <span className="text-blue-600 font-bold text-sm">1</span>
                                </div>
                                <h2 className="text-lg font-semibold text-gray-900">Select Classrooms</h2>
                                {(allClassroomsSelected || selectedClassrooms.length > 0) && (
                                    <CheckCircleIcon className="w-5 h-5 text-green-500 ml-auto" />
                                )}
                            </div>

                            {loading.classrooms ? (
                                <div className="flex items-center justify-center py-8">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                                </div>
                            ) : classrooms.length === 0 ? (
                                <div className="text-center py-8 text-gray-500">
                                    <BookOpenIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                                    <p>No classrooms found. Join a classroom first.</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {/* All Classrooms Button */}
                                    <button
                                        onClick={selectAllClassrooms}
                                        className={`w-full p-4 rounded-xl border-2 transition-all flex items-center gap-3 ${allClassroomsSelected
                                                ? 'border-blue-500 bg-blue-50 shadow-md'
                                                : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${allClassroomsSelected ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                                            }`}>
                                            {allClassroomsSelected && <CheckCircleIcon className="w-4 h-4 text-white" />}
                                        </div>
                                        <span className={`font-medium ${allClassroomsSelected ? 'text-blue-700' : 'text-gray-700'}`}>
                                            All Classrooms
                                        </span>
                                        <span className="text-sm text-gray-500 ml-auto">
                                            {classrooms.length} classroom{classrooms.length !== 1 ? 's' : ''}
                                        </span>
                                    </button>

                                    {/* Individual Classrooms */}
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                        {classrooms.map((classroom) => {
                                            const isSelected = !allClassroomsSelected && selectedClassrooms.includes(classroom.id)
                                            return (
                                                <button
                                                    key={classroom.id}
                                                    onClick={() => toggleClassroom(classroom.id)}
                                                    className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${isSelected
                                                            ? 'border-blue-500 bg-blue-50 shadow-md'
                                                            : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                                                        }`}
                                                >
                                                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isSelected ? 'bg-blue-500' : 'bg-gray-200'
                                                        }`}>
                                                        <BookOpenIcon className={`w-5 h-5 ${isSelected ? 'text-white' : 'text-gray-500'}`} />
                                                    </div>
                                                    <span className={`text-sm font-medium text-center line-clamp-2 ${isSelected ? 'text-blue-700' : 'text-gray-700'
                                                        }`}>
                                                        {classroom.name}
                                                    </span>
                                                    {classroom.subject && (
                                                        <span className="text-xs text-gray-500">{classroom.subject}</span>
                                                    )}
                                                </button>
                                            )
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Step 2: Topic Selection */}
                        <div className={`bg-white rounded-2xl border border-gray-200 p-6 shadow-sm transition-opacity ${(allClassroomsSelected || selectedClassrooms.length > 0) ? 'opacity-100' : 'opacity-50 pointer-events-none'
                            }`}>
                            <div className="flex items-center gap-3 mb-5">
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                                    <span className="text-blue-600 font-bold text-sm">2</span>
                                </div>
                                <h2 className="text-lg font-semibold text-gray-900">Select Topic</h2>
                                {selectedTopic && <CheckCircleIcon className="w-5 h-5 text-green-500 ml-auto" />}
                            </div>

                            {/* Weak topics hint */}
                            <div className="flex items-center gap-2 mb-4 text-sm text-amber-600 bg-amber-50 px-3 py-2 rounded-lg">
                                <ArrowTrendingDownIcon className="w-4 h-4" />
                                <span>Topics are sorted by confidence — weak topics appear first</span>
                            </div>

                            {loading.topics ? (
                                <div className="flex items-center justify-center py-8">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                                </div>
                            ) : topics.length === 0 ? (
                                <div className="text-center py-8 text-gray-500">
                                    <ExclamationTriangleIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                                    <p>No topics found for selected classrooms.</p>
                                    <p className="text-sm mt-1">Ask your teacher to set up the syllabus.</p>
                                </div>
                            ) : (
                                <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                                    {topics.map((topic) => {
                                        const colors = getConfidenceColor(topic.confidence_score)
                                        const isSelected = selectedTopic?.id === topic.id

                                        return (
                                            <button
                                                key={topic.id}
                                                onClick={() => setSelectedTopic(topic)}
                                                className={`w-full p-4 rounded-lg border-2 transition-all flex items-center gap-4 ${isSelected
                                                        ? 'border-blue-500 bg-blue-50'
                                                        : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                                                    }`}
                                            >
                                                {/* Topic Icon/Radio */}
                                                <div className={`w-5 h-5 rounded-full border-2 flex-shrink-0 flex items-center justify-center ${isSelected ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                                                    }`}>
                                                    {isSelected && <div className="w-2 h-2 rounded-full bg-white"></div>}
                                                </div>

                                                {/* Topic Info */}
                                                <div className="flex-1 text-left">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`font-medium ${isSelected ? 'text-blue-700' : 'text-gray-800'}`}>
                                                            {topic.name}
                                                        </span>
                                                        {topic.is_weak && (
                                                            <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full">
                                                                Needs Practice
                                                            </span>
                                                        )}
                                                    </div>
                                                    {topic.subject_name && (
                                                        <span className="text-xs text-gray-500">
                                                            {topic.subject_icon} {topic.subject_name}
                                                        </span>
                                                    )}
                                                </div>

                                                {/* Confidence Score */}
                                                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full ${colors.bg}`}>
                                                    <span className="text-sm">{getConfidenceEmoji(topic.confidence_score)}</span>
                                                    <span className={`text-sm font-semibold ${colors.text}`}>
                                                        {topic.confidence_score}%
                                                    </span>
                                                </div>
                                            </button>
                                        )
                                    })}
                                </div>
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
                                        <span className="text-blue-200">Classrooms</span>
                                        <span className="font-medium">
                                            {allClassroomsSelected
                                                ? 'All'
                                                : selectedClassrooms.length > 0
                                                    ? `${selectedClassrooms.length} selected`
                                                    : '—'
                                            }
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-blue-200">Topic</span>
                                        <span className="font-medium truncate max-w-[150px]">
                                            {selectedTopic?.name || '—'}
                                        </span>
                                    </div>
                                    {selectedTopic && (
                                        <div className="flex items-center justify-between">
                                            <span className="text-blue-200">Confidence</span>
                                            <span className="font-medium">
                                                {getConfidenceEmoji(selectedTopic.confidence_score)} {selectedTopic.confidence_score}%
                                            </span>
                                        </div>
                                    )}
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
                                    Select classrooms and a topic to continue
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
