'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import Link from 'next/link'
import {
    ClipboardDocumentListIcon,
    PlayIcon,
    CheckCircleIcon,
    ClockIcon,
    TrophyIcon,
    PlusIcon,
    BoltIcon,
    AcademicCapIcon,
    UserGroupIcon,
    SparklesIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'
import { getApiBaseUrl } from '@/utils/api'
import CreateAssessmentModal from '@/components/assessments/CreateAssessmentModal'
import ChallengeModal from '@/components/assessments/ChallengeModal'
import ReceivedChallenges from '@/components/assessments/ReceivedChallenges'
import DailyRevisionBanner from '@/components/assessments/DailyRevisionBanner'

interface Assessment {
    id: string
    title: string
    topic: string
    subject: string
    difficulty: 'easy' | 'medium' | 'hard' | 'mixed'
    num_questions: number
    time_limit_minutes: number
    assessment_type: 'teacher_created' | 'self_practice' | 'student_challenge'
    use_ai_questions: boolean
    created_by?: string
    completed?: boolean
    score?: number
    is_revision_assessment?: boolean
    revision_date?: string
    source_topics?: string[]
}

interface Classroom {
    id: string
    name: string
    subject_name: string
}

type TabType = 'available' | 'my-assessments' | 'challenges'

export default function AssessmentsPage() {
    const { data: session } = useSession()
    const [assessments, setAssessments] = useState<Assessment[]>([])
    const [classrooms, setClassrooms] = useState<Classroom[]>([])
    const [activeTab, setActiveTab] = useState<TabType>('available')
    const [isLoading, setIsLoading] = useState(true)
    const [showCreateModal, setShowCreateModal] = useState(false)
    const [showChallengeModal, setShowChallengeModal] = useState(false)
    const [selectedAssessment, setSelectedAssessment] = useState<Assessment | null>(null)
    const [pendingChallengeCount, setPendingChallengeCount] = useState(0)

    useEffect(() => {
        fetchAssessments()
        fetchClassrooms()
        fetchPendingChallenges()
    }, [activeTab])

    const fetchAssessments = async () => {
        setIsLoading(true)
        try {
            let url = `${getApiBaseUrl()}/api/assessments/`
            if (activeTab === 'my-assessments') {
                url = `${getApiBaseUrl()}/api/assessments/my-assessments`
            }

            const res = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setAssessments(data.assessments || [])
            }
        } catch (err) {
            console.error('Failed to fetch assessments:', err)
        }
        setIsLoading(false)
    }

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
        } catch (err) {
            console.error('Failed to fetch classrooms:', err)
        }
    }

    const fetchPendingChallenges = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/assessments/challenges/received?status=pending`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setPendingChallengeCount(data.count || 0)
            }
        } catch (err) {
            console.error('Failed to fetch pending challenges:', err)
        }
    }

    const handleAssessmentCreated = (assessment: Assessment) => {
        setAssessments(prev => [assessment, ...prev])
        setShowCreateModal(false)
    }

    const handleChallengeClick = (assessment: Assessment) => {
        setSelectedAssessment(assessment)
        setShowChallengeModal(true)
    }

    const difficultyColors = {
        easy: 'bg-green-100 text-green-700',
        medium: 'bg-yellow-100 text-yellow-700',
        hard: 'bg-red-100 text-red-700',
        mixed: 'bg-purple-100 text-purple-700'
    }

    const typeIcons = {
        teacher_created: AcademicCapIcon,
        self_practice: SparklesIcon,
        student_challenge: BoltIcon
    }

    const typeLabels = {
        teacher_created: 'Teacher',
        self_practice: 'Practice',
        student_challenge: 'Challenge'
    }

    const completedCount = assessments.filter(a => a.completed).length
    const avgScore = assessments.filter(a => a.score != null).length > 0
        ? Math.round(assessments.filter(a => a.score != null).reduce((sum, a) => sum + (a.score || 0), 0) / assessments.filter(a => a.score != null).length)
        : 0

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">📝 Assessments</h1>
                    <p className="text-gray-500">Practice, compete, and track your progress</p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="btn-primary flex items-center gap-2 w-fit"
                >
                    <PlusIcon className="w-5 h-5" />
                    Create Assessment
                </button>
            </div>

            {/* Daily Revision Banner */}
            <DailyRevisionBanner />

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <StatCard
                    label="Completed"
                    value={completedCount.toString()}
                    icon={<CheckCircleIcon className="w-5 h-5" />}
                    color="green"
                />
                <StatCard
                    label="Average Score"
                    value={`${avgScore}%`}
                    icon={<TrophyIcon className="w-5 h-5" />}
                    color="blue"
                />
                <StatCard
                    label="Pending"
                    value={assessments.filter(a => !a.completed).length.toString()}
                    icon={<ClockIcon className="w-5 h-5" />}
                    color="orange"
                />
                <StatCard
                    label="Challenges"
                    value={pendingChallengeCount.toString()}
                    icon={<BoltIcon className="w-5 h-5" />}
                    color="purple"
                    highlight={pendingChallengeCount > 0}
                />
            </div>

            {/* Tabs */}
            <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
                {[
                    { id: 'available' as TabType, label: 'Available', icon: ClipboardDocumentListIcon },
                    { id: 'my-assessments' as TabType, label: 'My Assessments', icon: UserGroupIcon },
                    { id: 'challenges' as TabType, label: 'Challenges', icon: BoltIcon, badge: pendingChallengeCount }
                ].map(tab => {
                    const Icon = tab.icon
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={clsx(
                                'px-4 py-2 rounded-md font-medium text-sm flex items-center gap-2 transition-colors',
                                activeTab === tab.id
                                    ? 'bg-white text-primary-600 shadow-sm'
                                    : 'text-gray-600 hover:text-gray-900'
                            )}
                        >
                            <Icon className="w-4 h-4" />
                            {tab.label}
                            {tab.badge && tab.badge > 0 && (
                                <span className="bg-orange-500 text-white text-xs px-1.5 py-0.5 rounded-full">
                                    {tab.badge}
                                </span>
                            )}
                        </button>
                    )
                })}
            </div>

            {/* Content */}
            {activeTab === 'challenges' ? (
                <ReceivedChallenges onAccepted={(assessment) => {
                    setAssessments(prev => [assessment, ...prev])
                    setActiveTab('available')
                }} />
            ) : isLoading ? (
                <div className="text-center py-12 text-gray-500">Loading assessments...</div>
            ) : assessments.length === 0 ? (
                <div className="text-center py-12">
                    <ClipboardDocumentListIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                        {activeTab === 'my-assessments'
                            ? 'You haven\'t created any assessments yet'
                            : 'No assessments available'}
                    </p>
                    {activeTab === 'my-assessments' && (
                        <button
                            onClick={() => setShowCreateModal(true)}
                            className="mt-4 btn-primary inline-flex items-center gap-2"
                        >
                            <PlusIcon className="w-4 h-4" />
                            Create Your First Assessment
                        </button>
                    )}
                </div>
            ) : (
                <div className="grid md:grid-cols-2 gap-4">
                    {assessments.map((assessment) => {
                        const TypeIcon = typeIcons[assessment.assessment_type] || ClipboardDocumentListIcon
                        return (
                            <div key={assessment.id} className="card-hover">
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <h3 className="font-bold text-gray-900">{assessment.title}</h3>
                                            {assessment.use_ai_questions && (
                                                <SparklesIcon className="w-4 h-4 text-yellow-500" title="AI Generated" />
                                            )}
                                            {assessment.is_revision_assessment && (
                                                <span className="px-1.5 py-0.5 bg-gradient-to-r from-indigo-100 to-purple-100 text-indigo-700 text-xs rounded-full font-medium">
                                                    📅 Revision
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm text-gray-500">{assessment.topic}</p>
                                        {assessment.subject && assessment.subject !== 'General' && assessment.subject !== 'Revision' && (
                                            <p className="text-xs text-gray-400 mt-0.5">📚 {assessment.subject}</p>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className={clsx(
                                            'px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1',
                                            assessment.assessment_type === 'teacher_created' ? 'bg-blue-100 text-blue-700' :
                                                assessment.assessment_type === 'student_challenge' ? 'bg-orange-100 text-orange-700' :
                                                    'bg-purple-100 text-purple-700'
                                        )}>
                                            <TypeIcon className="w-3 h-3" />
                                            {typeLabels[assessment.assessment_type]}
                                        </span>
                                        <span className={clsx(
                                            'px-2 py-1 rounded-full text-xs font-medium capitalize',
                                            difficultyColors[assessment.difficulty]
                                        )}>
                                            {assessment.difficulty}
                                        </span>
                                    </div>
                                </div>

                                <div className="flex items-center gap-4 text-sm text-gray-500 mb-4">
                                    <span className="flex items-center gap-1">
                                        <ClipboardDocumentListIcon className="w-4 h-4" />
                                        {assessment.num_questions} questions
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <ClockIcon className="w-4 h-4" />
                                        {assessment.time_limit_minutes} min
                                    </span>
                                </div>

                                {assessment.completed ? (
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <CheckCircleIcon className="w-5 h-5 text-green-500" />
                                            <span className="text-green-600 font-medium">Completed</span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <span className="text-2xl font-bold text-gray-900">
                                                {assessment.score}%
                                            </span>
                                            {activeTab === 'my-assessments' && (
                                                <button
                                                    onClick={() => handleChallengeClick(assessment)}
                                                    className="px-3 py-1.5 bg-orange-100 text-orange-700 rounded-lg text-sm font-medium hover:bg-orange-200 transition-colors flex items-center gap-1"
                                                >
                                                    <BoltIcon className="w-4 h-4" />
                                                    Challenge
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex gap-2">
                                        <Link
                                            href={`/assessments/take/${assessment.id}`}
                                            className="btn-primary flex-1 flex items-center justify-center gap-2"
                                        >
                                            <PlayIcon className="w-5 h-5" />
                                            Start Quiz
                                        </Link>
                                    </div>
                                )}
                            </div>
                        )
                    })}
                </div>
            )}

            {/* Modals */}
            <CreateAssessmentModal
                isOpen={showCreateModal}
                onClose={() => setShowCreateModal(false)}
                onSuccess={handleAssessmentCreated}
                classrooms={classrooms}
            />

            {selectedAssessment && (
                <ChallengeModal
                    isOpen={showChallengeModal}
                    onClose={() => {
                        setShowChallengeModal(false)
                        setSelectedAssessment(null)
                    }}
                    assessmentId={selectedAssessment.id}
                    assessmentTitle={selectedAssessment.title}
                    onChallengeSent={() => {
                        setShowChallengeModal(false)
                        setSelectedAssessment(null)
                    }}
                />
            )}
        </div>
    )
}

function StatCard({
    label,
    value,
    icon,
    color,
    highlight = false
}: {
    label: string
    value: string
    icon: React.ReactNode
    color: 'green' | 'blue' | 'orange' | 'purple'
    highlight?: boolean
}) {
    const colors = {
        green: 'bg-green-100 text-green-600',
        blue: 'bg-blue-100 text-blue-600',
        orange: 'bg-orange-100 text-orange-600',
        purple: 'bg-purple-100 text-purple-600',
    }

    return (
        <div className={clsx(
            'card flex items-center gap-4',
            highlight && 'ring-2 ring-orange-400 ring-offset-2'
        )}>
            <div className={clsx('p-3 rounded-lg', colors[color])}>
                {icon}
            </div>
            <div>
                <p className="text-2xl font-bold text-gray-900">{value}</p>
                <p className="text-sm text-gray-500">{label}</p>
            </div>
        </div>
    )
}
