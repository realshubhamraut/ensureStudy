'use client'

import { useState, useEffect } from 'react'
import { XMarkIcon, UserIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Student {
    id: string
    username: string
    name: string
    avatar_url?: string
}

interface ChallengeModalProps {
    isOpen: boolean
    onClose: () => void
    assessmentId: string
    assessmentTitle: string
    onChallengeSent: () => void
}

export default function ChallengeModal({
    isOpen,
    onClose,
    assessmentId,
    assessmentTitle,
    onChallengeSent
}: ChallengeModalProps) {
    const [students, setStudents] = useState<Student[]>([])
    const [selectedStudent, setSelectedStudent] = useState<string>('')
    const [message, setMessage] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [isFetching, setIsFetching] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [searchQuery, setSearchQuery] = useState('')

    useEffect(() => {
        if (!isOpen) return

        const fetchStudents = async () => {
            setIsFetching(true)
            try {
                const res = await fetch('/api/assessments/challenge/students')
                if (res.ok) {
                    const data = await res.json()
                    setStudents(data.students || [])
                }
            } catch (err) {
                console.error('Failed to fetch students:', err)
            }
            setIsFetching(false)
        }

        fetchStudents()
    }, [isOpen])

    const filteredStudents = students.filter(s =>
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.username.toLowerCase().includes(searchQuery.toLowerCase())
    )

    const handleSendChallenge = async () => {
        if (!selectedStudent) return

        setIsLoading(true)
        setError(null)

        try {
            const res = await fetch('/api/assessments/challenge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    assessment_id: assessmentId,
                    recipient_id: selectedStudent,
                    message: message || undefined
                })
            })

            if (!res.ok) {
                const data = await res.json()
                throw new Error(data.error || 'Failed to send challenge')
            }

            onChallengeSent()
            onClose()
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Something went wrong')
        } finally {
            setIsLoading(false)
        }
    }

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between bg-gradient-to-r from-orange-500 to-red-500">
                    <div>
                        <h2 className="text-lg font-bold text-white">Challenge a Friend</h2>
                        <p className="text-sm text-white/80 truncate max-w-[200px]">{assessmentTitle}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 rounded-full hover:bg-white/20 transition-colors"
                    >
                        <XMarkIcon className="w-6 h-6 text-white" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-4">
                    {/* Search */}
                    <div>
                        <input
                            type="text"
                            placeholder="Search classmates..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                        />
                    </div>

                    {/* Student List */}
                    <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg">
                        {isFetching ? (
                            <div className="p-4 text-center text-gray-500">Loading classmates...</div>
                        ) : filteredStudents.length === 0 ? (
                            <div className="p-4 text-center text-gray-500">No classmates found</div>
                        ) : (
                            <div className="divide-y divide-gray-100">
                                {filteredStudents.map(student => (
                                    <button
                                        key={student.id}
                                        type="button"
                                        onClick={() => setSelectedStudent(student.id)}
                                        className={clsx(
                                            'w-full px-4 py-3 flex items-center gap-3 transition-colors',
                                            selectedStudent === student.id
                                                ? 'bg-orange-50 border-l-4 border-orange-500'
                                                : 'hover:bg-gray-50'
                                        )}
                                    >
                                        {student.avatar_url ? (
                                            <img
                                                src={student.avatar_url}
                                                alt={student.name}
                                                className="w-8 h-8 rounded-full object-cover"
                                            />
                                        ) : (
                                            <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                                                <UserIcon className="w-4 h-4 text-gray-500" />
                                            </div>
                                        )}
                                        <div className="text-left">
                                            <div className="font-medium text-gray-900">{student.name}</div>
                                            <div className="text-xs text-gray-500">@{student.username}</div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Message */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Add a message (optional)
                        </label>
                        <textarea
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            placeholder="Think you can beat my score? 😏"
                            rows={2}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none"
                        />
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                            {error}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSendChallenge}
                        disabled={isLoading || !selectedStudent}
                        className={clsx(
                            'px-5 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
                            isLoading || !selectedStudent
                                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                : 'bg-orange-500 text-white hover:bg-orange-600'
                        )}
                    >
                        {isLoading ? (
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        ) : (
                            <PaperAirplaneIcon className="w-4 h-4" />
                        )}
                        Send Challenge
                    </button>
                </div>
            </div>
        </div>
    )
}
