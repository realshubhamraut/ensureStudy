'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import {
    AcademicCapIcon,
    PlusIcon,
    ClipboardDocumentIcon,
    ArrowPathIcon
} from '@heroicons/react/24/outline'

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    subject: string
    join_code: string
    student_count: number
    teacher?: { name: string }
}

export default function JoinClassroomPage() {
    const router = useRouter()
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [code, setCode] = useState('')
    const [myClassrooms, setMyClassrooms] = useState<Classroom[]>([])
    const [fetchingClassrooms, setFetchingClassrooms] = useState(true)

    useEffect(() => {
        fetchMyClassrooms()
    }, [])

    const fetchMyClassrooms = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/my-classrooms`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setMyClassrooms(data.classrooms)
            }
        } catch (err) {
            console.error('Failed to fetch classrooms:', err)
        } finally {
            setFetchingClassrooms(false)
        }
    }

    const handleJoin = async () => {
        if (!code.trim()) {
            setError('Please enter a classroom code')
            return
        }

        setLoading(true)
        setError('')

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/join`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code: code.trim() })
            })

            const data = await res.json()

            if (res.ok) {
                alert(`Successfully joined ${data.classroom.name}!`)
                setCode('')
                fetchMyClassrooms()
            } else {
                setError(data.error || 'Failed to join classroom')
            }
        } catch (err) {
            setError('Something went wrong. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="max-w-2xl mx-auto space-y-8">
            {/* Join Classroom Card */}
            <div className="card">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500">
                        <AcademicCapIcon className="w-8 h-8 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Join a Classroom</h1>
                        <p className="text-gray-600">Enter the code provided by your teacher</p>
                    </div>
                </div>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Classroom Code
                        </label>
                        <input
                            type="text"
                            value={code}
                            onChange={(e) => setCode(e.target.value.toUpperCase())}
                            placeholder="e.g., ABC123"
                            className="input-field text-center text-2xl font-mono tracking-widest uppercase"
                            maxLength={6}
                        />
                    </div>

                    {error && (
                        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                            {error}
                        </div>
                    )}

                    <button
                        onClick={handleJoin}
                        disabled={loading || !code.trim()}
                        className="w-full btn-primary flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <>
                                <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                Joining...
                            </>
                        ) : (
                            <>
                                <PlusIcon className="w-5 h-5" />
                                Join Classroom
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* My Classrooms */}
            <div>
                <h2 className="text-lg font-semibold text-gray-900 mb-4">My Classrooms</h2>

                {fetchingClassrooms ? (
                    <div className="text-center py-8">
                        <div className="spinner mx-auto"></div>
                    </div>
                ) : myClassrooms.length === 0 ? (
                    <div className="text-center py-12 bg-gray-50 rounded-xl">
                        <AcademicCapIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                        <p className="text-gray-500">You haven't joined any classrooms yet</p>
                        <p className="text-sm text-gray-400 mt-1">
                            Ask your teacher for a classroom code to get started
                        </p>
                    </div>
                ) : (
                    <div className="grid gap-4">
                        {myClassrooms.map((classroom) => (
                            <div key={classroom.id} className="card-hover flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 rounded-xl bg-green-100">
                                        <ClipboardDocumentIcon className="w-6 h-6 text-green-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-gray-900">{classroom.name}</h3>
                                        <p className="text-sm text-gray-500">
                                            {classroom.grade && `Grade ${classroom.grade}`}
                                            {classroom.section && ` • Section ${classroom.section}`}
                                            {classroom.teacher && ` • ${classroom.teacher.name}`}
                                        </p>
                                    </div>
                                </div>
                                <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                                    Enrolled
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
