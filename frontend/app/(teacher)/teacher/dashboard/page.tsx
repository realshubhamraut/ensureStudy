'use client'
import { getApiBaseUrl } from '@/utils/api'

import { useState, useEffect, useRef } from 'react'
import { useSession } from 'next-auth/react'
import {
    PaperAirplaneIcon,
    SparklesIcon,
    UserGroupIcon,
    ClipboardDocumentListIcon,
    CalendarDaysIcon,
    AcademicCapIcon,
    ArrowPathIcon,
    ChartBarIcon,
    ExclamationTriangleIcon,
    DocumentTextIcon,
    ClockIcon,
    UserIcon
} from '@heroicons/react/24/outline'

interface ChatMessage {
    id: string
    type: 'user' | 'assistant'
    content: string
    tableData?: {
        headers: string[]
        rows: string[][]
    }
    timestamp: Date
    loading?: boolean
}

interface Stats {
    totalStudents: number
    pendingEvaluations: number
    upcomingDeadlines: number
    activeClassrooms: number
}

const SUGGESTION_CHIPS = [
    {
        label: 'Show pending evaluations for the recent physics exam',
        query: 'Show pending evaluations for the recent physics exam',
        color: 'bg-blue-100 text-blue-700 hover:bg-blue-200 border-blue-200'
    },
    {
        label: 'Who are weak at Units and Measurements topic?',
        query: 'Who are weak at Units and Measurements topic?',
        color: 'bg-purple-100 text-purple-700 hover:bg-purple-200 border-purple-200'
    },
    {
        label: 'Students struggling with Trigonometry concepts',
        query: 'Students struggling with Trigonometry concepts',
        color: 'bg-red-100 text-red-700 hover:bg-red-200 border-red-200'
    },
    {
        label: 'What assignments are due this week?',
        query: 'What assignments are due this week?',
        color: 'bg-orange-100 text-orange-700 hover:bg-orange-200 border-orange-200'
    },
    {
        label: 'Which students haven\'t submitted the last homework?',
        query: 'Which students haven\'t submitted the last homework?',
        color: 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200 border-yellow-200'
    },
    {
        label: 'Compare Class 10-A and 10-B performance',
        query: 'Compare Class 10-A and 10-B performance',
        color: 'bg-green-100 text-green-700 hover:bg-green-200 border-green-200'
    },
    {
        label: 'Give me a summary of all my students',
        query: 'Give me a summary of all my students',
        color: 'bg-cyan-100 text-cyan-700 hover:bg-cyan-200 border-cyan-200'
    },
    {
        label: 'Show recent exam results and average scores',
        query: 'Show recent exam results and average scores',
        color: 'bg-pink-100 text-pink-700 hover:bg-pink-200 border-pink-200'
    }
]

export default function TeacherDashboardPage() {
    const { data: session } = useSession()
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [stats, setStats] = useState<Stats>({
        totalStudents: 0,
        pendingEvaluations: 0,
        upcomingDeadlines: 0,
        activeClassrooms: 0
    })
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLInputElement>(null)

    // Auto-scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Fetch initial stats
    useEffect(() => {
        fetchStats()
    }, [])

    const fetchStats = async () => {
        try {
            // Fetch classrooms to get stats
            const res = await fetch(`${getApiBaseUrl()}/api/classroom`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                const classrooms = data.classrooms || []
                const totalStudents = classrooms.reduce((sum: number, c: any) => sum + (c.student_count || 0), 0)
                setStats({
                    totalStudents,
                    pendingEvaluations: 12, // TODO: Fetch from API
                    upcomingDeadlines: 5,   // TODO: Fetch from API
                    activeClassrooms: classrooms.length
                })
            }
        } catch (error) {
            console.error('Failed to fetch stats:', error)
        }
    }

    const handleSend = async (queryText?: string) => {
        const query = queryText || input.trim()
        if (!query) return

        // Add user message
        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            type: 'user',
            content: query,
            timestamp: new Date()
        }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setLoading(true)

        // Add loading message
        const loadingMessage: ChatMessage = {
            id: (Date.now() + 1).toString(),
            type: 'assistant',
            content: '',
            timestamp: new Date(),
            loading: true
        }
        setMessages(prev => [...prev, loadingMessage])

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/teacher-assistant/query`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            })

            const data = await res.json()

            // Remove loading message and add real response
            setMessages(prev => {
                const filtered = prev.filter(m => !m.loading)
                const assistantMessage: ChatMessage = {
                    id: (Date.now() + 2).toString(),
                    type: 'assistant',
                    content: data.summary || data.message || 'Here are the results:',
                    tableData: data.table_data,
                    timestamp: new Date()
                }
                return [...filtered, assistantMessage]
            })
        } catch (error) {
            // Remove loading and show error
            setMessages(prev => {
                const filtered = prev.filter(m => !m.loading)
                const errorMessage: ChatMessage = {
                    id: (Date.now() + 2).toString(),
                    type: 'assistant',
                    content: 'Sorry, I encountered an error processing your request. Please try again.',
                    timestamp: new Date()
                }
                return [...filtered, errorMessage]
            })
        } finally {
            setLoading(false)
        }
    }

    const handleChipClick = (query: string) => {
        setInput(query)
        inputRef.current?.focus()
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="h-[calc(100vh-120px)] flex flex-col">
            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
                <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-blue-100 rounded-xl">
                            <UserGroupIcon className="w-7 h-7 text-blue-600" />
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-gray-900">{stats.totalStudents}</p>
                            <p className="text-sm text-gray-500">Total Students</p>
                        </div>
                    </div>
                </div>
                <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-orange-100 rounded-xl">
                            <ClipboardDocumentListIcon className="w-7 h-7 text-orange-600" />
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-gray-900">{stats.pendingEvaluations}</p>
                            <p className="text-sm text-gray-500">Pending Evals</p>
                        </div>
                    </div>
                </div>
                <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-red-100 rounded-xl">
                            <CalendarDaysIcon className="w-7 h-7 text-red-600" />
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-gray-900">{stats.upcomingDeadlines}</p>
                            <p className="text-sm text-gray-500">Deadlines</p>
                        </div>
                    </div>
                </div>
                <div className="bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-purple-100 rounded-xl">
                            <AcademicCapIcon className="w-7 h-7 text-purple-600" />
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-gray-900">{stats.activeClassrooms}</p>
                            <p className="text-sm text-gray-500">Classrooms</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Chat Container */}
            <div className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col overflow-hidden">

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center">
                            <div className="p-4 bg-gradient-to-br from-purple-100 to-pink-100 rounded-2xl mb-4">
                                <SparklesIcon className="w-12 h-12 text-purple-600" />
                            </div>
                            <h3 className="text-xl font-semibold text-gray-900 mb-2">
                                Welcome, {session?.user?.username || 'Teacher'}! 👋
                            </h3>
                            <p className="text-gray-500 max-w-lg mb-6">
                                Track student performance, evaluations, and get insights across all your classes.
                            </p>
                            <p className="text-sm text-gray-400">Try one of the suggestions below to get started</p>
                        </div>
                    ) : (
                        messages.map((message) => (
                            <div
                                key={message.id}
                                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${message.type === 'user'
                                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                                        : 'bg-gray-100 text-gray-900'
                                        }`}
                                >
                                    {message.loading ? (
                                        <div className="flex items-center gap-2">
                                            <ArrowPathIcon className="w-4 h-4 animate-spin" />
                                            <span className="text-sm">Analyzing your data...</span>
                                        </div>
                                    ) : (
                                        <>
                                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                                            {message.tableData && (
                                                <div className="mt-3 overflow-x-auto">
                                                    <table className="w-full text-sm border-collapse">
                                                        <thead>
                                                            <tr className="border-b border-gray-300">
                                                                {message.tableData.headers.map((header, idx) => (
                                                                    <th key={idx} className="text-left py-2 px-3 font-semibold">
                                                                        {header}
                                                                    </th>
                                                                ))}
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {message.tableData.rows.map((row, rowIdx) => (
                                                                <tr key={rowIdx} className="border-b border-gray-200">
                                                                    {row.map((cell, cellIdx) => (
                                                                        <td key={cellIdx} className="py-2 px-3">
                                                                            {cell}
                                                                        </td>
                                                                    ))}
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Suggestion Chips */}
                {messages.length === 0 && (
                    <div className="px-6 pb-6">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {SUGGESTION_CHIPS.map((chip, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => handleChipClick(chip.query)}
                                    className={`px-4 py-3 rounded-xl text-sm font-medium border transition-all hover:scale-[1.02] hover:shadow-md ${chip.color}`}
                                >
                                    {chip.label}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Input Area */}
                <div className="p-4 border-t border-gray-100 bg-gray-50">
                    <div className="flex items-center gap-3 max-w-4xl mx-auto">
                        <div className="flex-1 relative">
                            <input
                                ref={inputRef}
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="Ask anything about your students..."
                                className="w-full px-4 py-3 pr-12 bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent shadow-sm"
                                disabled={loading}
                            />
                        </div>
                        <button
                            onClick={() => handleSend()}
                            disabled={loading || !input.trim()}
                            className="p-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
                        >
                            {loading ? (
                                <ArrowPathIcon className="w-5 h-5 animate-spin" />
                            ) : (
                                <PaperAirplaneIcon className="w-5 h-5" />
                            )}
                        </button>
                    </div>
                    <p className="text-center text-xs text-gray-400 mt-2">
                        Press Enter to send • Queries search across all your classes
                    </p>
                </div>
            </div>
        </div>
    )
}
