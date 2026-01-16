'use client'

import { useState, useRef, useEffect } from 'react'
import {
    ChatBubbleLeftRightIcon,
    PaperAirplaneIcon,
    PlayIcon,
    XMarkIcon,
    SparklesIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Citation {
    recording_id: string
    meeting_id: string
    meeting_title?: string
    timestamp_start: number
    timestamp_end: number
    speaker_name?: string
    text: string
    relevance_score: number
}

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    citations?: Citation[]
    timestamp: Date
}

interface MeetingQAProps {
    classroomId: string
    accessToken: string
    aiServiceUrl: string
    onSeekToTimestamp?: (recordingId: string, timestamp: number) => void
}

/**
 * AI Chat interface for asking questions about meeting content
 * Shows answers with clickable citations that link to video timestamps
 */
export function MeetingQA({
    classroomId,
    accessToken,
    aiServiceUrl,
    onSeekToTimestamp
}: MeetingQAProps) {
    const [isOpen, setIsOpen] = useState(false)
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLInputElement>(null)

    // Scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Focus input when opened
    useEffect(() => {
        if (isOpen) {
            inputRef.current?.focus()
        }
    }, [isOpen])

    const formatTime = (seconds: number): string => {
        const m = Math.floor(seconds / 60)
        const s = Math.floor(seconds % 60)
        return `${m}:${s.toString().padStart(2, '0')}`
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || isLoading) return

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date()
        }

        setMessages(prev => [...prev, userMessage])
        setInput('')
        setIsLoading(true)

        try {
            const res = await fetch(`${aiServiceUrl}/api/meeting/ask`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: userMessage.content,
                    classroom_id: classroomId
                })
            })

            if (!res.ok) {
                throw new Error('Failed to get answer')
            }

            const data = await res.json()

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.answer,
                citations: data.citations,
                timestamp: new Date()
            }

            setMessages(prev => [...prev, assistantMessage])
        } catch (error) {
            console.error('Q&A error:', error)
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date()
            }
            setMessages(prev => [...prev, errorMessage])
        } finally {
            setIsLoading(false)
        }
    }

    const handleCitationClick = (citation: Citation) => {
        if (onSeekToTimestamp) {
            onSeekToTimestamp(citation.recording_id, citation.timestamp_start)
        }
    }

    // Floating button when closed
    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-primary-600 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all flex items-center justify-center z-40"
                title="Ask AI about recordings"
            >
                <SparklesIcon className="w-6 h-6" />
            </button>
        )
    }

    return (
        <div className="fixed bottom-6 right-6 w-96 h-[500px] bg-white rounded-2xl shadow-2xl border border-gray-200 flex flex-col z-50 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-primary-600 to-purple-600 text-white">
                <div className="flex items-center gap-2">
                    <SparklesIcon className="w-5 h-5" />
                    <h3 className="font-semibold">Ask AI about Recordings</h3>
                </div>
                <button
                    onClick={() => setIsOpen(false)}
                    className="p-1 hover:bg-white/20 rounded"
                >
                    <XMarkIcon className="w-5 h-5" />
                </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-gray-500 py-8">
                        <ChatBubbleLeftRightIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                        <p className="font-medium">Ask anything about your class recordings</p>
                        <p className="text-sm mt-1">I'll find the answer and show you where to watch it!</p>
                    </div>
                )}

                {messages.map(message => (
                    <div
                        key={message.id}
                        className={clsx(
                            'flex',
                            message.role === 'user' ? 'justify-end' : 'justify-start'
                        )}
                    >
                        <div
                            className={clsx(
                                'max-w-[85%] rounded-2xl px-4 py-2',
                                message.role === 'user'
                                    ? 'bg-primary-600 text-white'
                                    : 'bg-gray-100 text-gray-900'
                            )}
                        >
                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                            {/* Citations */}
                            {message.citations && message.citations.length > 0 && (
                                <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
                                    <p className="text-xs text-gray-500 font-medium">Sources:</p>
                                    {message.citations.map((citation, i) => (
                                        <button
                                            key={i}
                                            onClick={() => handleCitationClick(citation)}
                                            className="block w-full text-left p-2 bg-white rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors"
                                        >
                                            <div className="flex items-center gap-2 text-xs text-primary-600 mb-1">
                                                <PlayIcon className="w-3 h-3" />
                                                <span>{formatTime(citation.timestamp_start)}</span>
                                                {citation.speaker_name && (
                                                    <span className="text-gray-500">• {citation.speaker_name}</span>
                                                )}
                                            </div>
                                            <p className="text-xs text-gray-600 line-clamp-2">
                                                "{citation.text}"
                                            </p>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-gray-100 rounded-2xl px-4 py-3">
                            <div className="flex items-center gap-2">
                                <div className="flex space-x-1">
                                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                </div>
                                <span className="text-sm text-gray-500">Searching recordings...</span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-3 border-t border-gray-200">
                <div className="flex items-center gap-2">
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="What was discussed about...?"
                        className="flex-1 px-4 py-2 bg-gray-100 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="p-2 bg-primary-600 text-white rounded-full disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary-700 transition-colors"
                    >
                        <PaperAirplaneIcon className="w-5 h-5" />
                    </button>
                </div>
            </form>
        </div>
    )
}
