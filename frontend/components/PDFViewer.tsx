'use client'

import { useState, useMemo, useRef, useEffect } from 'react'
import {
    ArrowDownTrayIcon,
    XMarkIcon,
    ArrowTopRightOnSquareIcon,
    ExclamationTriangleIcon,
    DocumentTextIcon,
    SparklesIcon,
    PaperAirplaneIcon,
    ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline'
import { getApiBaseUrl, getAiServiceUrl } from '@/utils/api'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

interface PDFViewerProps {
    pdfUrl: string
    title?: string
    fileSize?: number
    materialId?: string
    classroomId?: string
    onClose?: () => void
}

function formatSize(bytes?: number): string {
    if (!bytes) return ''
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

interface ChatMessage {
    role: 'user' | 'ai'
    text: string
}

export default function PDFViewer({ pdfUrl, title, fileSize, materialId, classroomId, onClose }: PDFViewerProps) {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    // AI Chat state
    const [showChat, setShowChat] = useState(false)
    const [chatInput, setChatInput] = useState('')
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
    const [chatLoading, setChatLoading] = useState(false)
    const chatEndRef = useRef<HTMLDivElement>(null)

    // Fix URL to use the correct API based on current frontend port
    // Port 3000 (run-local.sh) -> API on port 8000
    // Port 4000 (run-lan.sh) -> API on port 9000
    const correctedUrl = useMemo(() => {
        if (!pdfUrl) return pdfUrl

        // Get the current API base URL dynamically
        const apiBaseUrl = getApiBaseUrl()

        // Extract just the filename from the URL
        const match = pdfUrl.match(/\/api\/files\/([^\/]+)$/)
        if (match) {
            const filename = match[1]
            return `${apiBaseUrl}/api/files/${filename}`
        }

        // Fallback: replace any localhost:PORT pattern with current API URL
        return pdfUrl.replace(/https?:\/\/[^\/]+\/api\/files\//, `${apiBaseUrl}/api/files/`)
    }, [pdfUrl])

    const isValidUrl = correctedUrl && correctedUrl !== '#' && correctedUrl.length > 1

    const handleDownload = () => {
        if (!isValidUrl) return
        const link = document.createElement('a')
        link.href = correctedUrl
        link.download = title ? `${title}.pdf` : 'document.pdf'
        link.target = '_blank'
        link.click()
    }

    const handleOpenInNewTab = () => {
        if (!isValidUrl) return
        window.open(correctedUrl, '_blank')
    }

    // Scroll to bottom of chat
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [chatMessages])

    // Quick action prompts
    const quickActions = [
        { label: '📝 Summarize', prompt: 'Summarize this document in bullet points' },
        { label: '📖 Explain like a story', prompt: 'Explain the key concepts from this document like a story' },
        { label: '💡 Key concepts', prompt: 'What are the key concepts and takeaways from this document?' },
        { label: '❓ Quiz me', prompt: 'Create 5 practice questions based on this document' },
    ]

    const sendMessage = async (messageText?: string) => {
        const text = messageText || chatInput.trim()
        if (!text) return

        setChatMessages(prev => [...prev, { role: 'user', text }])
        setChatInput('')
        setChatLoading(true)

        try {
            // Use dynamic AI service URL based on current frontend port
            const aiServiceUrl = getAiServiceUrl()

            console.log('[PDF Chat] Sending request to:', `${aiServiceUrl}/api/tutor/document-chat`)
            console.log('[PDF Chat] Query:', text)
            console.log('[PDF Chat] Document ID:', materialId)
            console.log('[PDF Chat] Classroom ID:', classroomId)

            const res = await fetch(`${aiServiceUrl}/api/tutor/document-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: JSON.stringify({
                    query: text,
                    classroom_id: classroomId,
                    document_id: materialId,
                    document_title: title,
                    context_type: 'pdf_document'
                })
            })

            console.log('[PDF Chat] Response status:', res.status)

            if (res.ok) {
                const data = await res.json()
                console.log('[PDF Chat] Response data:', data)
                setChatMessages(prev => [...prev, {
                    role: 'ai',
                    text: data.answer || data.response || 'I understand. Let me help you with that.'
                }])
            } else {
                setChatMessages(prev => [...prev, {
                    role: 'ai',
                    text: 'Sorry, I encountered an error. Please try again.'
                }])
            }
        } catch (err) {
            console.error('Chat error:', err)
            setChatMessages(prev => [...prev, {
                role: 'ai',
                text: 'Unable to connect to AI service. Make sure the backend is running.'
            }])
        } finally {
            setChatLoading(false)
        }
    }

    // Invalid URL placeholder
    if (!isValidUrl) {
        return (
            <div className="flex flex-col h-full bg-gray-900">
                <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
                    <div className="flex items-center gap-3">
                        <DocumentTextIcon className="w-5 h-5 text-primary-400" />
                        <span className="text-white font-medium truncate max-w-[300px]">
                            {title || 'Document'}
                        </span>
                    </div>
                    {onClose && (
                        <button onClick={onClose} className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded">
                            <XMarkIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-gray-800">
                    <ExclamationTriangleIcon className="w-16 h-16 text-yellow-500 mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Demo Mode</h3>
                    <p className="text-gray-400 mb-6 max-w-md">
                        This is sample/demo data. Upload real PDFs to view them here.
                    </p>
                    <button onClick={onClose} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg">
                        Close
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className="flex flex-col h-full bg-gray-900">
            {/* Compact Single-Row Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                    <DocumentTextIcon className="w-5 h-5 text-primary-400 flex-shrink-0" />
                    <span className="text-white font-medium truncate">{title || 'Document'}</span>
                    {fileSize && <span className="text-gray-500 text-sm flex-shrink-0">{formatSize(fileSize)}</span>}
                </div>

                <div className="flex items-center gap-1 flex-shrink-0">
                    <button onClick={handleOpenInNewTab} className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5" title="Open in new tab">
                        <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                        <span className="hidden sm:inline">Open in Tab</span>
                    </button>
                    <button onClick={handleDownload} className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5" title="Download PDF">
                        <ArrowDownTrayIcon className="w-4 h-4" />
                        <span className="hidden sm:inline">Download</span>
                    </button>
                    {onClose && (
                        <>
                            <div className="w-px h-5 bg-gray-700 mx-1" />
                            <button onClick={onClose} className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded" title="Close">
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* PDF Content */}
            <div className="flex-1 relative bg-gray-700">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
                        <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
                            <p className="text-gray-400">Loading document...</p>
                        </div>
                    </div>
                )}

                <object
                    data={correctedUrl}
                    type="application/pdf"
                    className="w-full h-full"
                    onLoad={() => setLoading(false)}
                    onError={() => { setLoading(false); setError('Failed to load PDF') }}
                >
                    <iframe src={correctedUrl} className="w-full h-full border-0" onLoad={() => setLoading(false)} title={title || 'PDF Document'} />
                </object>

                {error && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800">
                        <p className="text-red-400 mb-4">{error}</p>
                        <button onClick={handleOpenInNewTab} className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg flex items-center gap-2">
                            <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                            Open in New Tab
                        </button>
                    </div>
                )}

                {/* AI Chat Bubble */}
                {!showChat ? (
                    <button
                        onClick={() => setShowChat(true)}
                        className="absolute bottom-4 right-4 w-14 h-14 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white rounded-full shadow-lg flex items-center justify-center transition-all hover:scale-105 z-20"
                        title="Ask AI about this PDF"
                    >
                        <SparklesIcon className="w-7 h-7" />
                    </button>
                ) : (
                    <div className="absolute bottom-4 right-4 w-96 h-[480px] bg-white rounded-xl shadow-2xl flex flex-col overflow-hidden z-20 animate-slide-in">
                        {/* Chat Header */}
                        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white">
                            <div className="flex items-center gap-2">
                                <SparklesIcon className="w-5 h-5" />
                                <span className="font-medium">Ask about this PDF</span>
                            </div>
                            <button onClick={() => setShowChat(false)} className="p-1 hover:bg-white/20 rounded">
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Chat Messages */}
                        <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-gray-50">
                            {chatMessages.length === 0 && (
                                <div className="text-center py-8">
                                    <ChatBubbleLeftRightIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                                    <p className="text-gray-500 text-sm mb-4">Ask me anything about this document!</p>
                                    <div className="flex flex-wrap gap-2 justify-center">
                                        {quickActions.map((action, i) => (
                                            <button
                                                key={i}
                                                onClick={() => sendMessage(action.prompt)}
                                                className="px-3 py-1.5 text-xs bg-white border border-gray-200 text-gray-700 rounded-full hover:bg-gray-100 hover:border-gray-300 transition-colors"
                                            >
                                                {action.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {chatMessages.map((msg, i) => (
                                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[85%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-purple-600 text-white' : 'bg-white border border-gray-200 text-gray-800'}`}>
                                        {msg.role === 'user' ? (
                                            <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                                        ) : (
                                            <div className="text-sm prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5">
                                                <ReactMarkdown
                                                    remarkPlugins={[remarkMath]}
                                                    rehypePlugins={[rehypeKatex]}
                                                >
                                                    {msg.text}
                                                </ReactMarkdown>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {chatLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-white border border-gray-200 p-3 rounded-lg">
                                        <div className="flex items-center gap-2">
                                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>

                        {/* Quick Actions (when conversation started) */}
                        {chatMessages.length > 0 && (
                            <div className="px-3 py-2 border-t border-gray-100 bg-white">
                                <div className="flex flex-wrap gap-1.5">
                                    {['Summarize', 'Key concepts?', 'Quiz me'].map((q, i) => (
                                        <button
                                            key={i}
                                            onClick={() => sendMessage(q)}
                                            className="px-2.5 py-1 text-xs bg-gray-100 text-gray-600 rounded-full hover:bg-gray-200"
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Chat Input */}
                        <div className="p-3 border-t border-gray-200 bg-white">
                            <div className="flex gap-2 items-center">
                                <input
                                    type="text"
                                    value={chatInput}
                                    onChange={(e) => setChatInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                                    placeholder="Ask about this PDF..."
                                    className="flex-1 px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent focus:outline-none text-sm"
                                />
                                <button
                                    onClick={() => sendMessage()}
                                    disabled={chatLoading || !chatInput.trim()}
                                    className="p-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <PaperAirplaneIcon className="w-5 h-5" />
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <style jsx>{`
                @keyframes slide-in {
                    from { transform: translateY(20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                .animate-slide-in {
                    animation: slide-in 0.2s ease-out;
                }
            `}</style>
        </div>
    )
}
