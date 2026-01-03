'use client'
import { getApiBaseUrl, getAiServiceUrl } from '@/utils/api'


import { useState, useEffect, useRef } from 'react'
import {
    PaperAirplaneIcon,
    AcademicCapIcon,
    BookOpenIcon,
    ClockIcon,
    StarIcon,
    ChevronDownIcon,
    ChevronUpIcon,
    DocumentTextIcon,
    LightBulbIcon,
    ArrowPathIcon,
    TrashIcon,
    PlusIcon,
    MagnifyingGlassIcon,
    SparklesIcon,
    PhotoIcon,
    Bars3Icon,
    ChatBubbleLeftRightIcon,
    DocumentIcon,
    VideoCameraIcon,
    GlobeAltIcon,
    LinkIcon,
    XMarkIcon,
    ArrowLeftIcon,
    ArrowsPointingOutIcon,
    ArrowTopRightOnSquareIcon,
    ExclamationTriangleIcon,
    Squares2X2Icon,
    MapIcon
} from '@heroicons/react/24/outline'
import { StarIcon as StarSolidIcon } from '@heroicons/react/24/solid'
import { logClick, logInput, logError, logApiCall, logApiResponse, logScroll } from '@/utils/logger'
import dynamic from 'next/dynamic'

// Dynamic import for MindmapViewer (uses mermaid which needs client-side only)
const MindmapViewer = dynamic(() => import('@/components/viewers/MindmapViewer'), { ssr: false })

// Dynamic import for MarkdownRenderer (uses KaTeX which needs client-side only)
const MarkdownRenderer = dynamic(() => import('@/components/chat/MarkdownRenderer'), { ssr: false })

// ============================================================================
// Types
// ============================================================================

interface Source {
    document_id: string
    chunk_id: string
    title: string
    similarity_score: number
}

interface TutorResponse {
    answer_short: string
    answer_detailed: string | null
    sources: Source[]
    confidence_score: number
    recommended_actions: string[]
}

interface Message {
    id: string
    type: 'user' | 'assistant'
    content: string
    response?: TutorResponse
    timestamp: Date
    subject?: string
}

interface SourceItem {
    id: string
    type: 'pdf' | 'note' | 'article' | 'video' | 'image' | 'pptx' | 'webpage' | 'flowchart'
    title: string
    url?: string
    thumbnailUrl?: string
    embedUrl?: string
    relevance: number
    snippet?: string
    source?: string
    duration?: string
    fileSize?: string
    cachedContent?: string
    cachedSummary?: string
    cachedImages?: string[]
    mermaidCode?: string
    // New fields for dynamic web resources
    trustScore?: number      // 0.0 - 1.0 trust score
    sourceType?: string      // 'encyclopedia', 'academic', 'educational', etc.
}

interface Conversation {
    id: string
    title: string
    messages: Message[]
    pinned: boolean
    createdAt: Date
    updatedAt: Date
    sources?: SourceItem[]
}

// ============================================================================
// Component
// ============================================================================

export default function AITutorPage() {
    // State
    const [conversations, setConversations] = useState<Conversation[]>([])
    const [activeConversation, setActiveConversation] = useState<Conversation | null>(null)
    const [input, setInput] = useState('')
    const [subject, setSubject] = useState<string>('general')
    const [selectedClassroom, setSelectedClassroom] = useState<string>('')
    const [classrooms, setClassrooms] = useState<any[]>([])
    const [responseMode, setResponseMode] = useState<'short' | 'detailed'>('short')
    const [languageStyle, setLanguageStyle] = useState<'scientific' | 'layman' | 'simple'>('layman')
    const [loading, setLoading] = useState(false)
    const [showHistory, setShowHistory] = useState(true)
    const [expandedMessages, setExpandedMessages] = useState<Set<string>>(new Set())
    const [searchQuery, setSearchQuery] = useState('')

    // Additional options
    const [findResources, setFindResources] = useState(false)
    const [useNotes, setUseNotes] = useState(true)
    const [chatImage, setChatImage] = useState<string | null>(null)

    // Reasoning/Thinking state
    interface ReasoningStep {
        id: string
        type: 'searching' | 'analyzing' | 'reading' | 'thinking' | 'complete'
        title: string
        detail?: string
        status: 'active' | 'done'
    }
    const [reasoningSteps, setReasoningSteps] = useState<ReasoningStep[]>([])
    const [showReasoning, setShowReasoning] = useState(true)

    // Sources sidebar state (SourceItem defined at top level)
    const [sources, setSources] = useState<SourceItem[]>([])
    const [showSources, setShowSources] = useState(false)  // Start closed, auto-open when resources found
    const [activeSource, setActiveSource] = useState<SourceItem | null>(null)  // For preview panel

    // Embedded viewer state
    const [viewerMode, setViewerMode] = useState<'list' | 'viewer'>('list')
    const [viewerLoading, setViewerLoading] = useState(false)
    const [viewerError, setViewerError] = useState<string | null>(null)
    const viewerRef = useRef<HTMLIFrameElement>(null)

    // Sidebar filter state
    type SourceFilter = 'all' | 'documents' | 'videos' | 'websites' | 'flowcharts' | 'images'
    const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')
    const [sidebarWidth, setSidebarWidth] = useState(500) // Default wider for better viewing
    const [isResizing, setIsResizing] = useState(false)

    // Image zoom state
    const [imageZoom, setImageZoom] = useState(100)
    const [imagePan, setImagePan] = useState({ x: 0, y: 0 })
    const [isDragging, setIsDragging] = useState(false)
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

    // Turn off loading immediately for document types (they don't have iframes/images to trigger onLoad)
    useEffect(() => {
        if (viewerMode === 'viewer' && activeSource && ['pdf', 'pptx', 'note', 'docx'].includes(activeSource.type)) {
            setViewerLoading(false)
        }
    }, [viewerMode, activeSource])

    // Source category counts
    const getSourceCounts = () => {
        const documents = sources.filter(s => ['pdf', 'pptx', 'note', 'docx'].includes(s.type)).length
        const videos = sources.filter(s => s.type === 'video').length
        const websites = sources.filter(s => s.type === 'article' || s.type === 'webpage').length
        const flowcharts = sources.filter(s => s.type === 'flowchart' as any).length
        const images = sources.filter(s => s.type === 'image').length
        return { documents, videos, websites, flowcharts, images, all: sources.length }
    }

    // Filter sources by category
    const getFilteredSources = () => {
        if (sourceFilter === 'all') return sources
        if (sourceFilter === 'documents') return sources.filter(s => ['pdf', 'pptx', 'note', 'docx'].includes(s.type))
        if (sourceFilter === 'videos') return sources.filter(s => s.type === 'video')
        if (sourceFilter === 'websites') return sources.filter(s => s.type === 'article' || s.type === 'webpage')
        if (sourceFilter === 'flowcharts') return sources.filter(s => s.type === 'flowchart' as any)
        if (sourceFilter === 'images') return sources.filter(s => s.type === 'image')
        return sources
    }

    // Sidebar resize handler
    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault()
        setIsResizing(true)
    }

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizing) return
            const newWidth = window.innerWidth - e.clientX
            setSidebarWidth(Math.max(320, Math.min(900, newWidth))) // Allow wider sidebar
        }
        const handleMouseUp = () => setIsResizing(false)

        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove)
            document.addEventListener('mouseup', handleMouseUp)
        }
        return () => {
            document.removeEventListener('mousemove', handleMouseMove)
            document.removeEventListener('mouseup', handleMouseUp)
        }
    }, [isResizing])

    const messagesEndRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLTextAreaElement>(null)

    // Load conversations from localStorage
    useEffect(() => {
        const saved = localStorage.getItem('ai_tutor_conversations')
        if (saved) {
            const parsed = JSON.parse(saved)
            setConversations(parsed.map((c: any) => ({
                ...c,
                createdAt: new Date(c.createdAt),
                updatedAt: new Date(c.updatedAt),
                messages: c.messages.map((m: any) => ({
                    ...m,
                    timestamp: new Date(m.timestamp)
                }))
            })))
        }
    }, [])

    // Fetch classrooms
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
            }
        }
        fetchClassrooms()
    }, [])

    // Save conversations
    useEffect(() => {
        if (conversations.length > 0) {
            localStorage.setItem('ai_tutor_conversations', JSON.stringify(conversations))
        }
    }, [conversations])

    // Scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [activeConversation?.messages])

    // Create new conversation
    const createNewConversation = () => {
        logClick('new_conversation_btn', 'Creating new conversation')
        const newConv: Conversation = {
            id: `conv_${Date.now()}`,
            title: 'New Conversation',
            messages: [],
            pinned: false,
            createdAt: new Date(),
            updatedAt: new Date()
        }
        setConversations(prev => [newConv, ...prev])
        setActiveConversation(newConv)
        setSources([])  // Clear sources for new conversation
        setShowSources(false)
    }

    // Select conversation and restore sources
    const selectConversation = (conv: Conversation) => {
        setActiveConversation(conv)
        // Restore sources from the conversation
        if (conv.sources && conv.sources.length > 0) {
            setSources(conv.sources)
            setShowSources(true)
        } else {
            setSources([])
            setShowSources(false)
        }
        setActiveSource(null)
        setViewerMode('list')
    }

    // Open content in viewer with enhanced UX (auto-expand sidebar, collapse history)
    const openContentViewer = (source: SourceItem, options?: { resetZoom?: boolean }) => {
        setActiveSource(source)
        setViewerMode('viewer')
        setViewerError(null)

        // Set loading state based on content type
        const needsLoading = ['video', 'article', 'webpage', 'image'].includes(source.type)
        setViewerLoading(needsLoading)

        // Reset zoom for images
        if (options?.resetZoom) {
            setImageZoom(100)
            setImagePan({ x: 0, y: 0 })
        }

        // Auto-expand sidebar to max width for all content types
        if (['pdf', 'pptx', 'docx', 'video', 'flowchart', 'article', 'webpage', 'image'].includes(source.type)) {
            setSidebarWidth(900) // Max width for better viewing
            setShowHistory(false) // Collapse history panel
        }
    }

    // Toggle pin
    const togglePin = (convId: string) => {
        logClick('pin_conversation', `Toggling pin for ${convId}`)
        setConversations(prev => prev.map(c =>
            c.id === convId ? { ...c, pinned: !c.pinned } : c
        ))
    }

    // Delete conversation
    const deleteConversation = (convId: string) => {
        setConversations(prev => prev.filter(c => c.id !== convId))
        if (activeConversation?.id === convId) {
            setActiveConversation(null)
        }
    }

    // Toggle message expansion
    const toggleExpand = (messageId: string) => {
        logClick('toggle_expand', `Expanding message ${messageId}`)
        setExpandedMessages(prev => {
            const next = new Set(prev)
            if (next.has(messageId)) {
                next.delete(messageId)
            } else {
                next.add(messageId)
            }
            return next
        })
    }

    // Send message - accepts optional question parameter and optional forceWebSearch flag
    const sendMessage = async (quickQuestion?: string, forceWebSearch = false) => {
        const messageToSend = quickQuestion || input
        if (!messageToSend.trim() || loading) return

        let conv = activeConversation
        if (!conv) {
            const newConv: Conversation = {
                id: `conv_${Date.now()}`,
                title: messageToSend.slice(0, 50) + (messageToSend.length > 50 ? '...' : ''),
                messages: [],
                pinned: false,
                createdAt: new Date(),
                updatedAt: new Date()
            }
            conv = newConv
            setConversations(prev => [newConv, ...prev])
            setActiveConversation(newConv)
        }

        // Add user message
        const userMessage: Message = {
            id: `msg_${Date.now()}`,
            type: 'user',
            content: messageToSend,
            timestamp: new Date(),
            subject
        }

        const updatedConv = {
            ...conv,
            messages: [...conv.messages, userMessage],
            title: conv.messages.length === 0 ? messageToSend.slice(0, 50) : conv.title,
            updatedAt: new Date()
        }

        setConversations(prev => prev.map(c => c.id === conv!.id ? updatedConv : c))
        setActiveConversation(updatedConv)
        setInput('')
        setLoading(true)
        setReasoningSteps([])

        // Helper to add reasoning steps with delays
        const addReasoningStep = async (step: Omit<ReasoningStep, 'status'>) => {
            setReasoningSteps(prev => [...prev, { ...step, status: 'active' }])
            await new Promise(resolve => setTimeout(resolve, 600 + Math.random() * 400))
            setReasoningSteps(prev => prev.map(s => s.id === step.id ? { ...s, status: 'done' } : s))
        }

        // Simulate reasoning steps based on options
        const simulateReasoning = async () => {
            // Step 1: Understanding query
            await addReasoningStep({
                id: 'step_1',
                type: 'thinking',
                title: 'Understanding your question',
                detail: `Analyzing: "${messageToSend.slice(0, 50)}${messageToSend.length > 50 ? '...' : ''}"`
            })

            // Step 2: Check subject context
            if (subject !== 'general') {
                await addReasoningStep({
                    id: 'step_2',
                    type: 'analyzing',
                    title: `Focusing on ${subject.charAt(0).toUpperCase() + subject.slice(1)}`,
                    detail: 'Applying subject-specific knowledge'
                })
            }

            // Step 3: Search classroom notes if enabled
            if (useNotes) {
                await addReasoningStep({
                    id: 'step_3',
                    type: 'searching',
                    title: 'Searching your classroom materials',
                    detail: 'Looking through uploaded notes and documents...'
                })
                await addReasoningStep({
                    id: 'step_3b',
                    type: 'reading',
                    title: 'Found relevant documents',
                    detail: 'Reading "Chapter 5 - Fundamentals.pdf", "Study Notes.docx"'
                })
            }

            // Step 4: Find external resources if enabled
            if (findResources) {
                await addReasoningStep({
                    id: 'step_4',
                    type: 'searching',
                    title: 'Searching educational resources',
                    detail: 'khanacademy.org, wikipedia.org, coursera.org...'
                })
                await addReasoningStep({
                    id: 'step_4b',
                    type: 'reading',
                    title: 'Reviewing external sources',
                    detail: 'Found 3 reliable sources with relevant information'
                })
            }

            // Step 5: Analyze image if present
            if (chatImage) {
                await addReasoningStep({
                    id: 'step_5',
                    type: 'analyzing',
                    title: 'Analyzing uploaded image',
                    detail: 'Extracting text and visual information...'
                })
            }

            // Step 6: Formulating response
            await addReasoningStep({
                id: 'step_6',
                type: 'thinking',
                title: 'Formulating response',
                detail: responseMode === 'detailed' ? 'Creating comprehensive explanation...' : 'Creating concise answer...'
            })
        }

        // Start reasoning simulation

        try {
            // Run reasoning simulation concurrently with API call
            const reasoningPromise = simulateReasoning()

            // Call AI Tutor API with timeout
            console.log('[AI-TUTOR] Starting API call with question:', messageToSend.slice(0, 50))
            logApiCall('/api/ai-tutor/query', 'POST', { question: messageToSend, subject, find_resources: findResources })
            const startTime = Date.now()

            // Create timeout controller
            const apiController = new AbortController()
            const apiTimeoutId = setTimeout(() => {
                console.warn('[AI-TUTOR] ⚠ API timeout after 60s')
                apiController.abort()
            }, 60000)  // Increased from 15s to 60s for web crawling

            // Build conversation history for context (last 4 messages for follow-ups)
            const conversationHistory = updatedConv.messages.slice(-4).map(m => ({
                role: m.type === 'user' ? 'user' : 'assistant',
                content: m.content
            }))

            const res = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: localStorage.getItem('userId') || 'anonymous',
                    question: messageToSend,
                    subject: subject !== 'general' ? subject : undefined,
                    classroom_id: selectedClassroom || undefined,
                    response_mode: responseMode,
                    language_style: languageStyle,
                    find_resources: forceWebSearch || findResources,
                    conversation_history: conversationHistory.length > 0 ? conversationHistory : undefined
                }),
                signal: apiController.signal
            })

            clearTimeout(apiTimeoutId)
            console.log('[AI-TUTOR] API call completed, status:', res.status)

            // Wait for reasoning to complete
            await reasoningPromise
            logApiResponse('/api/ai-tutor/query', res.status, Date.now() - startTime)

            const data = await res.json()

            // Initialize sources array outside the conditional
            let allSources: SourceItem[] = []

            // Populate sources from API response if available
            if (data.success && data.data) {

                // Add classroom sources (documents)
                if (data.data.sources) {
                    const classroomSources = data.data.sources.map((s: { document_id: string; title: string; similarity_score: number }, i: number) => ({
                        id: `doc_${i}`,
                        type: 'pdf' as const,
                        title: s.title,
                        relevance: Math.round(s.similarity_score * 100),
                        snippet: 'From your classroom materials',
                        source: 'Classroom Materials'
                    }))
                    allSources.push(...classroomSources)
                }

                // Add web resources if available
                if (data.data.web_resources) {
                    const wr = data.data.web_resources

                    // Videos
                    if (wr.videos) {
                        const videoSources = wr.videos.map((v: { id: string; title: string; url: string; thumbnailUrl: string; embedUrl: string; duration: string; source: string; relevance: number }) => ({
                            id: v.id,
                            type: 'video' as const,
                            title: v.title,
                            url: v.url,
                            thumbnailUrl: v.thumbnailUrl,
                            embedUrl: v.embedUrl,
                            duration: v.duration,
                            relevance: v.relevance,
                            source: v.source || 'YouTube'
                        }))
                        allSources.push(...videoSources)
                    }

                    // Images
                    if (wr.images) {
                        const imageSources = wr.images.map((img: { id: string; title: string; url: string; thumbnailUrl: string; source: string; relevance: number }) => ({
                            id: img.id,
                            type: 'image' as const,
                            title: img.title,
                            url: img.url,
                            thumbnailUrl: img.thumbnailUrl,
                            relevance: img.relevance,
                            source: img.source || 'Web Images'
                        }))
                        allSources.push(...imageSources)
                    }

                    // Articles (with cached content from Readability extraction)
                    if (wr.articles) {
                        const articleSources = wr.articles.map((a: { id: string; type: string; title: string; url: string; snippet: string; source: string; relevance: number; cachedContent?: string; cachedSummary?: string; cachedImages?: string[] }) => ({
                            id: a.id,
                            type: (a.type === 'webpage' ? 'webpage' : 'article') as 'article' | 'webpage',
                            title: a.title,
                            url: a.url,
                            snippet: a.cachedSummary || a.snippet,
                            relevance: a.relevance,
                            source: a.source || 'Web',
                            cachedContent: a.cachedContent,
                            cachedSummary: a.cachedSummary,
                            cachedImages: a.cachedImages || []
                        }))
                        allSources.push(...articleSources)
                    }
                }

                // Add flowchart if returned from API
                if (data.data.flowchart_mermaid) {
                    allSources.push({
                        id: `flowchart_${Date.now()}`,
                        type: 'flowchart' as const,
                        title: 'Concept Flowchart',
                        relevance: 100,
                        snippet: 'Visual explanation of the concept',
                        source: 'AI Generated',
                        mermaidCode: data.data.flowchart_mermaid
                    })
                }

                setSources(allSources)
                if (allSources.length > 0) {
                    setShowSources(true)
                }
            }

            // Create assistant message
            const assistantMessage: Message = {
                id: `msg_${Date.now() + 1}`,
                type: 'assistant',
                content: data.success ? data.data.answer_short : data.error?.message || 'Failed to get answer',
                response: data.success ? data.data : undefined,
                timestamp: new Date()
            }

            const finalConv = {
                ...updatedConv,
                messages: [...updatedConv.messages, assistantMessage],
                updatedAt: new Date(),
                sources: allSources.length > 0 ? allSources : updatedConv.sources  // Preserve sources
            }

            setConversations(prev => prev.map(c => c.id === conv!.id ? finalConv : c))
            setActiveConversation(finalConv)

        } catch (error: any) {
            // Log the error
            if (error.name === 'AbortError') {
                console.warn('[AI-TUTOR] ⚠ Request timed out, using mock response')
            } else {
                logError('ai_tutor_query', error)
                console.error('[AI-TUTOR] API error:', error)
            }

            // Skip reasoning simulation if already done (from concurrent promise)
            // Use mock response for fallback

            // Mock response for development
            const mockResponse: TutorResponse = {
                answer_short: "Based on the study materials, this concept involves fundamental principles that build upon previous knowledge. [Source 1]",
                answer_detailed: "This is a comprehensive explanation of the topic...\n\nAccording to [Source 1], the key principles are:\n1. First principle\n2. Second principle\n3. Third principle\n\nFurther details from [Source 2] explain the practical applications.",
                sources: [
                    { document_id: 'doc1', chunk_id: 'ch1', title: 'Chapter 5 - Fundamentals', similarity_score: 0.94 },
                    { document_id: 'doc2', chunk_id: 'ch2', title: 'Practice Guide', similarity_score: 0.87 }
                ],
                confidence_score: 0.89,
                recommended_actions: [
                    'Review the foundational concepts',
                    'Try 3-5 practice problems',
                    'Watch supplementary video'
                ]
            }

            // Populate sources sidebar with mock data (simulating crawled resources)
            const mockSources: SourceItem[] = []

            // Always include classroom materials if useNotes is on
            if (useNotes) {
                mockSources.push(
                    {
                        id: 's1',
                        type: 'pdf',
                        title: 'Chapter 5 - Fundamentals.pdf',
                        relevance: 94,
                        snippet: 'The core principles of this concept involve understanding the basic building blocks...',
                        source: 'Classroom Materials',
                        fileSize: '2.4 MB'
                    },
                    {
                        id: 's2',
                        type: 'note',
                        title: 'Class Notes - Week 3',
                        relevance: 87,
                        snippet: 'Key takeaways from the lecture on this topic...',
                        source: 'Your Notes'
                    },
                    {
                        id: 's3',
                        type: 'pptx',
                        title: 'Lecture Slides - Introduction.pptx',
                        relevance: 85,
                        snippet: 'Slides covering the introduction and main concepts...',
                        source: 'Classroom Materials',
                        fileSize: '5.1 MB'
                    }
                )
            }

            // Add web resources if findResources is on - USE REAL API
            if (findResources) {
                try {
                    // Call the new web ingest API with 10s timeout
                    const controller = new AbortController()
                    const timeoutId = setTimeout(() => controller.abort(), 30000)  // 30s for web crawling

                    const webResponse = await fetch('${getAiServiceUrl()}/api/resources/web', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: messageToSend,
                            subject: subject,
                            max_sources: 3  // Reduced for faster response
                        }),
                        signal: controller.signal
                    })

                    clearTimeout(timeoutId)

                    if (webResponse.ok) {
                        const webData = await webResponse.json()

                        if (webData.success && webData.resources) {
                            // Convert API response to source items
                            webData.resources.forEach((resource: any, index: number) => {
                                if (resource.clean_content && !resource.error) {
                                    mockSources.push({
                                        id: resource.id || `web_${index}`,
                                        type: 'article',
                                        title: resource.title,
                                        url: resource.url,
                                        relevance: Math.round(resource.trust_score * 100),
                                        snippet: resource.summary?.slice(0, 200) + '...',
                                        source: resource.source_name,
                                        trustScore: resource.trust_score,
                                        sourceType: resource.source_type,
                                        cachedContent: resource.clean_content
                                    })
                                }
                            })

                            console.log(`✓ Fetched ${webData.resources.length} web resources (${webData.total_chunks_stored} chunks stored)`)
                        }
                    }
                } catch (err: any) {
                    if (err.name === 'AbortError') {
                        console.warn('Web resources fetch timed out after 10s, continuing without web resources')
                    } else {
                        console.error('Web resources fetch error:', err)
                    }
                    // Continue without web resources
                }
            }

            setSources(mockSources)

            // Auto-open sources sidebar if we have resources (either from classroom or web)
            if (mockSources.length > 0) {
                setShowSources(true)
            }

            const assistantMessage: Message = {
                id: `msg_${Date.now() + 1}`,
                type: 'assistant',
                content: mockResponse.answer_short,
                response: mockResponse,
                timestamp: new Date()
            }

            const finalConv = {
                ...updatedConv,
                messages: [...updatedConv.messages, assistantMessage],
                updatedAt: new Date(),
                sources: mockSources.length > 0 ? mockSources : updatedConv.sources  // Preserve sources
            }

            setConversations(prev => prev.map(c => c.id === conv!.id ? finalConv : c))
            setActiveConversation(finalConv)
            console.log('[AI-TUTOR] ✅ Response complete, message added')
        }

        console.log('[AI-TUTOR] Setting loading=false')
        setLoading(false)
        setChatImage(null)
        setReasoningSteps([])
    }

    // Filter conversations
    const filteredConversations = conversations
        .filter(c => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
        .sort((a, b) => {
            if (a.pinned !== b.pinned) return a.pinned ? -1 : 1
            return b.updatedAt.getTime() - a.updatedAt.getTime()
        })

    // Handle image upload
    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader()
            reader.onload = (event) => {
                setChatImage(event.target?.result as string)
            }
            reader.readAsDataURL(file)
        }
        e.target.value = ''
    }

    return (
        <div className="flex h-[calc(100vh-0px)] bg-gray-100 overflow-hidden">
            {/* History Sidebar */}
            <div className={`${showHistory ? 'w-72' : 'w-0'} transition-all duration-300 overflow-hidden flex-shrink-0 bg-gray-50 border-r border-gray-200`}>
                <div className="h-full flex flex-col w-72">
                    {/* Header */}
                    <div className="p-3 border-b border-gray-200">
                        <button
                            onClick={createNewConversation}
                            className="w-full flex items-center gap-3 p-3 text-gray-700 bg-white hover:bg-gray-100 rounded-lg transition-colors border border-gray-200"
                        >
                            <PlusIcon className="w-5 h-5" />
                            <span className="text-sm font-medium">New Chat</span>
                        </button>
                    </div>

                    {/* Search */}
                    <div className="p-3">
                        <div className="relative">
                            <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Search conversations..."
                                className="w-full pl-9 pr-3 py-2 bg-white rounded-lg text-sm text-gray-900 placeholder-gray-400 border border-gray-200 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            />
                        </div>
                    </div>

                    {/* Conversations List */}
                    <div className="flex-1 overflow-y-auto px-2 pb-2">
                        {filteredConversations.length === 0 ? (
                            <div className="text-center py-8 text-gray-500 text-sm">
                                No conversations yet
                            </div>
                        ) : (
                            filteredConversations.map(conv => (
                                <div
                                    key={conv.id}
                                    onClick={() => selectConversation(conv)}
                                    className={`group flex items-center gap-3 p-3 rounded-lg cursor-pointer mb-1 ${activeConversation?.id === conv.id
                                        ? 'bg-primary-50 text-primary-700 border border-primary-200'
                                        : 'text-gray-700 hover:bg-gray-100'
                                        }`}
                                >
                                    <ChatBubbleLeftRightIcon className="w-5 h-5 flex-shrink-0" />
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm truncate">{conv.title}</p>
                                    </div>
                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                                        <button
                                            onClick={(e) => { e.stopPropagation(); togglePin(conv.id) }}
                                            className="p-1 hover:bg-gray-200 rounded"
                                        >
                                            {conv.pinned ? (
                                                <StarSolidIcon className="w-4 h-4 text-yellow-500" />
                                            ) : (
                                                <StarIcon className="w-4 h-4 text-gray-400" />
                                            )}
                                        </button>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id) }}
                                            className="p-1 hover:bg-red-50 rounded text-gray-400 hover:text-red-500"
                                        >
                                            <TrashIcon className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-white relative min-w-0 overflow-hidden">
                {/* Toggle History Button */}
                <button
                    onClick={() => setShowHistory(!showHistory)}
                    className="absolute top-4 left-4 z-10 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                    title={showHistory ? 'Hide sidebar' : 'Show sidebar'}
                >
                    <Bars3Icon className="w-5 h-5" />
                </button>

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto">
                    {!activeConversation || activeConversation.messages.length === 0 ? (
                        /* Empty State */
                        <div className="h-full flex items-center justify-center">
                            <div className="text-center max-w-2xl px-6">
                                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center">
                                    <AcademicCapIcon className="w-10 h-10 text-white" />
                                </div>
                                <h1 className="text-4xl font-bold text-gray-900 mb-3">AI Tutor</h1>
                                <p className="text-gray-500 mb-8 text-lg">
                                    Ask me anything about your studies. I'll help you understand concepts, solve problems, and learn effectively.
                                </p>
                                <div className="grid grid-cols-2 gap-3 max-w-xl mx-auto">
                                    {[
                                        "Explain photosynthesis step by step",
                                        "What is Newton's first law of motion?",
                                        "Help me understand quadratic equations",
                                        "Summarize the causes of French Revolution"
                                    ].map((q, i) => (
                                        <button
                                            key={i}
                                            onClick={() => setInput(q)}
                                            className="p-4 text-left text-sm bg-gray-50 hover:bg-gray-100 rounded-xl text-gray-700 border border-gray-200 transition-colors"
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        /* Messages */
                        <div className="max-w-3xl mx-auto py-6 px-4 space-y-6">
                            {activeConversation.messages.map((msg, msgIndex) => (
                                <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    {msg.type === 'user' ? (
                                        /* User Message */
                                        <div className="max-w-2xl">
                                            <div className="bg-primary-600 text-white px-4 py-3 rounded-2xl rounded-br-sm">
                                                {msg.content}
                                            </div>
                                        </div>
                                    ) : (
                                        /* Assistant Message */
                                        <div className="max-w-3xl w-full space-y-3">
                                            {/* Answer Card */}
                                            {/* Answer Card - Clean, no border */}
                                            <div className="bg-transparent pl-1 py-2 font-sans">
                                                {/* Show the answer with enhanced markdown rendering */}
                                                <div className="text-gray-900">
                                                    <MarkdownRenderer content={msg.response?.answer_detailed || msg.content} />
                                                </div>

                                                {/* Source Chips - ChatGPT style */}
                                                {sources.length > 0 && (
                                                    <div className="mt-4 flex flex-wrap gap-2">
                                                        {sources.slice(0, 4).map((source, idx) => (
                                                            <button
                                                                key={source.id}
                                                                onClick={() => {
                                                                    openContentViewer(source)
                                                                    setShowSources(true)
                                                                }}
                                                                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-200 rounded-lg text-xs font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-300 transition-colors shadow-sm"
                                                            >
                                                                <span className="w-4 h-4 rounded bg-gradient-to-br from-primary-500 to-secondary-500 text-white flex items-center justify-center text-[10px] font-bold">
                                                                    {idx + 1}
                                                                </span>
                                                                <span className="truncate max-w-[150px]">{source.source || source.title.split(' - ')[0]}</span>
                                                                {source.trustScore && (
                                                                    <span className={`ml-1 px-1.5 py-0.5 rounded text-[10px] ${source.trustScore >= 0.9 ? 'bg-green-100 text-green-700' :
                                                                        source.trustScore >= 0.8 ? 'bg-blue-100 text-blue-700' :
                                                                            'bg-gray-100 text-gray-600'
                                                                        }`}>
                                                                        {Math.round(source.trustScore * 100)}%
                                                                    </span>
                                                                )}
                                                            </button>
                                                        ))}
                                                        {sources.length > 4 && (
                                                            <button
                                                                onClick={() => setShowSources(true)}
                                                                className="inline-flex items-center gap-1 px-3 py-1.5 bg-gray-100 border border-gray-200 rounded-lg text-xs font-medium text-gray-600 hover:bg-gray-200 transition-colors"
                                                            >
                                                                +{sources.length - 4} more
                                                            </button>
                                                        )}
                                                    </div>
                                                )}

                                                {/* Confidence Score */}
                                                {msg.response && (
                                                    <div className="mt-4 flex items-center gap-2">
                                                        <div className={`px-2 py-1 rounded-full text-xs font-medium ${msg.response.confidence_score >= 0.8
                                                            ? 'bg-green-100 text-green-700'
                                                            : msg.response.confidence_score >= 0.6
                                                                ? 'bg-yellow-100 text-yellow-700'
                                                                : 'bg-red-100 text-red-700'
                                                            }`}>
                                                            {Math.round(msg.response.confidence_score * 100)}% confident
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Suggestion Chips - ChatGPT style */}
                                                {msg.type === 'assistant' && msg.response && (
                                                    <div className="mt-5 pt-4 border-t border-gray-100">
                                                        {/* Explain Simpler Button */}
                                                        <div className="flex items-center gap-2 mb-3">
                                                            <button
                                                                onClick={() => {
                                                                    // Extract core topic from previous user message
                                                                    const prevMsg = msgIndex > 0 ? activeConversation.messages[msgIndex - 1] : null
                                                                    if (prevMsg?.type === 'user') {
                                                                        // Smart topic extraction: remove stopwords and take key terms
                                                                        const stopwords = ['explain', 'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'tell', 'me', 'about', 'summarize', 'describe']
                                                                        const words = prevMsg.content.toLowerCase().split(/\s+/)
                                                                        const keyWords = words.filter(w => !stopwords.includes(w) && w.length > 3)
                                                                        const topic = keyWords.slice(0, 4).join(' ') || prevMsg.content.slice(0, 30)

                                                                        // Send clean query with web search forced
                                                                        sendMessage(`Explain ${topic} in simpler terms`, true)
                                                                    } else {
                                                                        sendMessage('Explain this in simpler terms', true)
                                                                    }
                                                                }}
                                                                disabled={loading}
                                                                className="inline-flex items-center gap-1.5 px-3 py-2 bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-xl text-sm font-medium text-purple-700 hover:from-purple-100 hover:to-indigo-100 hover:border-purple-300 transition-all shadow-sm disabled:opacity-50"
                                                            >
                                                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                                                </svg>
                                                                Explain in simpler terms
                                                            </button>
                                                        </div>

                                                        {/* Follow-up Questions */}
                                                        <div className="space-y-2">
                                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Students also ask</span>
                                                            <div className="flex flex-wrap gap-2">
                                                                {(() => {
                                                                    // Generate contextual follow-up questions based on the answer content
                                                                    const content = (msg.response?.answer_detailed || msg.content).toLowerCase()
                                                                    const followUpQuestions: string[] = []

                                                                    // Topic-specific patterns
                                                                    if (content.includes('photosynthesis') || content.includes('plant')) {
                                                                        followUpQuestions.push("What happens during the Calvin cycle?")
                                                                        followUpQuestions.push("Why is chlorophyll green?")
                                                                        followUpQuestions.push("How do plants respire at night?")
                                                                    } else if (content.includes('equation') || content.includes('formula')) {
                                                                        followUpQuestions.push("Can you show a worked example?")
                                                                        followUpQuestions.push("What are the common mistakes to avoid?")
                                                                    } else if (content.includes('history') || content.includes('war') || content.includes('century')) {
                                                                        followUpQuestions.push("What were the main causes?")
                                                                        followUpQuestions.push("What were the consequences?")
                                                                    } else if (content.includes('python') || content.includes('code') || content.includes('programming')) {
                                                                        followUpQuestions.push("Can you show a code example?")
                                                                        followUpQuestions.push("What are common errors to avoid?")
                                                                    } else {
                                                                        // Generic follow-ups
                                                                        followUpQuestions.push("Can you give an example?")
                                                                        followUpQuestions.push("Why is this important?")
                                                                        followUpQuestions.push("How is this used in real life?")
                                                                    }

                                                                    return followUpQuestions.slice(0, 3).map((q, idx) => (
                                                                        <button
                                                                            key={idx}
                                                                            onClick={() => sendMessage(q)}
                                                                            disabled={loading}
                                                                            className="inline-flex items-center gap-1 px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-lg text-xs text-gray-600 hover:bg-gray-100 hover:border-gray-300 transition-colors disabled:opacity-50"
                                                                        >
                                                                            <span className="text-gray-400">→</span>
                                                                            {q}
                                                                        </button>
                                                                    ))
                                                                })()}
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>


                                        </div>
                                    )}
                                </div>
                            ))}

                            {/* Reasoning/Thinking Display */}
                            {loading && (
                                <div className="flex justify-start">
                                    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden max-w-md w-full">
                                        {/* Header */}
                                        <button
                                            onClick={() => setShowReasoning(!showReasoning)}
                                            className="w-full px-4 py-3 flex items-center justify-between bg-gradient-to-r from-primary-50 to-secondary-50 border-b border-gray-100"
                                        >
                                            <div className="flex items-center gap-2">
                                                <SparklesIcon className="w-5 h-5 text-primary-600 animate-pulse" />
                                                <span className="font-medium text-gray-900">Thinking...</span>
                                            </div>
                                            <ChevronDownIcon className={`w-4 h-4 text-gray-500 transition-transform ${showReasoning ? 'rotate-180' : ''}`} />
                                        </button>

                                        {/* Steps List */}
                                        {showReasoning && reasoningSteps.length > 0 && (
                                            <div className="p-3 space-y-1 max-h-64 overflow-y-auto">
                                                {reasoningSteps.map((step) => (
                                                    <div key={step.id} className={`flex items-start gap-3 py-2 px-2 rounded-lg transition-colors ${step.status === 'active' ? 'bg-primary-50' : ''}`}>
                                                        {/* Status Icon */}
                                                        <div className="flex-shrink-0 mt-0.5">
                                                            {step.status === 'active' ? (
                                                                <ArrowPathIcon className="w-4 h-4 text-primary-600 animate-spin" />
                                                            ) : (
                                                                <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center">
                                                                    <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                                                    </svg>
                                                                </div>
                                                            )}
                                                        </div>

                                                        {/* Content */}
                                                        <div className="flex-1 min-w-0">
                                                            <p className={`text-sm font-medium ${step.status === 'active' ? 'text-primary-700' : 'text-gray-700'}`}>
                                                                {step.title}
                                                            </p>
                                                            {step.detail && (
                                                                <p className="text-xs text-gray-500 mt-0.5 truncate">
                                                                    {step.detail}
                                                                </p>
                                                            )}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {/* Fallback if no steps yet */}
                                        {showReasoning && reasoningSteps.length === 0 && (
                                            <div className="p-4 flex items-center gap-3">
                                                <ArrowPathIcon className="w-5 h-5 text-primary-600 animate-spin" />
                                                <span className="text-gray-500 text-sm">Initializing...</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="border-t border-gray-200 bg-white p-4">
                    <div className="max-w-3xl mx-auto">
                        {/* Options Bar */}
                        <div className="flex items-center gap-3 mb-3 flex-wrap justify-center">
                            <div className="flex bg-white rounded-lg border border-gray-200 p-1">
                                <select
                                    value={selectedClassroom}
                                    onChange={(e) => setSelectedClassroom(e.target.value)}
                                    className="text-sm bg-transparent border-none focus:ring-0 text-gray-700 py-0 pl-2 pr-8 cursor-pointer"
                                >
                                    <option value="">All Classrooms</option>
                                    {classrooms.map(c => (
                                        <option key={c.id} value={c.id}>{c.name}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Response Mode Toggle */}
                            <div className="flex rounded-lg border border-gray-200 p-0.5 bg-white">
                                <button
                                    onClick={() => setResponseMode('short')}
                                    className={`px-3 py-1 text-sm rounded-md transition ${responseMode === 'short' ? 'bg-primary-600 text-white' : 'text-gray-500 hover:bg-gray-50'
                                        }`}
                                >
                                    Short
                                </button>
                                <button
                                    onClick={() => setResponseMode('detailed')}
                                    className={`px-3 py-1 text-sm rounded-md transition ${responseMode === 'detailed' ? 'bg-primary-600 text-white' : 'text-gray-500 hover:bg-gray-50'
                                        }`}
                                >
                                    Detailed
                                </button>
                            </div>

                            <label className="flex items-center gap-2 cursor-pointer px-3 py-1.5 rounded-lg border border-gray-200 bg-white hover:bg-gray-50 transition-colors">
                                <input
                                    type="checkbox"
                                    checked={findResources}
                                    onChange={(e) => setFindResources(e.target.checked)}
                                    className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                />
                                <span className="text-sm text-gray-600">Find resources</span>
                            </label>

                            <label className="flex items-center gap-2 cursor-pointer px-3 py-1.5 rounded-lg border border-gray-200 bg-white hover:bg-gray-50 transition-colors">
                                <input
                                    type="checkbox"
                                    checked={useNotes}
                                    onChange={(e) => setUseNotes(e.target.checked)}
                                    className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                />
                                <span className="text-sm text-gray-600">Use notes</span>
                            </label>
                        </div>

                        {/* Image Preview */}
                        {chatImage && (
                            <div className="mb-3 flex justify-center">
                                <div className="relative inline-block">
                                    <img src={chatImage} alt="Upload preview" className="h-20 rounded-lg border border-gray-200" />
                                    <button
                                        onClick={() => setChatImage(null)}
                                        className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs flex items-center justify-center hover:bg-red-600"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Input Box */}
                        <div className="flex items-end gap-3">
                            <div className="flex-1 relative">
                                <textarea
                                    ref={inputRef}
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault()
                                            sendMessage()
                                        }
                                    }}
                                    placeholder="Ask anything about your studies..."
                                    rows={1}
                                    className="w-full pl-12 pr-4 py-4 bg-gray-50 rounded-2xl border border-gray-200 resize-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-900"
                                    style={{ minHeight: '56px', maxHeight: '150px' }}
                                />
                                {/* Image Upload Button - Inside Input */}
                                <label className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-primary-600 cursor-pointer transition-colors">
                                    <PhotoIcon className="w-5 h-5" />
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={handleImageUpload}
                                        className="hidden"
                                    />
                                </label>
                            </div>
                            <button
                                onClick={() => sendMessage()}
                                disabled={(!input.trim() && !chatImage) || loading}
                                className="p-4 bg-primary-600 text-white rounded-2xl hover:bg-primary-700 disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed transition-colors shadow-md"
                            >
                                <PaperAirplaneIcon className="w-5 h-5" />
                            </button>
                        </div>

                        <p className="text-xs text-gray-400 text-center mt-3">
                            AI Tutor can make mistakes. Verify important information.
                        </p>
                    </div>
                </div>
            </div>

            {/* Sources Sidebar - Right Side (Resizable) */}
            <div
                className={`${showSources && sources.length > 0 ? '' : 'w-0'} transition-all duration-300 overflow-hidden flex-shrink-0 bg-white border-l border-gray-200 relative`}
                style={{ width: showSources && sources.length > 0 ? sidebarWidth : 0 }}
            >
                {/* Resize Handle */}
                <div
                    className="absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-primary-400 transition-colors z-10"
                    onMouseDown={handleMouseDown}
                    style={{ backgroundColor: isResizing ? 'rgb(99, 102, 241)' : 'transparent' }}
                />

                <div className="h-full flex flex-col" style={{ width: sidebarWidth }}>
                    {/* Dynamic Header - Changes based on viewer mode */}
                    {viewerMode === 'list' ? (
                        // List Mode Header with Category Filters
                        <div className="border-b border-gray-200">
                            {/* Header Row */}
                            <div className="p-3 flex items-center justify-between bg-gray-50">
                                <div className="flex items-center gap-2">
                                    <BookOpenIcon className="w-5 h-5 text-primary-600" />
                                    <h2 className="font-semibold text-gray-900">Resources</h2>
                                </div>
                                <button
                                    onClick={() => { setShowSources(false); setActiveSource(null); setViewerMode('list'); }}
                                    className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                >
                                    <XMarkIcon className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Category Filter Icons */}
                            <div className="px-3 py-2 flex gap-2 overflow-x-auto">
                                {(() => {
                                    const counts = getSourceCounts()
                                    const filters: { key: SourceFilter; icon: React.ReactNode; label: string; count: number }[] = [
                                        { key: 'all', icon: <Squares2X2Icon className="w-5 h-5" />, label: 'All', count: counts.all },
                                        { key: 'documents', icon: <DocumentIcon className="w-5 h-5" />, label: 'Docs', count: counts.documents },
                                        { key: 'videos', icon: <VideoCameraIcon className="w-5 h-5" />, label: 'Videos', count: counts.videos },
                                        { key: 'websites', icon: <GlobeAltIcon className="w-5 h-5" />, label: 'Web', count: counts.websites },
                                        { key: 'flowcharts', icon: <MapIcon className="w-5 h-5" />, label: 'Flow', count: counts.flowcharts },
                                        { key: 'images', icon: <PhotoIcon className="w-5 h-5" />, label: 'Images', count: counts.images },
                                    ]
                                    return filters.map(f => (
                                        <button
                                            key={f.key}
                                            onClick={() => setSourceFilter(f.key)}
                                            className={`
                                                flex flex-col items-center justify-center p-2 rounded-lg min-w-[52px] transition-all
                                                ${sourceFilter === f.key
                                                    ? 'bg-primary-100 text-primary-700 ring-2 ring-primary-500'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                                }
                                                ${f.count === 0 ? 'opacity-40' : ''}
                                            `}
                                            title={f.label}
                                        >
                                            {f.icon}
                                            <span className="text-[10px] font-bold mt-0.5">{f.count}</span>
                                        </button>
                                    ))
                                })()}
                            </div>
                        </div>
                    ) : (
                        // Viewer Mode Header
                        <div className="border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
                            <div className="p-3 flex items-center justify-between">
                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                    <button
                                        onClick={() => { setViewerMode('list'); setViewerError(null); }}
                                        className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0"
                                        title="Back to list"
                                    >
                                        <ArrowLeftIcon className="w-5 h-5" />
                                    </button>
                                    <div className="min-w-0 flex-1">
                                        <p className="text-sm font-medium text-gray-900 truncate">
                                            {activeSource?.title}
                                        </p>
                                        <p className="text-xs text-gray-500">{activeSource?.source}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-1 flex-shrink-0">
                                    {activeSource?.url && (
                                        <a
                                            href={activeSource.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                                            title="Open in new tab"
                                        >
                                            <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                                        </a>
                                    )}
                                    {(activeSource?.type === 'video' || activeSource?.type === 'image') && (
                                        <button
                                            onClick={() => viewerRef.current?.requestFullscreen?.()}
                                            className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                                            title="Fullscreen"
                                        >
                                            <ArrowsPointingOutIcon className="w-5 h-5" />
                                        </button>
                                    )}
                                    <button
                                        onClick={() => { setShowSources(false); setActiveSource(null); setViewerMode('list'); }}
                                        className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                    >
                                        <XMarkIcon className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Viewer Content - Full screen when in viewer mode */}
                    {viewerMode === 'viewer' && activeSource && (
                        <div className="flex-1 flex flex-col overflow-hidden bg-gray-900">
                            {/* Loading State */}
                            {viewerLoading && (
                                <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
                                    <div className="flex flex-col items-center">
                                        <ArrowPathIcon className="w-8 h-8 text-white animate-spin" />
                                        <p className="text-white text-sm mt-2">Loading...</p>
                                    </div>
                                </div>
                            )}

                            {/* Error State */}
                            {viewerError && (
                                <div className="flex-1 flex flex-col items-center justify-center p-6 text-center bg-white">
                                    <ExclamationTriangleIcon className="w-12 h-12 text-amber-500 mb-3" />
                                    <p className="text-gray-900 font-medium mb-2">Unable to load preview</p>
                                    <p className="text-gray-500 text-sm mb-4">{viewerError}</p>
                                    <a
                                        href={activeSource.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                                    >
                                        <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                                        Open in new tab
                                    </a>
                                </div>
                            )}

                            {/* YouTube Video Player */}
                            {activeSource.type === 'video' && activeSource.embedUrl && !viewerError && (
                                <div className="flex-1 flex items-center justify-center bg-black" ref={viewerRef as any}>
                                    <iframe
                                        src={activeSource.embedUrl + '?autoplay=0&rel=0'}
                                        className="w-full h-full"
                                        allowFullScreen
                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
                                        onLoad={() => setViewerLoading(false)}
                                        onError={() => {
                                            setViewerLoading(false)
                                            setViewerError('Video failed to load')
                                        }}
                                    />
                                </div>
                            )}

                            {/* Image Viewer with Zoom Controls */}
                            {activeSource.type === 'image' && !viewerError && (
                                <div
                                    className="flex-1 flex flex-col bg-gray-900 overflow-hidden relative"
                                    ref={viewerRef as any}
                                >
                                    {/* Zoomable/Draggable Image Container */}
                                    <div
                                        className={`flex-1 flex items-center justify-center overflow-hidden p-4 ${imageZoom > 100 ? 'cursor-grab' : ''} ${isDragging ? 'cursor-grabbing' : ''}`}
                                        onWheel={(e) => {
                                            e.preventDefault()
                                            const delta = e.deltaY > 0 ? -25 : 25
                                            setImageZoom(z => Math.max(25, Math.min(400, z + delta)))
                                        }}
                                        onMouseDown={(e) => {
                                            if (imageZoom > 100) {
                                                setIsDragging(true)
                                                setDragStart({ x: e.clientX - imagePan.x, y: e.clientY - imagePan.y })
                                            }
                                        }}
                                        onMouseMove={(e) => {
                                            if (isDragging) {
                                                setImagePan({
                                                    x: e.clientX - dragStart.x,
                                                    y: e.clientY - dragStart.y
                                                })
                                            }
                                        }}
                                        onMouseUp={() => setIsDragging(false)}
                                        onMouseLeave={() => setIsDragging(false)}
                                    >
                                        <img
                                            src={activeSource.url || activeSource.thumbnailUrl}
                                            alt={activeSource.title}
                                            className={`object-contain rounded-lg shadow-2xl select-none ${!isDragging ? 'transition-transform duration-200' : ''}`}
                                            draggable={false}
                                            style={{
                                                transform: `scale(${imageZoom / 100}) translate(${imagePan.x / (imageZoom / 100)}px, ${imagePan.y / (imageZoom / 100)}px)`,
                                                maxWidth: imageZoom > 100 ? 'none' : '100%',
                                                maxHeight: imageZoom > 100 ? 'none' : '100%',
                                                willChange: 'transform',
                                                backfaceVisibility: 'hidden'
                                            }}
                                            onLoad={() => setViewerLoading(false)}
                                            onError={() => {
                                                setViewerLoading(false)
                                                setViewerError('Image failed to load')
                                            }}
                                        />
                                    </div>

                                    {/* Zoom Controls - Bottom Bar */}
                                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-full px-4 py-2 shadow-lg">
                                        <button
                                            onClick={() => setImageZoom(z => Math.max(25, z - 25))}
                                            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/20 text-white transition-colors"
                                            title="Zoom out"
                                        >
                                            <span className="text-xl font-bold">−</span>
                                        </button>
                                        <span className="text-white text-sm font-medium w-14 text-center">{imageZoom}%</span>
                                        <button
                                            onClick={() => setImageZoom(z => Math.min(400, z + 25))}
                                            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/20 text-white transition-colors"
                                            title="Zoom in"
                                        >
                                            <span className="text-xl font-bold">+</span>
                                        </button>
                                        <div className="w-px h-6 bg-white/30 mx-1" />
                                        <button
                                            onClick={() => { setImageZoom(100); setImagePan({ x: 0, y: 0 }); }}
                                            className="px-3 py-1 text-white text-sm hover:bg-white/20 rounded-full transition-colors"
                                            title="Reset zoom"
                                        >
                                            Reset
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Webpage/Article Viewer - Display cached content instead of iframe */}
                            {(activeSource.type === 'article' || activeSource.type === 'webpage') && !viewerError && (
                                <div className="flex-1 flex flex-col bg-white overflow-hidden" ref={viewerRef as any}>
                                    {/* If we have cached content, show it directly */}
                                    {activeSource.cachedContent ? (
                                        <div className="flex-1 overflow-y-auto p-6">
                                            {/* Article header */}
                                            <h1 className="text-2xl font-bold text-gray-900 mb-4 leading-tight">
                                                {activeSource.title}
                                            </h1>

                                            {/* Source badge */}
                                            <div className="flex items-center gap-2 mb-6">
                                                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                                                    {activeSource.source}
                                                </span>
                                                <a
                                                    href={activeSource.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-xs text-blue-600 hover:underline flex items-center gap-1"
                                                >
                                                    View original
                                                    <ArrowTopRightOnSquareIcon className="w-3 h-3" />
                                                </a>
                                            </div>

                                            {/* Images from article */}
                                            {activeSource.cachedImages && activeSource.cachedImages.length > 0 && (
                                                <div className="mb-6 flex gap-3 overflow-x-auto pb-2">
                                                    {activeSource.cachedImages.slice(0, 3).map((img, idx) => (
                                                        <img
                                                            key={idx}
                                                            src={img}
                                                            alt=""
                                                            className="h-32 w-auto rounded-lg shadow-sm flex-shrink-0 object-cover bg-gray-100"
                                                            onError={(e) => {
                                                                (e.target as HTMLImageElement).style.display = 'none'
                                                            }}
                                                        />
                                                    ))}
                                                </div>
                                            )}

                                            {/* Article content */}
                                            <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed whitespace-pre-wrap">
                                                {activeSource.cachedContent}
                                            </div>
                                        </div>
                                    ) : activeSource.url?.includes('wikipedia.org') ? (
                                        /* Wikipedia: Allow iframe embedding (they support it) */
                                        <iframe
                                            src={activeSource.url}
                                            className="w-full flex-1 border-0"
                                            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
                                            onLoad={() => setViewerLoading(false)}
                                            onError={() => {
                                                setViewerLoading(false)
                                                setViewerError('Wikipedia page could not be loaded.')
                                            }}
                                        />
                                    ) : activeSource.url ? (
                                        /* OTHER SITES: Show Open in New Tab button instead of iframe */
                                        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-gradient-to-b from-gray-50 to-white">
                                            <div className="w-16 h-16 rounded-full bg-primary-100 flex items-center justify-center mb-6">
                                                <GlobeAltIcon className="w-8 h-8 text-primary-600" />
                                            </div>
                                            <h2 className="text-xl font-semibold text-gray-900 mb-2">{activeSource.title}</h2>
                                            <p className="text-gray-500 mb-2 text-sm">{activeSource.source}</p>
                                            <p className="text-gray-400 text-xs mb-6 max-w-md">
                                                This website cannot be embedded due to security restrictions.
                                                Open it in a new tab to view the full content.
                                            </p>
                                            {activeSource.snippet && (
                                                <div className="bg-gray-50 rounded-lg p-4 mb-6 max-w-lg text-left border border-gray-100">
                                                    <p className="text-gray-600 text-sm italic">"{activeSource.snippet}"</p>
                                                </div>
                                            )}
                                            <a
                                                href={activeSource.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 transition-colors shadow-lg hover:shadow-xl font-medium"
                                            >
                                                <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                                                Open in New Tab
                                            </a>
                                        </div>
                                    ) : (
                                        <div className="flex-1 flex items-center justify-center text-gray-500">
                                            No content available
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* PDF/Document Viewer */}
                            {(activeSource.type === 'pdf' || activeSource.type === 'pptx' || activeSource.type === 'note') && !viewerError && (
                                <div className="flex-1 flex flex-col items-center justify-center p-6 text-center bg-white">
                                    <DocumentIcon className="w-16 h-16 text-gray-300 mb-4" />
                                    <p className="text-gray-900 font-medium mb-2">{activeSource.title}</p>
                                    {activeSource.fileSize && (
                                        <p className="text-gray-500 text-sm mb-4">{activeSource.fileSize}</p>
                                    )}
                                    <p className="text-gray-500 text-sm mb-4">Document preview is available in your classroom</p>
                                    <button className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
                                        <DocumentIcon className="w-4 h-4" />
                                        Open Document
                                    </button>
                                </div>
                            )}

                            {/* Flowchart Viewer */}
                            {activeSource.type === 'flowchart' && activeSource.mermaidCode && !viewerError && (
                                <MindmapViewer
                                    code={activeSource.mermaidCode}
                                    title={activeSource.title}
                                />
                            )}
                        </div>
                    )}

                    {/* Categorized Sources List - Only show in list mode */}
                    {viewerMode === 'list' && (
                        <div className="flex-1 overflow-y-auto">
                            {/* Section 1: Documents */}
                            {(() => {
                                const filteredSources = getFilteredSources()
                                const documents = filteredSources.filter(s => ['pdf', 'pptx', 'note', 'docx'].includes(s.type))
                                if (documents.length === 0) return null
                                return (
                                    <div className="border-b border-gray-100">
                                        <div className="px-4 py-3 bg-gray-50 flex items-center gap-2">
                                            <DocumentIcon className="w-4 h-4 text-red-600" />
                                            <span className="text-sm font-semibold text-gray-900">Documents</span>
                                            <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full ml-auto">
                                                {documents.length}
                                            </span>
                                        </div>
                                        <div className="p-2 space-y-2">
                                            {documents.map((source) => (
                                                <div
                                                    key={source.id}
                                                    onClick={() => openContentViewer(source)}
                                                    className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all group relative ${activeSource?.id === source.id
                                                        ? 'bg-primary-50 border border-primary-200'
                                                        : 'hover:bg-gray-100 hover:shadow-sm'
                                                        }`}
                                                >
                                                    <div className={`p-2 rounded-lg flex-shrink-0 ${source.type === 'pdf' ? 'bg-red-50 text-red-600' :
                                                        source.type === 'pptx' ? 'bg-orange-50 text-orange-600' :
                                                            'bg-blue-50 text-blue-600'
                                                        }`}>
                                                        <DocumentIcon className="w-5 h-5" />
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <p className="text-sm font-medium text-gray-900 truncate">{source.title}</p>
                                                        <div className="flex items-center gap-2 mt-0.5">
                                                            <span className="text-xs text-gray-500">{source.source || 'Classroom'}</span>
                                                            {source.fileSize && <span className="text-xs text-gray-400">{source.fileSize}</span>}
                                                        </div>
                                                    </div>
                                                    {/* Hover Open button - appears between name and score */}
                                                    <div className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                                                        <span className={`text-xs px-3 py-1.5 rounded-full font-medium shadow-sm ${source.type === 'pdf' ? 'bg-red-600 text-white' :
                                                            source.type === 'pptx' ? 'bg-orange-600 text-white' :
                                                                'bg-blue-600 text-white'
                                                            }`}>
                                                            Open
                                                        </span>
                                                    </div>
                                                    {/* Relevance/Trust badge - shows trust for web resources */}
                                                    <span
                                                        className={`text-xs px-2 py-0.5 rounded-full flex-shrink-0 ${source.trustScore
                                                            ? source.trustScore >= 0.9
                                                                ? 'bg-green-100 text-green-700'
                                                                : source.trustScore >= 0.8
                                                                    ? 'bg-blue-100 text-blue-700'
                                                                    : source.trustScore >= 0.7
                                                                        ? 'bg-yellow-100 text-yellow-700'
                                                                        : 'bg-gray-100 text-gray-600'
                                                            : source.relevance >= 90
                                                                ? 'bg-green-100 text-green-700'
                                                                : 'bg-gray-100 text-gray-600'
                                                            }`}
                                                        title={source.trustScore ? `Trust: ${Math.round(source.trustScore * 100)}% • ${source.sourceType || 'web'}` : `Relevance: ${source.relevance}%`}
                                                    >
                                                        {source.trustScore
                                                            ? `${Math.round(source.trustScore * 100)}%`
                                                            : `${source.relevance}%`
                                                        }
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })()}

                            {/* Section 2: Videos */}
                            {(() => {
                                const filteredSources = getFilteredSources()
                                const videos = filteredSources.filter(s => s.type === 'video').slice(0, 5)
                                if (videos.length === 0) return null
                                return (
                                    <div className="border-b border-gray-100">
                                        <div className="px-4 py-3 bg-gray-50 flex items-center gap-2">
                                            <VideoCameraIcon className="w-4 h-4 text-purple-600" />
                                            <span className="text-sm font-semibold text-gray-900">Videos</span>
                                            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full ml-auto">
                                                Top {videos.length}
                                            </span>
                                        </div>
                                        <div className="p-2 space-y-3">
                                            {videos.map((source) => (
                                                <div
                                                    key={source.id}
                                                    onClick={() => openContentViewer(source)}
                                                    className={`rounded-xl overflow-hidden cursor-pointer transition-all group ${activeSource?.id === source.id
                                                        ? 'ring-2 ring-primary-400 shadow-md'
                                                        : 'hover:shadow-lg hover:scale-[1.02]'
                                                        }`}
                                                >
                                                    {source.thumbnailUrl ? (
                                                        <div className="relative">
                                                            <img
                                                                src={source.thumbnailUrl}
                                                                alt={source.title}
                                                                className="w-full h-40 object-cover"
                                                            />
                                                            {/* Default play button */}
                                                            <div className="absolute inset-0 bg-black/30 flex items-center justify-center group-hover:bg-black/50 transition-colors">
                                                                <div className="w-14 h-14 bg-white/95 rounded-full flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                                                                    <svg className="w-6 h-6 text-red-600 ml-1" fill="currentColor" viewBox="0 0 24 24">
                                                                        <path d="M8 5v14l11-7z" />
                                                                    </svg>
                                                                </div>
                                                            </div>
                                                            {/* Hover overlay - just play icon, no text */}
                                                            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                                            {source.duration && (
                                                                <span className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-2 py-1 rounded font-medium">
                                                                    {source.duration}
                                                                </span>
                                                            )}
                                                        </div>
                                                    ) : (
                                                        <div className="h-40 bg-purple-100 flex items-center justify-center">
                                                            <VideoCameraIcon className="w-12 h-12 text-purple-400" />
                                                        </div>
                                                    )}
                                                    <div className="p-3 bg-white border border-gray-100 border-t-0">
                                                        <p className="text-sm font-medium text-gray-900 line-clamp-2">{source.title}</p>
                                                        <p className="text-xs text-gray-500 mt-1">{source.source || 'YouTube'}</p>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })()}

                            {/* Section 3: Images & Diagrams */}
                            {(() => {
                                const filteredSources = getFilteredSources()
                                const images = filteredSources.filter(s => s.type === 'image')
                                if (images.length === 0) return null
                                return (
                                    <div className="border-b border-gray-100">
                                        <div className="px-4 py-3 bg-gray-50 flex items-center gap-2">
                                            <PhotoIcon className="w-4 h-4 text-teal-600" />
                                            <span className="text-sm font-semibold text-gray-900">Images & Diagrams</span>
                                            <span className="text-xs bg-teal-100 text-teal-700 px-2 py-0.5 rounded-full ml-auto">
                                                {images.length}
                                            </span>
                                        </div>
                                        <div className="p-3 space-y-3">
                                            {images.map((source) => (
                                                <div
                                                    key={source.id}
                                                    onClick={() => openContentViewer(source, { resetZoom: true })}
                                                    className={`rounded-xl overflow-hidden cursor-pointer transition-all group ${activeSource?.id === source.id
                                                        ? 'ring-2 ring-primary-400 shadow-lg'
                                                        : 'hover:shadow-lg hover:scale-[1.01]'
                                                        }`}
                                                >
                                                    {/* Large Image */}
                                                    <div className="relative w-full h-32 bg-gray-100">
                                                        {source.thumbnailUrl ? (
                                                            <img
                                                                src={source.thumbnailUrl}
                                                                alt={source.title}
                                                                className="w-full h-full object-cover"
                                                            />
                                                        ) : (
                                                            <div className="w-full h-full bg-teal-50 flex items-center justify-center">
                                                                <PhotoIcon className="w-10 h-10 text-teal-400" />
                                                            </div>
                                                        )}
                                                        {/* Hover overlay */}
                                                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                            <div className="w-12 h-12 rounded-full bg-white/90 flex items-center justify-center shadow-lg">
                                                                <PhotoIcon className="w-6 h-6 text-teal-600" />
                                                            </div>
                                                        </div>
                                                    </div>
                                                    {/* Description - Separate from image */}
                                                    <div className="p-3 bg-white border border-gray-100 border-t-0">
                                                        <p className="text-sm font-medium text-gray-900 line-clamp-2">{source.title}</p>
                                                        <p className="text-xs text-gray-500 mt-1">{source.source || 'Image'}</p>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })()}

                            {/* Section: Flowcharts */}
                            {(() => {
                                const filteredSources = getFilteredSources()
                                const flowcharts = filteredSources.filter(s => s.type === 'flowchart')
                                if (flowcharts.length === 0) return null
                                return (
                                    <div className="border-b border-gray-100">
                                        <div className="px-4 py-3 bg-gray-50 flex items-center gap-2">
                                            <MapIcon className="w-4 h-4 text-purple-600" />
                                            <span className="text-sm font-semibold text-gray-900">Flowcharts</span>
                                            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full ml-auto">
                                                {flowcharts.length}
                                            </span>
                                        </div>
                                        <div className="p-3 space-y-2">
                                            {flowcharts.map((source) => (
                                                <div
                                                    key={source.id}
                                                    onClick={() => openContentViewer(source)}
                                                    className={`p-4 rounded-xl cursor-pointer transition-all group border ${activeSource?.id === source.id
                                                        ? 'ring-2 ring-purple-400 shadow-lg bg-purple-50 border-purple-200'
                                                        : 'bg-gradient-to-br from-purple-50 to-indigo-50 border-purple-100 hover:shadow-lg hover:scale-[1.01]'
                                                        }`}
                                                >
                                                    <div className="flex items-center gap-3 mb-2">
                                                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-md">
                                                            <MapIcon className="w-6 h-6 text-white" />
                                                        </div>
                                                        <div className="flex-1">
                                                            <p className="font-semibold text-gray-900">{source.title}</p>
                                                            <p className="text-xs text-purple-600">{source.source || 'Visual Explanation'}</p>
                                                        </div>
                                                        <span className="bg-purple-600 text-white text-xs px-3 py-1.5 rounded-full font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                                                            View
                                                        </span>
                                                    </div>
                                                    <p className="text-sm text-gray-600">{source.snippet || 'Interactive concept visualization'}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })()}

                            {/* Section 4: Articles & References */}
                            {(() => {
                                const filteredSources = getFilteredSources()
                                const articles = filteredSources.filter(s => ['article', 'webpage'].includes(s.type))
                                if (articles.length === 0) return null
                                return (
                                    <div className="border-b border-gray-100">
                                        <div className="px-4 py-3 bg-gray-50 flex items-center gap-2">
                                            <GlobeAltIcon className="w-4 h-4 text-green-600" />
                                            <span className="text-sm font-semibold text-gray-900">Articles & References</span>
                                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full ml-auto">
                                                {articles.length}
                                            </span>
                                        </div>
                                        <div className="p-2 space-y-2">
                                            {articles.map((source) => (
                                                <div
                                                    key={source.id}
                                                    onClick={() => openContentViewer(source)}
                                                    className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-all group ${activeSource?.id === source.id
                                                        ? 'bg-primary-50 border border-primary-200'
                                                        : 'hover:bg-gray-100 hover:shadow-sm'
                                                        }`}
                                                >
                                                    <div className="p-2 rounded-lg flex-shrink-0 bg-green-50 text-green-600 group-hover:bg-green-100 transition-colors">
                                                        <GlobeAltIcon className="w-5 h-5" />
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <p className="text-sm font-medium text-gray-900 line-clamp-1">{source.title}</p>
                                                        <p className="text-xs text-green-600 mt-0.5">{source.source || 'Web'}</p>
                                                        {source.snippet && (
                                                            <p className="text-xs text-gray-500 mt-1 line-clamp-2">{source.snippet}</p>
                                                        )}
                                                    </div>
                                                    {/* Hover overlay button */}
                                                    <div className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                                                        <span className="bg-green-600 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow-sm">
                                                            Read
                                                        </span>
                                                    </div>
                                                    {/* Link icon - hide on hover */}
                                                    <LinkIcon className="w-4 h-4 text-gray-400 flex-shrink-0 mt-1 group-hover:hidden" />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })()}

                            {/* Empty state */}
                            {sources.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-full text-center px-4">
                                    <BookOpenIcon className="w-12 h-12 text-gray-300 mb-3" />
                                    <p className="text-sm text-gray-500">No resources found yet</p>
                                    <p className="text-xs text-gray-400 mt-1">Resources will appear here when you ask a question</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Footer - only show in list mode */}
                    {viewerMode === 'list' && (
                        <div className="p-3 border-t border-gray-200 bg-gray-50">
                            <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
                                <span className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                    Classroom
                                </span>
                                <span className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                    Web
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Sources Toggle Button - Shows when sidebar is hidden */}
            {!showSources && sources.length > 0 && (
                <button
                    onClick={() => setShowSources(true)}
                    className="fixed right-4 top-1/2 -translate-y-1/2 p-3 bg-white border border-gray-200 rounded-xl shadow-lg hover:shadow-xl transition-shadow z-20"
                    title="Show sources"
                >
                    <BookOpenIcon className="w-5 h-5 text-primary-600" />
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-primary-600 text-white text-xs rounded-full flex items-center justify-center">
                        {sources.length}
                    </span>
                </button>
            )}
        </div>
    )
}
