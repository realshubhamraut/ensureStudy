'use client'
import { getApiBaseUrl, fixFileUrl } from '@/utils/api'


import { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import {
    ArrowLeftIcon,
    DocumentTextIcon,
    VideoCameraIcon,
    MusicalNoteIcon,
    PhotoIcon,
    ArrowDownTrayIcon,
    UserCircleIcon,
    FolderIcon,
    CalendarIcon,
    MegaphoneIcon,
    PlayIcon,
    SparklesIcon,
    BookOpenIcon,
    XMarkIcon,
    AcademicCapIcon,
    LinkIcon,
    CheckCircleIcon,
    CloudArrowUpIcon,
    TrashIcon,
    EyeIcon,
    GlobeAltIcon,
    FilmIcon
} from '@heroicons/react/24/outline'
import PDFViewer from '@/components/PDFViewer'
import ImageViewer from '@/components/ImageViewer'
import { RecordingsList } from '@/components/meeting/RecordingsList'
import { MeetingQA } from '@/components/meeting/MeetingQA'

interface Material {
    id: string
    name: string
    type: string
    size: number
    url: string
    uploaded_at: string
}

interface Announcement {
    id: string
    message: string
    created_at: string
}

interface Meeting {
    id: string
    title: string
    status: 'scheduled' | 'live' | 'ended'
    scheduled_at?: string
    started_at?: string
    ended_at?: string
    duration_minutes?: number
    meeting_link: string
    recording_url?: string
    transcript?: string
}

interface Assignment {
    id: string
    title: string
    description: string
    due_date: string | null
    points: number | null
    attachments?: { id: string; type: 'file' | 'link'; url: string; filename?: string }[]
    my_submission?: {
        id: string
        status: 'submitted' | 'grading' | 'graded' | 'failed_grading'
        grade: number | null
        feedback: string | null
        files?: { id: string; url: string; filename: string; type: 'file' | 'link' }[]
        // AI Grading fields
        ai_graded?: boolean
        ai_confidence?: number
        graded_at?: string
        detailed_feedback?: {
            total_grade: number
            max_points: number
            percentage: number
            overall_feedback: string
            question_grades: {
                question_number: number
                question_text: string
                student_answer: string
                points_earned: number
                max_points: number
                percentage: number
                feedback: string
                confidence: number
            }[]
        }
    }
}

interface ExamResult {
    id: string
    exam_session: {
        id: string
        name: string
        exam_type: string
        subject: string
        class_name: string
        date: string
        results_declared_at: string
    }
    total_score: number
    max_score: number
    percentage: number
    grade: string
    question_evaluations: {
        question_number: string
        score: number
        max_marks: number
        feedback: string
    }[]
}

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    subject: string
    teacher?: { id: string; name: string }
    syllabus_url?: string
    has_syllabus?: boolean
}

type TabType = 'stream' | 'materials' | 'sessions' | 'assignments' | 'results'
type DateFilter = 'all' | 'today' | 'yesterday' | 'week' | 'custom'

export default function StudentClassroomDetailPage() {
    const params = useParams()
    const router = useRouter()
    const classroomId = params.id as string

    const [activeTab, setActiveTab] = useState<TabType>('stream')
    const [classroom, setClassroom] = useState<Classroom | null>(null)
    const [materials, setMaterials] = useState<Material[]>([])
    const [announcements, setAnnouncements] = useState<Announcement[]>([])
    const [meetings, setMeetings] = useState<Meeting[]>([])
    const [loading, setLoading] = useState(true)
    const [dateFilter, setDateFilter] = useState<DateFilter>('all')
    const [customDate, setCustomDate] = useState('')
    const [showDatePicker, setShowDatePicker] = useState(false)
    const [meetingView, setMeetingView] = useState<'upcoming' | 'past'>('past')
    const [mounted, setMounted] = useState(false)

    // For rendering modals with portal (to fix scroll issues)
    useEffect(() => {
        setMounted(true)
    }, [])

    // Transcript and AI Chat state
    const [showTranscriptModal, setShowTranscriptModal] = useState(false)
    const [showAIChatModal, setShowAIChatModal] = useState(false)
    const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null)
    const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'ai', text: string, image?: string }[]>([])
    const [chatInput, setChatInput] = useState('')
    const [chatLoading, setChatLoading] = useState(false)

    // AI Chat options
    const [responseLength, setResponseLength] = useState<'short' | 'detailed'>('short')
    const [findResources, setFindResources] = useState(false)
    const [useClassroomNotes, setUseClassroomNotes] = useState(true)
    const [chatImage, setChatImage] = useState<string | null>(null)

    // Syllabus modal state
    const [showSyllabusModal, setShowSyllabusModal] = useState(false)

    // Assignment states
    const [assignments, setAssignments] = useState<Assignment[]>([])
    const [showSubmissionModal, setShowSubmissionModal] = useState(false)
    const [selectedAssignment, setSelectedAssignment] = useState<Assignment | null>(null)
    const [submissionFiles, setSubmissionFiles] = useState<{ url: string; filename: string; type: 'file' | 'link' }[]>([])
    const [submissionLinkInput, setSubmissionLinkInput] = useState('')
    const [submitting, setSubmitting] = useState(false)
    const [showDetailedFeedbackModal, setShowDetailedFeedbackModal] = useState(false)
    const submissionFileRef = useRef<HTMLInputElement>(null)

    // Exam Results states
    const [examResults, setExamResults] = useState<ExamResult[]>([])
    const [loadingResults, setLoadingResults] = useState(false)
    const [expandedResult, setExpandedResult] = useState<string | null>(null)

    // INTEGRATION NOTE: Document upload state for Materials tab
    const [showDocumentUploadModal, setShowDocumentUploadModal] = useState(false)
    const [documentUploadFile, setDocumentUploadFile] = useState<File | null>(null)
    const [documentUploadTitle, setDocumentUploadTitle] = useState('')
    const [documentUploading, setDocumentUploading] = useState(false)
    const documentFileRef = useRef<HTMLInputElement>(null)

    // Document Viewer Modal state
    const [showDocumentViewer, setShowDocumentViewer] = useState(false)
    const [viewingDocument, setViewingDocument] = useState<Material | null>(null)

    // Fetch exam results for student
    const fetchResults = async () => {
        setLoadingResults(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/my-results`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setExamResults(data.results || [])
            }
        } catch (error) {
            console.error('Failed to fetch results:', error)
        }
        setLoadingResults(false)
    }

    useEffect(() => {
        fetchClassroom()
        fetchResults()
    }, [classroomId])

    const fetchClassroom = async () => {
        try {
            // First try syllabus endpoint for enrolled students
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/syllabus`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                const syllabusData = await res.json()
                console.log('Syllabus data:', syllabusData)
                setClassroom({
                    id: classroomId,
                    name: syllabusData.classroom_name || 'Class',
                    grade: '10',
                    section: 'A',
                    subject: 'Subject',
                    teacher: { id: '1', name: syllabusData.teacher_name || 'Teacher' },
                    syllabus_url: syllabusData.syllabus_url,
                    has_syllabus: syllabusData.has_syllabus
                })
            } else {
                console.log('Syllabus fetch failed, using fallback')
                // Fallback to mock data
                setClassroom({
                    id: classroomId,
                    name: 'Class',
                    grade: '10',
                    section: 'A',
                    subject: 'Subject',
                    teacher: { id: '1', name: 'Teacher' },
                    syllabus_url: undefined,
                    has_syllabus: false
                })
            }

            // Fetch materials from API
            const materialsRes = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/materials`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (materialsRes.ok) {
                const materialsData = await materialsRes.json()
                setMaterials(materialsData.materials || [])
            }

            // Fetch meetings from API first (most important for live meetings)
            try {
                const meetingsRes = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/meetings`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (meetingsRes.ok) {
                    const meetingsData = await meetingsRes.json()
                    console.log('Meetings data:', meetingsData)
                    setMeetings(meetingsData.meetings || [])
                }
            } catch (e) {
                console.error('Failed to fetch meetings:', e)
            }

            // Fetch announcements from API (may not exist)
            try {
                const announcementsRes = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/announcements`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (announcementsRes.ok) {
                    const announcementsData = await announcementsRes.json()
                    setAnnouncements(announcementsData.announcements || [])
                }
            } catch (e) {
                console.error('Failed to fetch announcements:', e)
            }
        } catch (error) {
            console.error('Failed to fetch classroom:', error)
        } finally {
            setLoading(false)
        }
    }

    const filterMaterials = () => {
        if (dateFilter === 'all') return materials
        const today = new Date()
        today.setHours(0, 0, 0, 0)

        return materials.filter(m => {
            const uploadDate = new Date(m.uploaded_at)
            uploadDate.setHours(0, 0, 0, 0)

            switch (dateFilter) {
                case 'today':
                    return uploadDate.getTime() === today.getTime()
                case 'yesterday':
                    const yesterday = new Date(today)
                    yesterday.setDate(yesterday.getDate() - 1)
                    return uploadDate.getTime() === yesterday.getTime()
                case 'week':
                    const weekAgo = new Date(today)
                    weekAgo.setDate(weekAgo.getDate() - 7)
                    return uploadDate >= weekAgo
                case 'custom':
                    if (!customDate) return true
                    const custom = new Date(customDate)
                    return uploadDate.getTime() === custom.getTime()
                default:
                    return true
            }
        })
    }

    // Open full transcript modal
    const openTranscript = (meeting: Meeting) => {
        setSelectedMeeting(meeting)
        setShowTranscriptModal(true)
        // Scroll to top so modal is visible
        window.scrollTo({ top: 0, behavior: 'smooth' })
    }

    // Open AI chat modal
    const openAIChat = (meeting: Meeting) => {
        setSelectedMeeting(meeting)
        setChatMessages([
            { role: 'ai', text: `I'm ready to answer questions about "${meeting.title}". What would you like to know about this lecture?` }
        ])
        setShowAIChatModal(true)
    }

    // Send message to AI
    const sendChatMessage = async () => {
        if ((!chatInput.trim() && !chatImage) || !selectedMeeting) return

        const userMessage = chatInput.trim()
        setChatInput('')

        // Add user message (with optional image)
        setChatMessages(prev => [...prev, { role: 'user', text: userMessage || 'Sent an image', image: chatImage || undefined }])
        setChatImage(null)
        setChatLoading(true)

        // Simulate AI response (in production, call AI API with meeting transcript context)
        await new Promise(resolve => setTimeout(resolve, 1500))

        // Generate contextual response based on the question and options
        let aiResponse = ''
        const lowerQ = userMessage.toLowerCase()

        // Add context based on options
        let contextNote = ''
        if (findResources) contextNote += '📚 Found related resources. '
        if (useClassroomNotes) contextNote += '📝 Referenced classroom notes. '

        if (lowerQ.includes('summary') || lowerQ.includes('about')) {
            aiResponse = responseLength === 'detailed'
                ? `${contextNote}This lecture "${selectedMeeting.title}" covered key concepts including the main topic discussion, practical examples, and a Q&A session. The lecture lasted ${selectedMeeting.duration_minutes} minutes.\n\nKey Points:\n• Introduction to the topic\n• Core concepts explained\n• Practical examples demonstrated\n• Student questions addressed\n• Summary and next steps`
                : `${contextNote}This lecture "${selectedMeeting.title}" covered key concepts including the main topic discussion and examples. Duration: ${selectedMeeting.duration_minutes} min.`
        } else if (lowerQ.includes('example') || lowerQ.includes('problem')) {
            aiResponse = responseLength === 'detailed'
                ? `${contextNote}In this lecture, several examples were discussed:\n\n1. Basic concept examples\n2. Numerical problems with step-by-step solutions\n3. Real-world applications\n4. Practice problems for homework\n\nWould you like me to explain any specific example in more detail?`
                : `${contextNote}Several examples were discussed including numerical problems and real-world applications.`
        } else if (chatImage) {
            aiResponse = `${contextNote}I can see the image you shared. Based on the lecture "${selectedMeeting.title}", this appears to be related to the concepts we covered. Would you like me to explain how this connects to the lecture material?`
        } else {
            aiResponse = responseLength === 'detailed'
                ? `${contextNote}Based on the lecture "${selectedMeeting.title}", here's what I found relevant to your question:\n\n• The topic was covered comprehensively with explanations\n• Multiple examples were provided\n• Key formulas and concepts were explained\n\nIs there anything specific you'd like me to clarify?`
                : `${contextNote}Based on the lecture, the topic was covered with explanations and examples. What would you like to know more about?`
        }

        setChatMessages(prev => [...prev, { role: 'ai', text: aiResponse }])
        setChatLoading(false)
    }

    // Handle image upload for chat
    const handleChatImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader()
            reader.onload = (event) => {
                setChatImage(event.target?.result as string)
            }
            reader.readAsDataURL(file)
        }
    }

    const getFileIcon = (type: string) => {
        if (type.includes('video')) return <VideoCameraIcon className="w-6 h-6 text-red-500" />
        if (type.includes('audio')) return <MusicalNoteIcon className="w-6 h-6 text-purple-500" />
        if (type.includes('image')) return <PhotoIcon className="w-6 h-6 text-green-500" />
        if (type.includes('pdf')) return <DocumentTextIcon className="w-6 h-6 text-red-600" />
        return <DocumentTextIcon className="w-6 h-6 text-blue-500" />
    }

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return bytes + ' B'
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
    }

    // ===== Assignment Functions =====
    const fetchAssignments = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/assignments`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setAssignments(data.assignments || [])
            }
        } catch (error) {
            console.error('Failed to fetch assignments:', error)
        }
    }

    useEffect(() => {
        if (classroomId) {
            fetchAssignments()
        }
    }, [classroomId])

    const openSubmissionModal = (assignment: Assignment) => {
        setSelectedAssignment(assignment)
        // Pre-fill with existing submission files if any
        if (assignment.my_submission?.files) {
            setSubmissionFiles(assignment.my_submission.files)
        } else {
            setSubmissionFiles([])
        }
        setShowSubmissionModal(true)
    }

    const handleSubmissionFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files
        if (!files || files.length === 0) return

        for (const file of Array.from(files)) {
            if (file.type === 'application/pdf') {
                const formData = new FormData()
                formData.append('file', file)

                try {
                    const uploadRes = await fetch(`${getApiBaseUrl()}/api/files/upload`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                        },
                        body: formData
                    })

                    let fileUrl = ''
                    if (uploadRes.ok) {
                        const uploadData = await uploadRes.json()
                        fileUrl = uploadData.url
                    } else {
                        fileUrl = URL.createObjectURL(file)
                    }

                    setSubmissionFiles(prev => [...prev, {
                        url: fileUrl,
                        filename: file.name,
                        type: 'file'
                    }])
                } catch (error) {
                    console.error('Upload failed:', error)
                }
            } else {
                alert('Only PDF files are allowed')
            }
        }

        if (submissionFileRef.current) submissionFileRef.current.value = ''
    }

    const addSubmissionLink = () => {
        if (!submissionLinkInput.trim()) return
        setSubmissionFiles(prev => [...prev, {
            url: submissionLinkInput,
            filename: submissionLinkInput,
            type: 'link'
        }])
        setSubmissionLinkInput('')
    }

    const removeSubmissionFile = (index: number) => {
        setSubmissionFiles(prev => prev.filter((_, i) => i !== index))
    }

    const submitAssignment = async () => {
        if (!selectedAssignment || submissionFiles.length === 0) {
            alert('Please add at least one file or link')
            return
        }

        setSubmitting(true)
        try {
            const endpoint = selectedAssignment.my_submission
                ? `${getApiBaseUrl()}/api/submission/${selectedAssignment.my_submission.id}`
                : `${getApiBaseUrl()}/api/assignment/${selectedAssignment.id}/submit`

            const method = selectedAssignment.my_submission ? 'PUT' : 'POST'

            const res = await fetch(endpoint, {
                method,
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ files: submissionFiles })
            })

            if (res.ok) {
                setShowSubmissionModal(false)
                fetchAssignments() // Refresh assignments
                alert(selectedAssignment.my_submission ? 'Submission updated!' : 'Assignment submitted successfully!')
            } else {
                alert('Failed to submit assignment')
            }
        } catch (error) {
            console.error('Submit failed:', error)
            alert('Failed to submit assignment')
        } finally {
            setSubmitting(false)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    if (!classroom) {
        return (
            <div className="text-center py-16">
                <p className="text-gray-500">Classroom not found</p>
                <Link href="/classrooms" className="text-primary-600 hover:underline mt-2 inline-block">
                    Back to Classrooms
                </Link>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-4">
                <Link href="/classrooms" className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg">
                    <ArrowLeftIcon className="w-5 h-5" />
                </Link>
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">{classroom.name}</h1>
                    <p className="text-gray-600">
                        {classroom.grade && `Grade ${classroom.grade}`}
                        {classroom.section && ` • ${classroom.section}`}
                        {classroom.subject && ` • ${classroom.subject}`}
                    </p>
                </div>
            </div>

            {/* Teacher Info with Syllabus Button */}
            {classroom.teacher && (
                <div className="card flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                            <UserCircleIcon className="w-8 h-8 text-white" />
                        </div>
                        <div>
                            <p className="text-sm text-gray-500">Teacher</p>
                            <p className="font-semibold text-gray-900">{classroom.teacher.name}</p>
                        </div>
                    </div>
                    {classroom.has_syllabus && classroom.syllabus_url && (
                        <button
                            onClick={() => setShowSyllabusModal(true)}
                            className="px-4 py-2 border-2 border-cyan-400 text-cyan-600 rounded-xl font-medium hover:bg-cyan-50 transition-colors flex items-center gap-2"
                        >
                            <BookOpenIcon className="w-5 h-5" />
                            View Syllabus
                        </button>
                    )}
                </div>
            )}

            {/* Live Meeting Alert */}
            {meetings.some(m => m.status === 'live') && (
                <div className="card bg-gradient-to-r from-red-500 to-pink-500 text-white">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                            <div>
                                <p className="font-medium">Live Meeting in Progress</p>
                                <p className="text-sm text-red-100">{meetings.find(m => m.status === 'live')?.title}</p>
                            </div>
                        </div>
                        <button
                            onClick={() => {
                                const liveMeeting = meetings.find(m => m.status === 'live')
                                if (liveMeeting) {
                                    router.push(`/meet/${liveMeeting.id}`)
                                }
                            }}
                            className="px-4 py-2 bg-white text-red-600 rounded-lg font-medium flex items-center gap-2"
                        >
                            <PlayIcon className="w-5 h-5" />
                            Join Now
                        </button>
                    </div>
                </div>
            )}

            {/* Tab Navigation */}
            <div className="flex rounded-xl bg-gray-100 p-1 w-fit">
                {[
                    { id: 'stream', label: 'Stream', icon: MegaphoneIcon },
                    { id: 'materials', label: 'Materials', icon: FolderIcon },
                    { id: 'sessions', label: 'Class Sessions', icon: VideoCameraIcon },
                    { id: 'assignments', label: `Assignments (${assignments.length})`, icon: AcademicCapIcon },
                    { id: 'results', label: `Results${examResults.length > 0 ? ` (${examResults.length})` : ''}`, icon: CheckCircleIcon },
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as TabType)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${activeTab === tab.id ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
                {/* Digitize Notes - separate link */}
                <Link
                    href={`/classrooms/${classroomId}/notes`}
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 text-purple-600 hover:bg-purple-50 ml-2"
                >
                    <SparklesIcon className="w-4 h-4" />
                    Digitize Notes
                </Link>
            </div>

            {/* Stream Tab (Announcements) */}
            {activeTab === 'stream' && (
                <div className="space-y-3">
                    {announcements.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <MegaphoneIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No announcements yet</p>
                        </div>
                    ) : (
                        announcements.map(a => (
                            <div key={a.id} className="card">
                                <p className="text-gray-900">{a.message}</p>
                                <p className="text-xs text-gray-500 mt-2">
                                    {(() => {
                                        const utcDate = a.created_at.endsWith('Z') ? a.created_at : a.created_at + 'Z'
                                        const date = new Date(utcDate)
                                        const now = new Date()
                                        const diff = now.getTime() - date.getTime()
                                        const minutes = Math.floor(diff / 60000)
                                        const hours = Math.floor(diff / 3600000)
                                        const days = Math.floor(diff / 86400000)
                                        if (minutes < 1) return 'Just now'
                                        if (minutes < 60) return `${minutes}m ago`
                                        if (hours < 24) return `${hours}h ago`
                                        if (days < 7) return `${days}d ago`
                                        return date.toLocaleDateString()
                                    })()}
                                </p>
                            </div>
                        ))
                    )}
                </div>
            )}

            {/* Materials Tab */}
            {activeTab === 'materials' && (
                <div>
                    {/* Header with Upload Button */}
                    <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                            <FolderIcon className="w-5 h-5 text-gray-400" />
                            Study Materials
                        </h2>
                        <div className="flex items-center gap-2 flex-wrap">
                            {/* Upload button removed - only teachers can upload materials */}
                            {['all', 'today', 'yesterday', 'week'].map((f) => (
                                <button
                                    key={f}
                                    onClick={() => { setDateFilter(f as DateFilter); setShowDatePicker(false) }}
                                    className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-colors ${dateFilter === f ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                        }`}
                                >
                                    {f === 'all' ? 'All' : f === 'today' ? 'Today' : f === 'yesterday' ? 'Yesterday' : 'This Week'}
                                </button>
                            ))}
                            <div className="relative">
                                <button
                                    onClick={() => setShowDatePicker(!showDatePicker)}
                                    className={`px-3 py-1.5 text-sm rounded-lg font-medium flex items-center gap-1 ${dateFilter === 'custom' ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-600'
                                        }`}
                                >
                                    <CalendarIcon className="w-4 h-4" />
                                    Custom
                                </button>
                                {showDatePicker && (
                                    <div className="absolute right-0 top-full mt-1 p-2 bg-white rounded-lg shadow-lg border z-10">
                                        <input
                                            type="date"
                                            value={customDate}
                                            onChange={(e) => { setCustomDate(e.target.value); setDateFilter('custom') }}
                                            className="input-field text-sm"
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {filterMaterials().length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <FolderIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">
                                {dateFilter === 'all' ? 'No materials uploaded yet' : 'No materials for selected date'}
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {filterMaterials().map((material) => (
                                <div key={material.id} className="card-hover flex items-center justify-between">
                                    <div
                                        className="flex items-center gap-4 flex-1 cursor-pointer hover:opacity-80"
                                        onClick={() => {
                                            // Open viewer for PDFs and images, download for others
                                            if (material.type.includes('pdf') || material.type.includes('image')) {
                                                setViewingDocument(material)
                                                setShowDocumentViewer(true)
                                            } else if (material.type.includes('video')) {
                                                window.open(fixFileUrl(material.url), '_blank')
                                            } else {
                                                window.open(fixFileUrl(material.url), '_blank')
                                            }
                                        }}
                                    >
                                        <div className="p-3 bg-gray-100 rounded-lg">
                                            {getFileIcon(material.type)}
                                        </div>
                                        <div>
                                            <p className="font-medium text-gray-900">{material.name}</p>
                                            <p className="text-sm text-gray-500">
                                                {formatSize(material.size)} • {new Date(material.uploaded_at).toLocaleDateString()}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        {/* View Button - for PDFs */}
                                        {material.type.includes('pdf') && (
                                            <button
                                                onClick={() => {
                                                    setViewingDocument(material)
                                                    setShowDocumentViewer(true)
                                                }}
                                                className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg"
                                                title="View document"
                                            >
                                                <EyeIcon className="w-5 h-5" />
                                            </button>
                                        )}
                                        {/* Download Button */}
                                        <button
                                            onClick={() => window.open(fixFileUrl(material.url), '_blank')}
                                            className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg"
                                            title="Download"
                                        >
                                            <ArrowDownTrayIcon className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Class Sessions Tab (Combined Meetings + Recordings) */}
            {activeTab === 'sessions' && (
                <div className="space-y-6">
                    {/* Meetings Section with Filters */}
                    <div>
                        {/* Header with Filters */}
                        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                                <VideoCameraIcon className="w-5 h-5 text-gray-400" />
                                Meetings
                            </h2>

                            {/* Filter Controls */}
                            <div className="flex items-center gap-3">
                                {/* Upcoming / Past Toggle */}
                                <div className="flex bg-gray-100 rounded-lg p-0.5">
                                    <button
                                        onClick={() => setMeetingView('upcoming')}
                                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${meetingView === 'upcoming'
                                            ? 'bg-white text-gray-900 shadow-sm'
                                            : 'text-gray-500 hover:text-gray-700'
                                            }`}
                                    >
                                        Upcoming
                                    </button>
                                    <button
                                        onClick={() => setMeetingView('past')}
                                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${meetingView === 'past'
                                            ? 'bg-white text-gray-900 shadow-sm'
                                            : 'text-gray-500 hover:text-gray-700'
                                            }`}
                                    >
                                        Past Recordings
                                    </button>
                                </div>

                                {/* Date Filter */}
                                <div className="relative">
                                    <select
                                        value={dateFilter}
                                        onChange={(e) => setDateFilter(e.target.value as DateFilter)}
                                        className="appearance-none bg-white border border-gray-200 rounded-lg px-3 py-1.5 pr-8 text-xs text-gray-700 cursor-pointer hover:border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500"
                                    >
                                        <option value="all">All Dates</option>
                                        <option value="today">Today</option>
                                        <option value="yesterday">Yesterday</option>
                                        <option value="week">This Week</option>
                                        <option value="custom">Custom Date</option>
                                    </select>
                                    <CalendarIcon className="w-3.5 h-3.5 text-gray-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                                </div>

                                {/* Custom Date Picker */}
                                {dateFilter === 'custom' && (
                                    <input
                                        type="date"
                                        value={customDate}
                                        onChange={(e) => setCustomDate(e.target.value)}
                                        className="bg-white border border-gray-200 rounded-lg px-3 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                                    />
                                )}
                            </div>
                        </div>

                        {meetings.length === 0 ? (
                            <div className="text-center py-12 bg-gray-50 rounded-xl">
                                <VideoCameraIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                                <p className="text-gray-500">No meetings scheduled</p>
                            </div>
                        ) : (() => {
                            // Filter meetings based on view and date
                            const filteredMeetings = meetings.filter(m => {
                                // Filter by upcoming/past
                                if (meetingView === 'upcoming') {
                                    if (m.status !== 'live' && m.status !== 'scheduled') return false
                                } else {
                                    if (m.status !== 'ended') return false
                                }

                                // Filter by date
                                if (dateFilter !== 'all') {
                                    const meetingDate = new Date(m.scheduled_at || m.ended_at || '')
                                    const today = new Date()
                                    today.setHours(0, 0, 0, 0)

                                    if (dateFilter === 'today') {
                                        const meetingDateOnly = new Date(meetingDate)
                                        meetingDateOnly.setHours(0, 0, 0, 0)
                                        if (meetingDateOnly.getTime() !== today.getTime()) return false
                                    } else if (dateFilter === 'yesterday') {
                                        const yesterday = new Date(today)
                                        yesterday.setDate(yesterday.getDate() - 1)
                                        const meetingDateOnly = new Date(meetingDate)
                                        meetingDateOnly.setHours(0, 0, 0, 0)
                                        if (meetingDateOnly.getTime() !== yesterday.getTime()) return false
                                    } else if (dateFilter === 'week') {
                                        const weekAgo = new Date(today)
                                        weekAgo.setDate(weekAgo.getDate() - 7)
                                        if (meetingDate < weekAgo) return false
                                    } else if (dateFilter === 'custom' && customDate) {
                                        const customDateObj = new Date(customDate)
                                        customDateObj.setHours(0, 0, 0, 0)
                                        const meetingDateOnly = new Date(meetingDate)
                                        meetingDateOnly.setHours(0, 0, 0, 0)
                                        if (meetingDateOnly.getTime() !== customDateObj.getTime()) return false
                                    }
                                }
                                return true
                            })

                            if (filteredMeetings.length === 0) {
                                return (
                                    <div className="text-center py-12 bg-gray-50 rounded-xl">
                                        <VideoCameraIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                                        <p className="text-gray-500">
                                            {meetingView === 'upcoming' ? 'No upcoming meetings' : 'No past recordings found'}
                                        </p>
                                    </div>
                                )
                            }

                            return (
                                <div className="space-y-4">
                                    {filteredMeetings.map(m => (
                                        <div key={m.id} className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-md transition-all">
                                            {/* For Live/Scheduled - show header */}
                                            {m.status !== 'ended' && (
                                                <div className="flex items-center justify-between p-4">
                                                    <div className="flex items-center gap-3 flex-1 min-w-0">
                                                        <div className={`p-2.5 rounded-lg shrink-0 ${m.status === 'live' ? 'bg-red-100' : 'bg-gray-100'}`}>
                                                            <VideoCameraIcon className={`w-5 h-5 ${m.status === 'live' ? 'text-red-600' : 'text-gray-500'}`} />
                                                        </div>
                                                        <div className="flex-1 min-w-0">
                                                            <h3 className="font-medium text-gray-900 truncate">{m.title}</h3>
                                                            <p className="text-sm">
                                                                {m.status === 'live' && <span className="text-red-600 font-medium">● Live Now</span>}
                                                                {m.status === 'scheduled' && <span className="text-gray-500">Starts: {new Date(m.scheduled_at!).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}</span>}
                                                            </p>
                                                        </div>
                                                    </div>
                                                    {m.status === 'live' && (
                                                        <button
                                                            onClick={() => router.push(`/meet/${m.id}`)}
                                                            className="btn-primary text-sm flex items-center gap-2 shrink-0"
                                                        >
                                                            <PlayIcon className="w-4 h-4" />
                                                            Join
                                                        </button>
                                                    )}
                                                </div>
                                            )}

                                            {/* For Ended meetings - optimized single row layout */}
                                            {m.status === 'ended' && m.recording_url && (
                                                <div className="flex h-28">
                                                    {/* Video Thumbnail */}
                                                    <div
                                                        className="relative w-44 shrink-0 bg-gray-900 cursor-pointer group"
                                                        onClick={() => window.open(fixFileUrl(m.recording_url!), '_blank')}
                                                    >
                                                        <video
                                                            src={fixFileUrl(m.recording_url)}
                                                            className="w-full h-full object-cover"
                                                            muted
                                                            preload="metadata"
                                                            onLoadedMetadata={(e) => {
                                                                (e.target as HTMLVideoElement).currentTime = 1
                                                            }}
                                                        />
                                                        <div className="absolute inset-0 bg-black/30 flex items-center justify-center group-hover:bg-black/40 transition-colors">
                                                            <div className="w-12 h-12 bg-white/95 rounded-full flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
                                                                <PlayIcon className="w-6 h-6 text-gray-800 ml-0.5" />
                                                            </div>
                                                        </div>
                                                        <div className="absolute bottom-2 right-2 bg-black/80 text-white text-xs font-medium px-1.5 py-0.5 rounded">
                                                            {m.duration_minutes}:00
                                                        </div>
                                                    </div>

                                                    {/* Title next to video */}
                                                    <div className="w-48 p-3 border-r border-gray-100 flex flex-col justify-center shrink-0">
                                                        <h3 className="font-semibold text-gray-900 text-sm line-clamp-2 mb-2">{m.title}</h3>
                                                        <a
                                                            href={fixFileUrl(m.recording_url)}
                                                            target="_blank"
                                                            className="text-xs text-primary-600 hover:underline flex items-center gap-1"
                                                        >
                                                            <PlayIcon className="w-3 h-3" />
                                                            Watch Recording
                                                        </a>
                                                    </div>

                                                    {/* Summary section */}
                                                    <div className="flex-1 p-3 flex flex-col justify-center min-w-0">
                                                        <div className="flex items-center gap-1 text-xs text-purple-600 mb-1">
                                                            <SparklesIcon className="w-3.5 h-3.5" />
                                                            <span className="font-medium">Summary</span>
                                                        </div>
                                                        <p className="text-xs text-gray-600 line-clamp-3">
                                                            {m.transcript || 'AI summary will appear here after processing...'}
                                                        </p>
                                                    </div>

                                                    {/* Right side: Stacked buttons (Transcript on top, AI on bottom) */}
                                                    <div className="w-40 shrink-0 flex flex-col border-l border-gray-200">
                                                        <button
                                                            onClick={() => openTranscript(m)}
                                                            className="flex-1 bg-gray-50 hover:bg-gray-100 text-gray-600 flex items-center justify-center gap-2 transition-colors border-b border-gray-200 px-3"
                                                        >
                                                            <DocumentTextIcon className="w-4 h-4 text-gray-500" />
                                                            <span className="text-xs font-medium">Transcript</span>
                                                        </button>
                                                        <button
                                                            onClick={() => openAIChat(m)}
                                                            className="flex-1 bg-purple-50 hover:bg-purple-100 text-purple-700 flex items-center justify-center gap-2 transition-colors px-3"
                                                        >
                                                            <SparklesIcon className="w-4 h-4 text-purple-500" />
                                                            <span className="text-xs font-medium">Ask AI</span>
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )
                        })()}
                    </div>

                    {/* Past Recordings Section */}
                    <div className="mt-8 pt-8 border-t border-gray-200">
                        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2 mb-4">
                            <FilmIcon className="w-5 h-5 text-gray-400" />
                            Past Recordings
                        </h2>
                        <RecordingsList
                            classroomId={classroomId}
                            accessToken={typeof window !== 'undefined' ? localStorage.getItem('accessToken') || '' : ''}
                        />
                    </div>
                </div>
            )}

            {/* Full Transcript Modal - rendered via portal to body */}
            {showTranscriptModal && selectedMeeting && mounted && createPortal(
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
                        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">📄 Full Transcript</h2>
                                <p className="text-sm text-gray-500">{selectedMeeting.title}</p>
                            </div>
                            <button
                                onClick={() => setShowTranscriptModal(false)}
                                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                            >
                                ✕
                            </button>
                        </div>
                        <div className="p-6 overflow-y-auto max-h-[60vh]">
                            <div className="prose prose-sm max-w-none">
                                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                                    <p className="text-sm text-gray-500">
                                        <strong>Duration:</strong> {selectedMeeting.duration_minutes} minutes<br />
                                        <strong>Date:</strong> {new Date(selectedMeeting.ended_at!).toLocaleDateString()}
                                    </p>
                                </div>
                                <h4 className="font-semibold text-gray-900 mb-2">Lecture Transcript</h4>
                                <p className="text-gray-700 whitespace-pre-wrap">
                                    {selectedMeeting.transcript || 'Transcript not available.'}
                                </p>
                                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                                    <p className="text-sm text-blue-700">
                                        <strong>Note:</strong> This transcript was auto-generated using AI speech-to-text.
                                        Some words may be inaccurate. For important information, please refer to the recording.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>,
                document.body
            )}

            {/* AI Chat Sidebar (Right Side) - rendered via portal to body */}
            {showAIChatModal && selectedMeeting && mounted && createPortal(
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 bg-black/30 z-40"
                        onClick={() => setShowAIChatModal(false)}
                    />

                    {/* Sidebar */}
                    <div className="fixed right-4 top-4 bottom-4 w-full max-w-sm bg-white shadow-2xl z-50 flex flex-col rounded-2xl overflow-hidden animate-slide-in-right">
                        {/* Header */}
                        <div className="p-3 flex items-center justify-between bg-gradient-to-r from-purple-600 to-pink-600 text-white">
                            <div>
                                <h2 className="text-lg font-bold flex items-center gap-2">
                                    <span>✨</span> Ask Questions
                                </h2>
                                <p className="text-sm text-purple-100 truncate max-w-[250px]">{selectedMeeting.title}</p>
                            </div>
                            <button
                                onClick={() => setShowAIChatModal(false)}
                                className="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-lg"
                            >
                                ✕
                            </button>
                        </div>

                        {/* Chat Messages */}
                        <div className="flex-1 overflow-y-auto p-3 space-y-3">
                            {chatMessages.map((msg, i) => (
                                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[85%] p-2.5 rounded-lg ${msg.role === 'user'
                                        ? 'bg-primary-600 text-white'
                                        : 'bg-gray-100 text-gray-800'
                                        }`}>
                                        {msg.role === 'ai' && <span className="text-xs text-gray-500 block mb-1">AI Assistant</span>}
                                        {msg.image && (
                                            <img src={msg.image} alt="Shared" className="max-h-32 rounded mb-2" />
                                        )}
                                        <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                                    </div>
                                </div>
                            ))}
                            {chatLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-gray-100 text-gray-800 p-3 rounded-lg">
                                        <span className="text-xs text-gray-500 block mb-1">AI Assistant</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Suggested Questions */}
                        <div className="px-3 py-2 border-t border-gray-100">
                            <div className="flex flex-wrap gap-1.5">
                                {['Summarize', 'Key concepts?', 'Examples?'].map((q, i) => (
                                    <button
                                        key={i}
                                        onClick={() => { setChatInput(q); }}
                                        className="px-2.5 py-1 text-xs bg-gray-100 text-gray-600 rounded-full hover:bg-gray-200"
                                    >
                                        {q}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Chat Input */}
                        <div className="p-3 border-t border-gray-200 bg-white">
                            <div className="flex gap-2 items-center">
                                <input
                                    type="text"
                                    value={chatInput}
                                    onChange={(e) => setChatInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                                    placeholder="Ask about this lecture..."
                                    className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none text-sm"
                                />
                                <button
                                    onClick={sendChatMessage}
                                    disabled={chatLoading || !chatInput.trim()}
                                    className="px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 text-sm"
                                >
                                    Send
                                </button>
                            </div>
                        </div>
                    </div>

                    <style jsx>{`
                        @keyframes slide-in-right {
                            from {
                                transform: translateX(100%);
                            }
                            to {
                                transform: translateX(0);
                            }
                        }
                        .animate-slide-in-right {
                            animation: slide-in-right 0.3s ease-out;
                        }
                    `}</style>
                </>,
                document.body
            )}

            {/* Assignments Tab */}
            {activeTab === 'assignments' && (
                <div className="space-y-4">
                    {assignments.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <AcademicCapIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No assignments yet</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {assignments.map(assignment => {
                                const isOverdue = assignment.due_date && new Date(assignment.due_date) < new Date()
                                const isSubmitted = !!assignment.my_submission
                                const isGraded = assignment.my_submission?.status === 'graded'

                                return (
                                    <div key={assignment.id} className="card">
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-3 mb-2">
                                                    <div className={`p-2 rounded-lg ${isGraded ? 'bg-green-100' : isSubmitted ? 'bg-blue-100' : 'bg-gray-100'}`}>
                                                        <AcademicCapIcon className={`w-5 h-5 ${isGraded ? 'text-green-600' : isSubmitted ? 'text-blue-600' : 'text-gray-600'}`} />
                                                    </div>
                                                    <div className="flex-1">
                                                        <h3 className="font-medium text-gray-900">{assignment.title}</h3>
                                                        <div className="flex items-center gap-3 mt-1 flex-wrap">
                                                            {assignment.points && (
                                                                <span className="text-sm text-gray-600">
                                                                    {assignment.points} points
                                                                </span>
                                                            )}
                                                            {assignment.due_date && (
                                                                <span className={`text-sm flex items-center gap-1 ${isOverdue && !isSubmitted ? 'text-red-600' : 'text-gray-600'}`}>
                                                                    Due: {new Date(assignment.due_date).toLocaleString()}
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>

                                                {assignment.description && (
                                                    <p className="text-gray-600 text-sm mb-3 ml-14">{assignment.description}</p>
                                                )}

                                                {/* Attachments */}
                                                {assignment.attachments && assignment.attachments.length > 0 && (
                                                    <div className="ml-14 mt-2">
                                                        <p className="text-xs text-gray-500 mb-1">Attachments:</p>
                                                        <div className="flex flex-wrap gap-2">
                                                            {assignment.attachments.map((att, idx) => (
                                                                <a
                                                                    key={idx}
                                                                    href={fixFileUrl(att.url)}
                                                                    target="_blank"
                                                                    rel="noopener noreferrer"
                                                                    className="text-sm text-primary-600 hover:underline flex items-center gap-1 bg-gray-50 px-2 py-1 rounded"
                                                                >
                                                                    {att.type === 'file' ? (
                                                                        <DocumentTextIcon className="w-4 h-4" />
                                                                    ) : (
                                                                        <LinkIcon className="w-4 h-4" />
                                                                    )}
                                                                    {att.filename || att.url}
                                                                </a>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Submission Status */}
                                                <div className="ml-14 mt-3 flex items-center gap-2 flex-wrap">
                                                    {isGraded ? (
                                                        <>
                                                            <span className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full flex items-center gap-1">
                                                                <CheckCircleIcon className="w-4 h-4" />
                                                                Graded: {assignment.my_submission?.grade}/{assignment.points} points
                                                            </span>
                                                            {assignment.my_submission?.ai_graded && (
                                                                <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
                                                                    AI Graded {assignment.my_submission?.ai_confidence ? `(${Math.round(assignment.my_submission.ai_confidence * 100)}% confidence)` : ''}
                                                                </span>
                                                            )}
                                                        </>
                                                    ) : isSubmitted ? (
                                                        <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full flex items-center gap-1">
                                                            <CheckCircleIcon className="w-4 h-4" />
                                                            {assignment.my_submission?.status === 'grading' ? 'Grading in progress...' : 'Submitted'}
                                                        </span>
                                                    ) : (
                                                        <span className={`text-sm px-3 py-1 rounded-full ${isOverdue ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700'}`}>
                                                            {isOverdue ? 'Overdue' : 'Not submitted'}
                                                        </span>
                                                    )}
                                                </div>

                                                {/* Submitted Files & Feedback */}
                                                {isSubmitted && assignment.my_submission && (
                                                    <div className="ml-14 mt-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                                                        {/* Submitted Files */}
                                                        {assignment.my_submission.files && assignment.my_submission.files.length > 0 && (
                                                            <div className="mb-3">
                                                                <p className="text-xs text-gray-500 mb-2 font-medium">Your Submitted Work:</p>
                                                                <div className="flex flex-wrap gap-2">
                                                                    {assignment.my_submission.files.map((file: { id: string; url: string; filename: string }, idx: number) => (
                                                                        <a
                                                                            key={idx}
                                                                            href={fixFileUrl(file.url)}
                                                                            target="_blank"
                                                                            rel="noopener noreferrer"
                                                                            className="text-sm text-blue-600 hover:underline flex items-center gap-1 bg-white px-3 py-2 rounded border border-gray-200 hover:bg-blue-50 transition-colors"
                                                                        >
                                                                            <DocumentTextIcon className="w-4 h-4" />
                                                                            {file.filename || 'Submission'}
                                                                        </a>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}

                                                        {/* Feedback */}
                                                        {isGraded && assignment.my_submission.feedback && (
                                                            <div className="mt-2 pt-2 border-t border-gray-200">
                                                                <p className="text-xs text-gray-500 mb-1 font-medium">Feedback:</p>
                                                                <p className="text-sm text-gray-700">{assignment.my_submission.feedback}</p>
                                                            </div>
                                                        )}

                                                        {/* Detailed Feedback Button */}
                                                        {isGraded && assignment.my_submission.detailed_feedback && (
                                                            <button
                                                                onClick={() => {
                                                                    setSelectedAssignment(assignment)
                                                                    setShowDetailedFeedbackModal(true)
                                                                }}
                                                                className="mt-2 text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1"
                                                            >
                                                                <EyeIcon className="w-4 h-4" />
                                                                View Detailed Feedback
                                                            </button>
                                                        )}
                                                    </div>
                                                )}
                                            </div>

                                            {/* Action Button */}
                                            {!isGraded && (
                                                <button
                                                    onClick={() => openSubmissionModal(assignment)}
                                                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
                                                >
                                                    {isSubmitted ? 'Update' : 'Submit'}
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* AI Q&A for Recordings - Only show on sessions tab */}
            {activeTab === 'sessions' && typeof window !== 'undefined' && (
                <MeetingQA
                    classroomId={classroomId}
                    accessToken={localStorage.getItem('accessToken') || ''}
                    aiServiceUrl={process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:9001'}
                />
            )}

            {/* Results Tab */}
            {activeTab === 'results' && (
                <div className="space-y-4">
                    <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                        <CheckCircleIcon className="w-5 h-5 text-gray-400" />
                        My Exam Results
                    </h2>

                    {loadingResults ? (
                        <div className="flex items-center justify-center py-12">
                            <div className="spinner"></div>
                        </div>
                    ) : examResults.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <CheckCircleIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No results declared yet</p>
                            <p className="text-sm text-gray-400 mt-1">Your exam results will appear here once declared by your teacher</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {examResults.map((result) => (
                                <div key={result.id} className="card border-l-4 border-l-green-500">
                                    <div className="flex items-center justify-between mb-3">
                                        <div>
                                            <h3 className="font-semibold text-gray-900">{result.exam_session.name}</h3>
                                            <div className="flex items-center gap-3 text-sm text-gray-500 mt-1">
                                                <span>{result.exam_session.subject}</span>
                                                <span>•</span>
                                                <span>{result.exam_session.class_name}</span>
                                                <span>•</span>
                                                <span>{new Date(result.exam_session.date).toLocaleDateString()}</span>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="flex items-center gap-2">
                                                <span className={`text-3xl font-bold ${result.grade === 'A+' || result.grade === 'A' ? 'text-green-600' :
                                                    result.grade === 'B' ? 'text-blue-600' :
                                                        result.grade === 'C' ? 'text-yellow-600' :
                                                            result.grade === 'D' ? 'text-orange-600' : 'text-red-600'
                                                    }`}>
                                                    {result.grade}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-500 mt-1">
                                                {result.total_score}/{result.max_score} ({Math.round(result.percentage)}%)
                                            </p>
                                        </div>
                                    </div>

                                    {/* Expand/Collapse Details */}
                                    <button
                                        onClick={() => setExpandedResult(expandedResult === result.id ? null : result.id)}
                                        className="text-sm text-primary-600 hover:underline flex items-center gap-1"
                                    >
                                        {expandedResult === result.id ? 'Hide Details' : 'View Question-wise Breakdown'}
                                    </button>

                                    {expandedResult === result.id && result.question_evaluations && (
                                        <div className="mt-4 pt-4 border-t border-gray-100 space-y-2">
                                            {result.question_evaluations.map((qe, idx) => (
                                                <div key={idx} className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-lg">
                                                    <span className="text-sm text-gray-700">Q{qe.question_number}</span>
                                                    <div className="flex items-center gap-4">
                                                        {qe.feedback && (
                                                            <span className="text-xs text-gray-500 max-w-xs truncate">{qe.feedback}</span>
                                                        )}
                                                        <span className={`font-medium ${qe.score >= qe.max_marks * 0.8 ? 'text-green-600' :
                                                            qe.score >= qe.max_marks * 0.5 ? 'text-yellow-600' : 'text-red-600'
                                                            }`}>
                                                            {qe.score}/{qe.max_marks}
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    <p className="text-xs text-gray-400 mt-3">
                                        Results declared on {new Date(result.exam_session.results_declared_at).toLocaleString()}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Submission Modal */}
            {showSubmissionModal && selectedAssignment && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="flex items-center justify-between p-6 border-b border-gray-200">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Submit Assignment</h2>
                                <p className="text-sm text-gray-500">{selectedAssignment.title}</p>
                            </div>
                            <button
                                onClick={() => setShowSubmissionModal(false)}
                                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                            >
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="p-6 space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Your Submission
                                </label>

                                {/* Upload PDF Button */}
                                <div className="flex gap-2 mb-3">
                                    <button
                                        onClick={() => submissionFileRef.current?.click()}
                                        className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center gap-2"
                                    >
                                        <CloudArrowUpIcon className="w-4 h-4" />
                                        Upload PDF
                                    </button>
                                    <input
                                        ref={submissionFileRef}
                                        type="file"
                                        accept=".pdf,application/pdf"
                                        multiple
                                        onChange={handleSubmissionFileUpload}
                                        className="hidden"
                                    />
                                </div>

                                {/* Add Link Input */}
                                <div className="flex gap-2 mb-3">
                                    <input
                                        type="url"
                                        value={submissionLinkInput}
                                        onChange={(e) => setSubmissionLinkInput(e.target.value)}
                                        placeholder="https://example.com/my-work"
                                        className="input-field flex-1"
                                        onKeyPress={(e) => e.key === 'Enter' && addSubmissionLink()}
                                    />
                                    <button
                                        onClick={addSubmissionLink}
                                        className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center gap-2"
                                    >
                                        <LinkIcon className="w-4 h-4" />
                                        Add Link
                                    </button>
                                </div>

                                {/* Submission Files List */}
                                {submissionFiles.length > 0 && (
                                    <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                                        {submissionFiles.map((file, idx) => (
                                            <div key={idx} className="flex items-center justify-between bg-white p-2 rounded">
                                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                                    {file.type === 'file' ? (
                                                        <DocumentTextIcon className="w-4 h-4 text-primary-600 flex-shrink-0" />
                                                    ) : (
                                                        <LinkIcon className="w-4 h-4 text-primary-600 flex-shrink-0" />
                                                    )}
                                                    <span className="text-sm text-gray-700 truncate">{file.filename}</span>
                                                </div>
                                                <button
                                                    onClick={() => removeSubmissionFile(idx)}
                                                    className="p-1 text-red-500 hover:bg-red-50 rounded"
                                                >
                                                    <TrashIcon className="w-4 h-4" />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="p-6 border-t border-gray-100 flex justify-end gap-3">
                            <button
                                onClick={() => setShowSubmissionModal(false)}
                                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={submitAssignment}
                                disabled={submitting || submissionFiles.length === 0}
                                className="btn-primary flex items-center gap-2 disabled:opacity-50"
                            >
                                {submitting ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                        Submitting...
                                    </>
                                ) : (
                                    <>
                                        <CloudArrowUpIcon className="w-4 h-4" />
                                        {selectedAssignment.my_submission ? 'Update Submission' : 'Submit Assignment'}
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Detailed Feedback Modal */}
            {showDetailedFeedbackModal && selectedAssignment?.my_submission?.detailed_feedback && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-xl max-w-3xl w-full max-h-[90vh] overflow-hidden shadow-2xl">
                        <div className="p-6 border-b border-gray-100 flex items-center justify-between">
                            <div>
                                <h3 className="text-lg font-semibold text-gray-900">Detailed Feedback</h3>
                                <p className="text-sm text-gray-500">{selectedAssignment.title}</p>
                            </div>
                            <button
                                onClick={() => setShowDetailedFeedbackModal(false)}
                                className="p-2 hover:bg-gray-100 rounded-full"
                            >
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="p-6 overflow-y-auto max-h-[60vh]">
                            {/* Overall Grade Summary */}
                            <div className="mb-6 p-4 bg-gradient-to-r from-primary-50 to-blue-50 rounded-xl">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-gray-600">Total Score</p>
                                        <p className="text-3xl font-bold text-primary-700">
                                            {selectedAssignment.my_submission.detailed_feedback.total_grade || selectedAssignment.my_submission.grade}
                                            <span className="text-lg text-gray-500">/{selectedAssignment.my_submission.detailed_feedback.max_points || selectedAssignment.points}</span>
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm text-gray-600">Percentage</p>
                                        <p className="text-2xl font-semibold text-gray-700">
                                            {Math.round(selectedAssignment.my_submission.detailed_feedback.percentage || 0)}%
                                        </p>
                                    </div>
                                    {selectedAssignment.my_submission.ai_confidence && (
                                        <div className="text-right">
                                            <p className="text-sm text-gray-600">AI Confidence</p>
                                            <p className={`text-lg font-semibold ${selectedAssignment.my_submission.ai_confidence >= 0.7 ? 'text-green-600' : selectedAssignment.my_submission.ai_confidence >= 0.5 ? 'text-yellow-600' : 'text-red-600'}`}>
                                                {Math.round(selectedAssignment.my_submission.ai_confidence * 100)}%
                                            </p>
                                        </div>
                                    )}
                                </div>
                                <p className="mt-3 text-gray-700">{selectedAssignment.my_submission.detailed_feedback.overall_feedback || selectedAssignment.my_submission.feedback}</p>
                            </div>

                            {/* Per-Question Grades */}
                            {selectedAssignment.my_submission.detailed_feedback.question_grades && (
                                <div className="space-y-3">
                                    <h4 className="font-medium text-gray-900">Question-by-Question Breakdown</h4>
                                    {selectedAssignment.my_submission.detailed_feedback.question_grades.map((qg: {
                                        question_number: number;
                                        question_text: string;
                                        student_answer: string;
                                        points_earned: number;
                                        max_points: number;
                                        percentage: number;
                                        feedback: string;
                                        confidence: number;
                                    }, idx: number) => (
                                        <div key={idx} className="p-4 border border-gray-200 rounded-lg bg-gray-50">
                                            <div className="flex items-start justify-between mb-2">
                                                <span className="font-medium text-gray-900">Q{qg.question_number}</span>
                                                <span className={`px-2 py-1 rounded text-sm font-medium ${qg.percentage >= 75 ? 'bg-green-100 text-green-700' :
                                                    qg.percentage >= 50 ? 'bg-yellow-100 text-yellow-700' :
                                                        qg.percentage >= 25 ? 'bg-orange-100 text-orange-700' :
                                                            'bg-red-100 text-red-700'
                                                    }`}>
                                                    {qg.points_earned.toFixed(1)}/{qg.max_points.toFixed(1)}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-700 mb-2">{qg.question_text}</p>
                                            <p className="text-xs text-gray-500 mb-2">
                                                <strong>Your Answer:</strong> {qg.student_answer || '[No answer provided]'}
                                            </p>
                                            <p className="text-sm text-primary-700 italic">{qg.feedback}</p>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="p-4 border-t border-gray-100 flex justify-end">
                            <button
                                onClick={() => setShowDetailedFeedbackModal(false)}
                                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Syllabus View Modal - Using PDFViewer */}
            {showSyllabusModal && classroom?.syllabus_url && (
                <div className="fixed inset-0 bg-black/90 z-50 flex flex-col">
                    <PDFViewer
                        pdfUrl={fixFileUrl(classroom.syllabus_url)}
                        title={`${classroom.name} - Syllabus`}
                        onClose={() => setShowSyllabusModal(false)}
                    />
                </div>
            )}

            {/* INTEGRATION NOTE: Document Upload Modal for Materials tab */}
            {showDocumentUploadModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full overflow-hidden">
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-gray-200">
                            <div className="flex items-center gap-3">
                                <CloudArrowUpIcon className="w-6 h-6 text-primary-600" />
                                <h2 className="text-xl font-bold text-gray-900">Upload Document</h2>
                            </div>
                            <button
                                onClick={() => {
                                    setShowDocumentUploadModal(false)
                                    setDocumentUploadFile(null)
                                    setDocumentUploadTitle('')
                                }}
                                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                            >
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>

                        {/* Body */}
                        <div className="p-6 space-y-4">
                            {/* File Drop Zone */}
                            <div
                                onClick={() => documentFileRef.current?.click()}
                                className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-primary-400 hover:bg-primary-50/50 cursor-pointer transition-colors"
                            >
                                <input
                                    ref={documentFileRef}
                                    type="file"
                                    accept=".pdf,.pptx,.docx,.doc,.png,.jpg,.jpeg"
                                    onChange={(e) => {
                                        const file = e.target.files?.[0]
                                        if (file) {
                                            setDocumentUploadFile(file)
                                            if (!documentUploadTitle) {
                                                setDocumentUploadTitle(file.name.replace(/\.[^/.]+$/, ''))
                                            }
                                        }
                                    }}
                                    className="hidden"
                                />
                                {documentUploadFile ? (
                                    <div>
                                        <CheckCircleIcon className="w-12 h-12 mx-auto text-green-500 mb-2" />
                                        <p className="font-medium text-gray-900">{documentUploadFile.name}</p>
                                        <p className="text-sm text-gray-500">
                                            {(documentUploadFile.size / (1024 * 1024)).toFixed(2)} MB
                                        </p>
                                    </div>
                                ) : (
                                    <div>
                                        <CloudArrowUpIcon className="w-12 h-12 mx-auto text-gray-400 mb-2" />
                                        <p className="font-medium text-gray-900">Click to select file</p>
                                        <p className="text-sm text-gray-500">
                                            PDF, PPTX, DOCX, or Images (max 50MB)
                                        </p>
                                    </div>
                                )}
                            </div>

                            {/* Title input */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Document Title
                                </label>
                                <input
                                    type="text"
                                    value={documentUploadTitle}
                                    onChange={(e) => setDocumentUploadTitle(e.target.value)}
                                    placeholder="e.g., Chapter 5 Notes"
                                    className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                />
                            </div>
                        </div>

                        {/* Footer */}
                        <div className="p-4 border-t border-gray-100 flex justify-end gap-3">
                            <button
                                onClick={() => {
                                    setShowDocumentUploadModal(false)
                                    setDocumentUploadFile(null)
                                    setDocumentUploadTitle('')
                                }}
                                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg"
                            >
                                Cancel
                            </button>
                            <button
                                disabled={documentUploading || !documentUploadFile || !documentUploadTitle}
                                onClick={async () => {
                                    if (!documentUploadFile || !documentUploadTitle) return

                                    setDocumentUploading(true)
                                    try {
                                        const formData = new FormData()
                                        formData.append('classroom_id', classroomId)
                                        formData.append('title', documentUploadTitle)
                                        formData.append('file', documentUploadFile)

                                        const res = await fetch(`${getApiBaseUrl()}/api/notes/upload`, {
                                            method: 'POST',
                                            headers: {
                                                'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                                            },
                                            body: formData
                                        })

                                        if (res.ok) {
                                            setShowDocumentUploadModal(false)
                                            setDocumentUploadFile(null)
                                            setDocumentUploadTitle('')
                                            alert('Document uploaded successfully! Processing will begin shortly.')
                                        } else {
                                            const error = await res.json()
                                            alert(`Upload failed: ${error.error || 'Unknown error'}`)
                                        }
                                    } catch (error) {
                                        console.error('Upload error:', error)
                                        alert('Upload failed. Please try again.')
                                    } finally {
                                        setDocumentUploading(false)
                                    }
                                }}
                                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {documentUploading ? (
                                    <>
                                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                        Uploading...
                                    </>
                                ) : (
                                    <>
                                        <CloudArrowUpIcon className="w-4 h-4" />
                                        Upload
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Document Viewer Modal - Full screen overlay */}
            {showDocumentViewer && viewingDocument && (
                <div className="fixed top-0 left-0 right-0 bottom-0 bg-black z-[100] flex flex-col" style={{ margin: 0, padding: 0 }}>
                    {viewingDocument.type.includes('pdf') ? (
                        /* For PDFs, use PDFViewer which has its own header */
                        <PDFViewer
                            pdfUrl={fixFileUrl(viewingDocument.url)}
                            title={viewingDocument.name}
                            fileSize={viewingDocument.size}
                            materialId={viewingDocument.id}
                            classroomId={classroomId}
                            onClose={() => {
                                setShowDocumentViewer(false)
                                setViewingDocument(null)
                            }}
                        />
                    ) : (
                        /* For other files, show the header + content */
                        <>
                            <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800">
                                <div className="flex items-center gap-3">
                                    <DocumentTextIcon className="w-5 h-5 text-primary-400" />
                                    <span className="text-white font-medium truncate max-w-[300px]">{viewingDocument.name}</span>
                                    <span className="text-gray-500 text-sm">{formatSize(viewingDocument.size)}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => window.open(fixFileUrl(viewingDocument.url), '_blank')}
                                        className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5"
                                        title="Open in new tab"
                                    >
                                        <EyeIcon className="w-4 h-4" />
                                        <span className="hidden sm:inline">Open in Tab</span>
                                    </button>
                                    <button
                                        onClick={() => {
                                            const link = document.createElement('a')
                                            link.href = fixFileUrl(viewingDocument.url)
                                            link.download = viewingDocument.name
                                            link.click()
                                        }}
                                        className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5"
                                        title="Download"
                                    >
                                        <ArrowDownTrayIcon className="w-4 h-4" />
                                        <span className="hidden sm:inline">Download</span>
                                    </button>
                                    <div className="w-px h-6 bg-gray-700 mx-1" />
                                    <button
                                        onClick={() => {
                                            setShowDocumentViewer(false)
                                            setViewingDocument(null)
                                        }}
                                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                                    >
                                        <XMarkIcon className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>

                            {/* Content */}
                            <div className="flex-1 overflow-hidden">
                                {viewingDocument.type.includes('image') ? (
                                    <ImageViewer
                                        imageUrl={viewingDocument.url}
                                        title={viewingDocument.name}
                                        onClose={() => {
                                            setShowDocumentViewer(false)
                                            setViewingDocument(null)
                                        }}
                                    />
                                ) : viewingDocument.type.includes('video') ? (
                                    <div className="flex items-center justify-center h-full p-8">
                                        <video
                                            src={viewingDocument.url}
                                            controls
                                            className="max-w-full max-h-full rounded-lg shadow-2xl"
                                        />
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-full text-center">
                                        <DocumentTextIcon className="w-16 h-16 text-gray-500 mb-4" />
                                        <p className="text-gray-400 mb-4">Preview not available for this file type</p>
                                        <button
                                            onClick={() => window.open(viewingDocument.url, '_blank')}
                                            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg flex items-center gap-2"
                                        >
                                            <ArrowDownTrayIcon className="w-5 h-5" />
                                            Download to view
                                        </button>
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    )
}
