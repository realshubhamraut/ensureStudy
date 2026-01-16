'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import {
    ArrowLeftIcon,
    UsersIcon,
    ClipboardDocumentIcon,
    Cog6ToothIcon,
    CloudArrowUpIcon,
    DocumentTextIcon,
    VideoCameraIcon,
    MusicalNoteIcon,
    PhotoIcon,
    TrashIcon,
    FolderIcon,
    MegaphoneIcon,
    PlusIcon,
    PlayIcon,
    PhoneXMarkIcon,
    BookOpenIcon,
    EyeIcon,
    AcademicCapIcon,
    LinkIcon,
    CalendarIcon,
    CheckCircleIcon,
    XMarkIcon,
    ArrowDownTrayIcon
} from '@heroicons/react/24/outline'
import PDFViewer from '@/components/PDFViewer'
import ImageViewer from '@/components/ImageViewer'

interface Student {
    id: string
    email: string
    username: string
    first_name: string
    last_name: string
    joined_at: string
}

interface Material {
    id: string
    name: string
    type: string
    size: number
    subject: string
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
    classroom_id: string
    title: string
    description: string
    due_date: string | null
    points: number | null
    status: 'draft' | 'published'
    created_at: string
    attachments?: AssignmentAttachment[]
    submission_count?: number
    submissions?: Submission[]
}

interface AssignmentAttachment {
    id?: string
    type: 'file' | 'link'
    url: string
    filename?: string
    file_size?: number
}

interface Submission {
    id: string
    assignment_id: string
    student_id: string
    submitted_at: string
    status: 'submitted' | 'graded'
    grade: number | null
    feedback: string | null
    student?: {
        id: string
        name: string
        email: string
    }
    files?: SubmissionFile[]
}

interface SubmissionFile {
    id: string
    url: string
    filename: string
    file_size: number
    type: 'file' | 'link'
}

interface Classroom {
    id: string
    name: string
    grade: string
    section: string
    subject: string
    join_code: string
    is_active: boolean
    syllabus_url?: string
    syllabus_content?: string
    syllabus_filename?: string
    has_syllabus?: boolean
}

type TabType = 'stream' | 'materials' | 'meet' | 'assignments' | 'students' | 'settings'

export default function TeacherClassroomDetailPage() {
    const params = useParams()
    const router = useRouter()
    const classroomId = params.id as string
    const fileInputRef = useRef<HTMLInputElement>(null)
    const syllabusFileRef = useRef<HTMLInputElement>(null)

    const [activeTab, setActiveTab] = useState<TabType>('stream')
    const [classroom, setClassroom] = useState<Classroom | null>(null)
    const [students, setStudents] = useState<Student[]>([])
    const [materials, setMaterials] = useState<Material[]>([])
    const [announcements, setAnnouncements] = useState<Announcement[]>([])
    const [meetings, setMeetings] = useState<Meeting[]>([])
    const [loading, setLoading] = useState(true)
    const [uploading, setUploading] = useState(false)
    const [newAnnouncement, setNewAnnouncement] = useState('')
    const [showMeetModal, setShowMeetModal] = useState(false)
    const [showUploadModal, setShowUploadModal] = useState(false)
    const [meetTitle, setMeetTitle] = useState('')
    const [scheduledAt, setScheduledAt] = useState('')
    const [enableRecording, setEnableRecording] = useState(true)
    const [uploadSubject, setUploadSubject] = useState('')
    const [selectedFiles, setSelectedFiles] = useState<File[]>([])
    const [syllabusText, setSyllabusText] = useState('')
    const [savingSyllabus, setSavingSyllabus] = useState(false)
    const [syllabusFile, setSyllabusFile] = useState<File | null>(null)
    const [showSyllabusModal, setShowSyllabusModal] = useState(false)

    // Document Viewer Modal state (for Materials)
    const [showDocumentViewer, setShowDocumentViewer] = useState(false)
    const [viewingDocument, setViewingDocument] = useState<Material | null>(null)

    // Assignment states
    const [assignments, setAssignments] = useState<Assignment[]>([])
    const [showAssignmentModal, setShowAssignmentModal] = useState(false)
    const [assignmentTitle, setAssignmentTitle] = useState('')
    const [assignmentDescription, setAssignmentDescription] = useState('')
    const [assignmentDueDate, setAssignmentDueDate] = useState('')
    const [assignmentPoints, setAssignmentPoints] = useState('')
    const [assignmentAttachments, setAssignmentAttachments] = useState<AssignmentAttachment[]>([])
    const [linkInput, setLinkInput] = useState('')
    const [savingAssignment, setSavingAssignment] = useState(false)
    const [selectedAssignment, setSelectedAssignment] = useState<Assignment | null>(null)
    const assignmentFileRef = useRef<HTMLInputElement>(null)

    // Submission viewer states
    const [showSubmissionsModal, setShowSubmissionsModal] = useState(false)
    const [viewingAssignment, setViewingAssignment] = useState<Assignment | null>(null)
    const [assignmentSubmissions, setAssignmentSubmissions] = useState<Submission[]>([])
    const [selectedSubmission, setSelectedSubmission] = useState<Submission | null>(null)
    const [gradeInput, setGradeInput] = useState('')
    const [feedbackInput, setFeedbackInput] = useState('')
    const [savingGrade, setSavingGrade] = useState(false)

    // Meeting query states
    const [meetingQuery, setMeetingQuery] = useState('')
    const [queryLoading, setQueryLoading] = useState(false)
    const [queryAnswer, setQueryAnswer] = useState<{ answer: string, sources: any[] } | null>(null)

    const subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'History', 'Geography', 'Computer Science']

    useEffect(() => {
        fetchClassroom()
    }, [classroomId])

    const fetchClassroom = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setClassroom(data.classroom)
                setStudents(data.students || [])

                // Fetch materials from API
                try {
                    const materialsRes = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/materials`, {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                        }
                    })
                    if (materialsRes.ok) {
                        const materialsData = await materialsRes.json()
                        setMaterials(materialsData.materials || [])
                    }
                } catch (e) {
                    console.error('Failed to fetch materials:', e)
                }

                // Fetch meetings from API (CRITICAL - for live meeting display)
                try {
                    console.log('Fetching meetings for classroom:', classroomId)
                    const meetingsRes = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/meetings`, {
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                        }
                    })
                    if (meetingsRes.ok) {
                        const meetingsData = await meetingsRes.json()
                        console.log('Fetched meetings:', meetingsData)
                        setMeetings(meetingsData.meetings || [])
                    }
                } catch (e) {
                    console.error('Failed to fetch meetings:', e)
                }

                // Fetch announcements from API (optional - may not exist)
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
            } else {
                alert('Classroom not found')
                router.push('/teacher/dashboard')
            }
        } catch (error) {
            console.error('Failed to fetch classroom:', error)
        } finally {
            setLoading(false)
        }
    }

    // Populate syllabus text when classroom loads
    useEffect(() => {
        if (classroom?.syllabus_content) {
            setSyllabusText(classroom.syllabus_content)
        }
    }, [classroom?.syllabus_content])

    const saveSyllabus = async (fileToUpload?: File) => {
        // Use passed file or fallback to state
        const file = fileToUpload || syllabusFile
        if (!classroom || !file) {
            alert('Please select a PDF file first')
            return
        }
        setSavingSyllabus(true)
        try {
            // Create FormData for file upload
            const formData = new FormData()
            formData.append('file', file)
            formData.append('classroom_id', classroomId)

            // Upload to file storage
            const uploadRes = await fetch(`${getApiBaseUrl()}/api/files/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: formData
            })

            let syllabusUrl = ''
            if (uploadRes.ok) {
                const uploadData = await uploadRes.json()
                syllabusUrl = uploadData.url
            } else {
                // Fallback: create a blob URL for demo
                syllabusUrl = URL.createObjectURL(file)
            }

            // Update classroom with syllabus URL
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/syllabus`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    syllabus_url: syllabusUrl,
                    syllabus_filename: file.name
                })
            })
            if (res.ok) {
                const data = await res.json()
                setClassroom(data.classroom)
                setSyllabusFile(null)
                alert('Syllabus uploaded successfully!')
            } else {
                alert('Failed to save syllabus')
            }
        } catch (error) {
            console.error('Upload error:', error)
            alert('Failed to upload syllabus')
        } finally {
            setSavingSyllabus(false)
        }
    }

    const deleteSyllabus = async () => {
        if (!confirm('Are you sure you want to remove the syllabus?')) return
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/syllabus`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                setClassroom({ ...classroom!, syllabus_url: undefined, syllabus_content: undefined, has_syllabus: false })
                setSyllabusText('')
                alert('Syllabus removed')
            }
        } catch (error) {
            alert('Failed to remove syllabus')
        }
    }

    const copyCode = () => {
        if (classroom) {
            navigator.clipboard.writeText(classroom.join_code)
            alert('Join code copied!')
        }
    }

    const postAnnouncement = async () => {
        if (!newAnnouncement.trim()) return

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/announcements`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: newAnnouncement })
            })

            if (!res.ok) {
                const error = await res.json()
                alert(`Failed to post announcement: ${error.error || 'Unknown error'}`)
                return
            }

            const data = await res.json()
            setAnnouncements([data.announcement, ...announcements])
            setNewAnnouncement('')
        } catch (error) {
            console.error('Failed to post announcement:', error)
            alert('Failed to post announcement. Please try again.')
        }
    }

    const startMeeting = async () => {
        if (!meetTitle.trim()) {
            alert('Enter meeting title')
            return
        }

        try {
            // Create meeting via API
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/meetings`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: meetTitle,
                    scheduled_at: scheduledAt || null,
                    is_recording_enabled: enableRecording
                })
            })

            if (!res.ok) {
                const error = await res.json()
                alert(`Failed to create meeting: ${error.error || 'Unknown error'}`)
                return
            }

            const data = await res.json()
            const meeting: Meeting = {
                id: data.meeting.id,
                title: data.meeting.title,
                status: scheduledAt ? 'scheduled' : 'scheduled',
                scheduled_at: data.meeting.scheduled_at,
                meeting_link: data.meeting.meeting_link
            }

            setMeetings([meeting, ...meetings])
            setShowMeetModal(false)
            setMeetTitle('')
            setScheduledAt('')
            setEnableRecording(true)

            // If no scheduled time, start immediately and navigate to meeting
            if (!scheduledAt) {
                await fetch(`${getApiBaseUrl()}/api/meeting/${meeting.id}/start`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })

                // Update local state
                setMeetings(prev => prev.map(m =>
                    m.id === meeting.id ? { ...m, status: 'live' as const, started_at: new Date().toISOString() } : m
                ))

                // Navigate to meeting
                router.push(`/meet/${meeting.id}`)
            } else {
                // Meeting scheduled for later - show confirmation
                alert(`Meeting "${meetTitle}" scheduled for ${new Date(scheduledAt).toLocaleString()}`)
            }

        } catch (error) {
            console.error('Failed to create meeting:', error)
            alert('Failed to create meeting. Please try again.')
        }
    }

    const endMeeting = async (id: string) => {
        const meeting = meetings.find(m => m.id === id)
        if (!meeting) return

        try {
            // Call backend API to end meeting
            const res = await fetch(`${getApiBaseUrl()}/api/meeting/${id}/end`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (!res.ok) {
                const error = await res.json()
                alert(`Failed to end meeting: ${error.error || 'Unknown error'}`)
                return
            }

            const data = await res.json()

            // Update local state with response from server
            setMeetings(meetings.map(m => m.id === id ? {
                ...m,
                status: 'ended' as const,
                ended_at: data.meeting.ended_at,
                duration_minutes: data.meeting.duration_minutes
            } : m))

            alert('Meeting ended! Recording will be processed.')

        } catch (error) {
            console.error('Failed to end meeting:', error)
            alert('Failed to end meeting. Please try again.')
        }
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files
        if (!files || files.length === 0) return
        setSelectedFiles(Array.from(files))
        setShowUploadModal(true)
        if (fileInputRef.current) fileInputRef.current.value = ''
    }

    const confirmUpload = async () => {
        setUploading(true)
        setShowUploadModal(false)

        const uploadedMaterials: Material[] = []

        for (const file of selectedFiles) {
            try {
                // Step 1: Upload file to storage
                const formData = new FormData()
                formData.append('file', file)

                const uploadRes = await fetch(`${getApiBaseUrl()}/api/files/upload`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    },
                    body: formData
                })

                if (!uploadRes.ok) {
                    console.error(`Failed to upload ${file.name}`)
                    continue
                }

                const uploadData = await uploadRes.json()

                // Step 2: Save material record to classroom
                const materialRes = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/materials`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: file.name,
                        url: uploadData.url,
                        type: file.type,
                        size: file.size
                    })
                })

                if (materialRes.ok) {
                    const materialData = await materialRes.json()
                    uploadedMaterials.push(materialData.material)
                }
            } catch (error) {
                console.error(`Error uploading ${file.name}:`, error)
            }
        }

        // Add uploaded materials to state
        if (uploadedMaterials.length > 0) {
            setMaterials([...uploadedMaterials, ...materials])
            alert(`Successfully uploaded ${uploadedMaterials.length} file(s)`)
        } else {
            alert('Failed to upload files. Please try again.')
        }

        setUploading(false)
        setSelectedFiles([])
        setUploadSubject('')
    }

    const deleteMaterial = (id: string) => {
        if (confirm('Delete this file?')) {
            setMaterials(materials.filter(m => m.id !== id))
        }
    }

    const toggleActive = async () => {
        if (!classroom) return
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ is_active: !classroom.is_active })
            })
            if (res.ok) {
                setClassroom({ ...classroom, is_active: !classroom.is_active })
            }
        } catch (error) {
            alert('Failed to update')
        }
    }

    const getFileIcon = (type: string) => {
        if (type.includes('video')) return <VideoCameraIcon className="w-5 h-5 text-red-500" />
        if (type.includes('audio')) return <MusicalNoteIcon className="w-5 h-5 text-purple-500" />
        if (type.includes('image')) return <PhotoIcon className="w-5 h-5 text-green-500" />
        return <DocumentTextIcon className="w-5 h-5 text-blue-500" />
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

    const openAssignmentModal = () => {
        setAssignmentTitle('')
        setAssignmentDescription('')
        setAssignmentDueDate('')
        setAssignmentPoints('')
        setAssignmentAttachments([])
        setLinkInput('')
        setShowAssignmentModal(true)
    }

    const handleAssignmentFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files
        if (!files || files.length === 0) return

        for (const file of Array.from(files)) {
            if (file.type === 'application/pdf') {
                // In production, upload to file storage
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
                    let uploadSuccess = false
                    if (uploadRes.ok) {
                        const uploadData = await uploadRes.json()
                        fileUrl = uploadData.url
                        uploadSuccess = true
                    } else {
                        // Fallback to blob URL
                        fileUrl = URL.createObjectURL(file)
                    }

                    setAssignmentAttachments(prev => [...prev, {
                        type: 'file',
                        url: fileUrl,
                        filename: file.name,
                        file_size: file.size
                    }])

                    // Show success message
                    if (uploadSuccess) {
                        console.log(`✅ Uploaded: ${file.name}`)
                    }
                } catch (error) {
                    console.error('Upload failed:', error)
                    // Still add file with blob URL as fallback
                    setAssignmentAttachments(prev => [...prev, {
                        type: 'file',
                        url: URL.createObjectURL(file),
                        filename: file.name,
                        file_size: file.size
                    }])
                }
            } else {
                alert('Only PDF files are allowed')
            }
        }

        if (assignmentFileRef.current) assignmentFileRef.current.value = ''
    }

    const addLink = () => {
        if (!linkInput.trim()) return
        setAssignmentAttachments(prev => [...prev, {
            type: 'link',
            url: linkInput,
            filename: linkInput
        }])
        setLinkInput('')
    }

    const removeAttachment = (index: number) => {
        setAssignmentAttachments(prev => prev.filter((_, i) => i !== index))
    }

    const saveAssignment = async () => {
        if (!assignmentTitle.trim()) {
            alert('Please enter assignment title')
            return
        }

        setSavingAssignment(true)
        try {
            const assignmentData = {
                title: assignmentTitle,
                description: assignmentDescription,
                due_date: assignmentDueDate || null,
                points: assignmentPoints ? parseInt(assignmentPoints) : null,
                status: 'published',
                attachments: assignmentAttachments
            }

            console.log('Creating assignment with data:', assignmentData)

            const res = await fetch(`${getApiBaseUrl()}/api/classroom/${classroomId}/assignments`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(assignmentData)
            })

            console.log('Response status:', res.status)

            if (res.ok) {
                const data = await res.json()
                console.log('Assignment created:', data)
                setAssignments([data.assignment, ...assignments])
                setShowAssignmentModal(false)
                // Clear form
                setAssignmentTitle('')
                setAssignmentDescription('')
                setAssignmentDueDate('')
                setAssignmentPoints('')
                setAssignmentAttachments([])
                alert('Assignment created successfully!')
            } else {
                const errorData = await res.json().catch(() => ({ error: 'Unknown error' }))
                console.error('Backend error:', errorData)
                alert(`Failed to create assignment: ${errorData.error || 'Unknown error'}`)
            }
        } catch (error) {
            console.error('Save failed with exception:', error)
            alert(`Failed to create assignment: ${error}`)
        } finally {
            setSavingAssignment(false)
        }
    }

    const deleteAssignment = async (assignmentId: string) => {
        if (!confirm('Delete this assignment?')) return

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/assignment/${assignmentId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })

            if (res.ok) {
                setAssignments(assignments.filter(a => a.id !== assignmentId))
                alert('Assignment deleted')
            }
        } catch (error) {
            alert('Failed to delete assignment')
        }
    }

    // Fetch submissions for an assignment
    const fetchSubmissions = async (assignmentId: string) => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/assignment/${assignmentId}/submissions`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setAssignmentSubmissions(data.submissions || [])
            }
        } catch (error) {
            console.error('Failed to fetch submissions:', error)
        }
    }

    // Open submissions modal for an assignment
    const openSubmissionsModal = async (assignment: Assignment) => {
        setViewingAssignment(assignment)
        setShowSubmissionsModal(true)
        await fetchSubmissions(assignment.id)
    }

    // Open grade modal for a specific submission
    const openGradeModal = (submission: Submission) => {
        setSelectedSubmission(submission)
        setGradeInput(submission.grade?.toString() || '')
        setFeedbackInput(submission.feedback || '')
    }

    // Save grade for a submission
    const saveGrade = async () => {
        if (!selectedSubmission) return

        const grade = parseInt(gradeInput)
        if (isNaN(grade) || grade < 0) {
            alert('Please enter a valid grade')
            return
        }

        setSavingGrade(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/submission/${selectedSubmission.id}/grade`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    grade: grade,
                    feedback: feedbackInput || null
                })
            })

            if (res.ok) {
                // Update the submission in local state
                setAssignmentSubmissions(prev =>
                    prev.map(s =>
                        s.id === selectedSubmission.id
                            ? { ...s, grade, feedback: feedbackInput, status: 'graded' as const }
                            : s
                    )
                )
                setSelectedSubmission(null)
                alert('Grade saved successfully!')
            } else {
                const error = await res.json()
                alert(`Failed to save grade: ${error.error}`)
            }
        } catch (error) {
            console.error('Grade save failed:', error)
            alert('Failed to save grade')
        } finally {
            setSavingGrade(false)
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
                <Link href="/teacher/dashboard" className="text-primary-600 hover:underline">Back</Link>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-4">
                <Link href="/teacher/dashboard" className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg">
                    <ArrowLeftIcon className="w-5 h-5" />
                </Link>
                <div className="flex-1">
                    <h1 className="text-2xl font-bold text-gray-900">{classroom.name}</h1>
                    <p className="text-gray-600">
                        {classroom.grade && `Grade ${classroom.grade}`}
                        {classroom.section && ` • ${classroom.section}`}
                        {classroom.subject && ` • ${classroom.subject}`}
                    </p>
                </div>
            </div>

            {/* Join Code and Syllabus Section */}
            <div className="grid md:grid-cols-2 gap-4">
                {/* Join Code Card */}
                <div className="card bg-gradient-to-r from-purple-500 to-pink-500 text-white">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-purple-100 text-sm">Share this code with students</p>
                            <code className="text-3xl font-mono font-bold tracking-widest">{classroom.join_code}</code>
                        </div>
                        <button onClick={copyCode} className="p-3 bg-white/20 hover:bg-white/30 rounded-xl">
                            <ClipboardDocumentIcon className="w-6 h-6" />
                        </button>
                    </div>
                </div>

                {/* Syllabus Upload Card */}
                <div className="card bg-gradient-to-r from-blue-500 to-cyan-500 text-white">
                    <div className="flex items-center justify-between">
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                                <BookOpenIcon className="w-5 h-5" />
                                <p className="text-blue-100 text-sm">Class Syllabus (PDF)</p>
                            </div>
                            {classroom.has_syllabus && classroom.syllabus_url ? (
                                <div>
                                    <p className="font-medium text-sm truncate">{classroom.syllabus_filename || 'Syllabus.pdf'}</p>
                                    <div className="flex gap-2 mt-2">
                                        <button
                                            onClick={() => setShowSyllabusModal(true)}
                                            className="px-3 py-1.5 bg-white/20 hover:bg-white/30 rounded-lg text-sm flex items-center gap-1"
                                        >
                                            <EyeIcon className="w-4 h-4" />
                                            View
                                        </button>
                                        <button
                                            onClick={deleteSyllabus}
                                            className="px-3 py-1.5 bg-white/20 hover:bg-white/30 rounded-lg text-sm flex items-center gap-1"
                                        >
                                            <TrashIcon className="w-4 h-4" />
                                            Remove
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div>
                                    <p className="text-sm">No syllabus uploaded</p>
                                    <button
                                        onClick={() => syllabusFileRef.current?.click()}
                                        className="mt-2 px-3 py-1.5 bg-white/20 hover:bg-white/30 rounded-lg text-sm flex items-center gap-1"
                                    >
                                        <CloudArrowUpIcon className="w-4 h-4" />
                                        Upload PDF
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Hidden file input */}
                    <input
                        ref={syllabusFileRef}
                        type="file"
                        accept=".pdf,application/pdf"
                        onChange={(e) => {
                            const file = e.target.files?.[0]
                            if (file && file.type === 'application/pdf') {
                                setSyllabusFile(file)
                                // Pass file directly to upload function
                                saveSyllabus(file)
                            } else if (file) {
                                alert('Please select a PDF file')
                            }
                        }}
                        className="hidden"
                    />
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex rounded-xl bg-gray-100 p-1 overflow-x-auto">
                {[
                    { id: 'stream', label: 'Stream', icon: MegaphoneIcon },
                    { id: 'materials', label: 'Materials', icon: FolderIcon },
                    { id: 'meet', label: 'Meet', icon: VideoCameraIcon },
                    { id: 'assignments', label: `Assignments (${assignments.length})`, icon: AcademicCapIcon },
                    { id: 'students', label: `Students (${students.length})`, icon: UsersIcon },
                    { id: 'settings', label: 'Settings', icon: Cog6ToothIcon },
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as TabType)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap ${activeTab === tab.id ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Stream Tab (Announcements) */}
            {activeTab === 'stream' && (
                <div className="space-y-4">
                    {/* Post Announcement */}
                    <div className="card">
                        <h3 className="font-medium text-gray-900 mb-3">Post Announcement</h3>
                        <textarea
                            value={newAnnouncement}
                            onChange={(e) => setNewAnnouncement(e.target.value)}
                            placeholder="Share something with your class..."
                            className="input-field min-h-[100px] resize-none"
                        />
                        <div className="flex justify-end mt-3">
                            <button onClick={postAnnouncement} className="btn-primary">
                                Post
                            </button>
                        </div>
                    </div>

                    {/* Announcements List */}
                    {announcements.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <MegaphoneIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No announcements yet</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {announcements.map(a => (
                                <div key={a.id} className="card">
                                    <p className="text-gray-900">{a.message}</p>
                                    <p className="text-xs text-gray-500 mt-2">
                                        {new Date(a.created_at).toLocaleString()}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Materials Tab */}
            {activeTab === 'materials' && (
                <div className="space-y-4">
                    <div onClick={() => fileInputRef.current?.click()} className="card border-2 border-dashed border-gray-300 hover:border-primary-400 cursor-pointer">
                        <div className="text-center py-8">
                            <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-600">{uploading ? 'Uploading...' : 'Click to upload materials'}</p>
                            <p className="text-sm text-gray-400 mt-1">PDFs, Videos, Images, Documents</p>
                        </div>
                    </div>
                    <input ref={fileInputRef} type="file" multiple onChange={handleFileSelect} className="hidden" />

                    {materials.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <FolderIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No materials uploaded yet</p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {materials.map(m => (
                                <div key={m.id} className="card-hover flex items-center justify-between">
                                    <div
                                        className="flex items-center gap-3 flex-1 cursor-pointer hover:opacity-80"
                                        onClick={() => {
                                            if (m.type.includes('pdf') || m.type.includes('image')) {
                                                setViewingDocument(m)
                                                setShowDocumentViewer(true)
                                            } else {
                                                window.open('#', '_blank') // Mock URL
                                            }
                                        }}
                                    >
                                        {getFileIcon(m.type)}
                                        <div>
                                            <p className="font-medium text-gray-900">{m.name}</p>
                                            <span className="text-xs text-gray-500">{formatSize(m.size)}</span>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        {/* View Button */}
                                        {m.type.includes('pdf') && (
                                            <button
                                                onClick={() => {
                                                    setViewingDocument(m)
                                                    setShowDocumentViewer(true)
                                                }}
                                                className="p-2 text-gray-400 hover:text-primary-600"
                                                title="View document"
                                            >
                                                <EyeIcon className="w-5 h-5" />
                                            </button>
                                        )}
                                        {/* Delete Button */}
                                        <button onClick={() => deleteMaterial(m.id)} className="p-2 text-gray-400 hover:text-red-500">
                                            <TrashIcon className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Meet Tab */}
            {activeTab === 'meet' && (
                <div className="space-y-4">
                    {/* Meeting Query Card */}
                    <div className="card bg-gradient-to-r from-purple-50 to-pink-50 border-purple-200">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-white">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456Z" />
                                </svg>
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-900">Ask About Meetings</h3>
                                <p className="text-xs text-gray-500">Query past meeting content using AI</p>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={meetingQuery}
                                onChange={(e) => setMeetingQuery(e.target.value)}
                                onKeyDown={async (e) => {
                                    if (e.key === 'Enter' && meetingQuery.trim()) {
                                        setQueryLoading(true)
                                        try {
                                            const res = await fetch(`${getApiBaseUrl().replace(':8000', ':8001')}/api/meetings/query`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({ classroom_id: classroomId, query: meetingQuery })
                                            })
                                            if (res.ok) {
                                                const data = await res.json()
                                                setQueryAnswer({ answer: data.answer, sources: data.sources })
                                            }
                                        } catch (err) { console.error(err) }
                                        setQueryLoading(false)
                                    }
                                }}
                                placeholder="What was discussed about Newton's laws?"
                                className="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
                            />
                            <button
                                onClick={async () => {
                                    if (!meetingQuery.trim()) return
                                    setQueryLoading(true)
                                    try {
                                        const res = await fetch(`${getApiBaseUrl().replace(':8000', ':8001')}/api/meetings/query`, {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ classroom_id: classroomId, query: meetingQuery })
                                        })
                                        if (res.ok) {
                                            const data = await res.json()
                                            setQueryAnswer({ answer: data.answer, sources: data.sources })
                                        }
                                    } catch (err) { console.error(err) }
                                    setQueryLoading(false)
                                }}
                                disabled={queryLoading}
                                className="px-4 py-2.5 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 disabled:opacity-50"
                            >
                                {queryLoading ? 'Searching...' : 'Ask'}
                            </button>
                        </div>
                        {queryAnswer && (
                            <div className="mt-4 p-4 bg-white rounded-xl border border-gray-200">
                                <p className="text-gray-800 whitespace-pre-wrap">{queryAnswer.answer}</p>
                                {queryAnswer.sources.length > 0 && (
                                    <div className="mt-3 pt-3 border-t border-gray-100">
                                        <p className="text-xs text-gray-500 mb-1">Sources:</p>
                                        {queryAnswer.sources.map((s, i) => (
                                            <p key={i} className="text-xs text-gray-400">"{s.text}"</p>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <button onClick={() => setShowMeetModal(true)} className="btn-primary flex items-center gap-2">
                        <PlusIcon className="w-5 h-5" />
                        Start New Meeting
                    </button>

                    {meetings.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <VideoCameraIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No meetings yet</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {meetings.map(m => (
                                <div key={m.id} className="card">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-4 flex-1">
                                            <div className={`p-3 rounded-xl ${m.status === 'live' ? 'bg-red-100' :
                                                m.status === 'ended' ? 'bg-green-100' : 'bg-gray-100'
                                                }`}>
                                                <VideoCameraIcon className={`w-6 h-6 ${m.status === 'live' ? 'text-red-600' :
                                                    m.status === 'ended' ? 'text-green-600' : 'text-gray-500'
                                                    }`} />
                                            </div>
                                            <div className="flex-1">
                                                <div className="flex items-center justify-between gap-2">
                                                    <p className="font-medium text-gray-900">{m.title}</p>
                                                    {m.scheduled_at && (
                                                        <div className="flex items-center gap-1 text-xs text-gray-400 shrink-0">
                                                            <CalendarIcon className="w-3.5 h-3.5" />
                                                            <span>{new Date(m.scheduled_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                                                        </div>
                                                    )}
                                                </div>
                                                <p className="text-sm text-gray-500">
                                                    {m.status === 'live' && <span className="text-red-600 font-medium">● Live Now</span>}
                                                    {m.status === 'scheduled' && `Starts: ${new Date(m.scheduled_at!).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}`}
                                                    {m.status === 'ended' && (
                                                        <span className="text-green-600">
                                                            Ended • {m.duration_minutes} min • Recording saved
                                                        </span>
                                                    )}
                                                </p>
                                            </div>
                                        </div>
                                        <div className="flex gap-2">
                                            {m.status === 'live' && (
                                                <>
                                                    <button onClick={() => router.push(`/meet/${m.id}`)} className="btn-primary text-sm flex items-center gap-1">
                                                        <PlayIcon className="w-4 h-4" /> Join
                                                    </button>
                                                    <button onClick={() => endMeeting(m.id)} className="px-3 py-2 bg-red-100 text-red-700 rounded-lg text-sm">
                                                        End
                                                    </button>
                                                </>
                                            )}
                                            {m.status === 'scheduled' && (
                                                <button onClick={async () => {
                                                    // Start the meeting via API
                                                    await fetch(`${getApiBaseUrl()}/api/meeting/${m.id}/start`, {
                                                        method: 'POST',
                                                        headers: {
                                                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                                                        }
                                                    })
                                                    setMeetings(meetings.map(mt => mt.id === m.id ? { ...mt, status: 'live' as const, started_at: new Date().toISOString() } : mt))
                                                    router.push(`/meet/${m.id}`)
                                                }} className="btn-primary text-sm">
                                                    Start Now
                                                </button>
                                            )}
                                        </div>
                                    </div>

                                    {/* Recording and Transcript Section for Ended Meetings */}
                                    {m.status === 'ended' && (
                                        <div className="mt-4 pt-4 border-t border-gray-100">
                                            <div className="grid md:grid-cols-2 gap-4">
                                                {/* Recording */}
                                                <div className="bg-gray-50 rounded-lg p-4">
                                                    <div className="flex items-center gap-2 mb-3">
                                                        <VideoCameraIcon className="w-5 h-5 text-primary-600" />
                                                        <span className="font-medium text-gray-900">Recording</span>
                                                    </div>
                                                    {m.recording_url ? (
                                                        <div>
                                                            <a
                                                                href={m.recording_url}
                                                                target="_blank"
                                                                className="text-sm text-primary-600 hover:underline flex items-center gap-1"
                                                            >
                                                                <PlayIcon className="w-4 h-4" />
                                                                Watch Recording
                                                            </a>
                                                            <p className="text-xs text-gray-500 mt-1">
                                                                Duration: {m.duration_minutes} minutes
                                                            </p>
                                                        </div>
                                                    ) : (
                                                        <p className="text-sm text-gray-500">Processing...</p>
                                                    )}
                                                </div>

                                                {/* Transcript */}
                                                <div className="bg-gray-50 rounded-lg p-4">
                                                    <div className="flex items-center gap-2 mb-3">
                                                        <DocumentTextIcon className="w-5 h-5 text-primary-600" />
                                                        <span className="font-medium text-gray-900">Transcript</span>
                                                    </div>
                                                    {m.transcript ? (
                                                        <div>
                                                            <p className="text-sm text-gray-600 line-clamp-3">{m.transcript}</p>
                                                            <button
                                                                onClick={() => {
                                                                    alert(m.transcript)
                                                                }}
                                                                className="text-sm text-primary-600 hover:underline mt-2"
                                                            >
                                                                View Full Transcript
                                                            </button>
                                                        </div>
                                                    ) : (
                                                        <p className="text-sm text-gray-500">Generating transcript...</p>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Assignments Tab */}
            {activeTab === 'assignments' && (
                <div className="space-y-4">
                    <button onClick={openAssignmentModal} className="btn-primary flex items-center gap-2">
                        <PlusIcon className="w-5 h-5" />
                        Create Assignment
                    </button>

                    {assignments.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <AcademicCapIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No assignments yet</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {assignments.map(assignment => (
                                <div key={assignment.id} className="card">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-3 mb-2">
                                                <div className="p-2 bg-primary-100 rounded-lg">
                                                    <AcademicCapIcon className="w-5 h-5 text-primary-600" />
                                                </div>
                                                <div>
                                                    <h3 className="font-medium text-gray-900">{assignment.title}</h3>
                                                    <div className="flex items-center gap-3 mt-1">
                                                        {assignment.points && (
                                                            <span className="text-sm text-gray-600 flex items-center gap-1">
                                                                {assignment.points} points
                                                            </span>
                                                        )}
                                                        {assignment.due_date && (
                                                            <span className="text-sm text-gray-600 flex items-center gap-1">
                                                                <CalendarIcon className="w-4 h-4" />
                                                                Due: {new Date(assignment.due_date).toLocaleDateString()}
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
                                                                href={att.url}
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

                                            {/* Submission count - clickable to view submissions */}
                                            <div className="ml-14 mt-3 flex items-center gap-2">
                                                <button
                                                    onClick={() => openSubmissionsModal(assignment)}
                                                    className="text-sm bg-primary-100 text-primary-700 px-3 py-1 rounded-full flex items-center gap-1 hover:bg-primary-200 transition-colors"
                                                >
                                                    <CheckCircleIcon className="w-4 h-4" />
                                                    {assignment.submission_count || 0} submission{(assignment.submission_count || 0) !== 1 ? 's' : ''} - View
                                                </button>
                                            </div>
                                        </div>

                                        <button
                                            onClick={() => deleteAssignment(assignment.id)}
                                            className="p-2 text-gray-400 hover:text-red-500"
                                        >
                                            <TrashIcon className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Students Tab */}
            {activeTab === 'students' && (
                <div>
                    {students.length === 0 ? (
                        <div className="text-center py-12 bg-gray-50 rounded-xl">
                            <UsersIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                            <p className="text-gray-500">No students yet</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {students.map(s => (
                                <div key={s.id} className="card-hover flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-medium">
                                            {(s.first_name?.[0] || s.username[0]).toUpperCase()}
                                        </div>
                                        <div>
                                            <p className="font-medium text-gray-900">{s.first_name && s.last_name ? `${s.first_name} ${s.last_name}` : s.username}</p>
                                            <p className="text-sm text-gray-500">{s.email}</p>
                                        </div>
                                    </div>
                                    <p className="text-sm text-gray-400">Joined {new Date(s.joined_at).toLocaleDateString()}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
                <div className="space-y-6">
                    {/* Accept Students Toggle */}
                    <div className="card">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="font-medium text-gray-900">Accept New Students</p>
                                <p className="text-sm text-gray-500">Allow joining with code</p>
                            </div>
                            <button onClick={toggleActive} className={`relative w-14 h-7 rounded-full ${classroom.is_active ? 'bg-green-500' : 'bg-gray-300'}`}>
                                <span className={`absolute top-1 w-5 h-5 rounded-full bg-white transition-transform ${classroom.is_active ? 'left-8' : 'left-1'}`}></span>
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Start Meeting Modal */}
            {showMeetModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <h2 className="text-xl font-bold text-gray-900 mb-4">New Meeting</h2>

                        {/* Meeting Title */}
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Meeting Title
                            </label>
                            <input
                                type="text"
                                value={meetTitle}
                                onChange={(e) => setMeetTitle(e.target.value)}
                                placeholder="e.g., Physics Class - Chapter 5"
                                className="input-field"
                            />
                        </div>

                        {/* Schedule Date/Time (Optional) */}
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Schedule For (Optional)
                            </label>
                            <input
                                type="datetime-local"
                                value={scheduledAt}
                                onChange={(e) => setScheduledAt(e.target.value)}
                                min={new Date().toISOString().slice(0, 16)}
                                className="input-field"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                                Leave empty to start immediately
                            </p>
                        </div>

                        {/* Recording Toggle */}
                        <div className="mb-6 flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <div>
                                <p className="font-medium text-gray-900">Enable Recording</p>
                                <p className="text-xs text-gray-500">Record meeting for later playback</p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setEnableRecording(prev => !prev)}
                                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer items-center rounded-full transition-colors duration-200 ${enableRecording ? 'bg-green-500' : 'bg-gray-300'
                                    }`}
                            >
                                <span
                                    style={{
                                        transform: enableRecording ? 'translateX(22px)' : 'translateX(2px)',
                                        transition: 'transform 200ms ease-in-out'
                                    }}
                                    className="inline-block h-5 w-5 rounded-full bg-white shadow-md"
                                />
                            </button>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex gap-3">
                            <button
                                onClick={() => {
                                    setShowMeetModal(false)
                                    setMeetTitle('')
                                    setScheduledAt('')
                                    setEnableRecording(true)
                                }}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={startMeeting}
                                className="flex-1 btn-primary"
                            >
                                {scheduledAt ? 'Schedule Meeting' : 'Start Now'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Upload Material Modal */}
            {showUploadModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
                        <h2 className="text-xl font-bold text-gray-900 mb-4">📁 Upload Materials</h2>

                        <div className="mb-4">
                            <p className="text-sm text-gray-600 mb-3">
                                {selectedFiles.length} file(s) selected:
                            </p>
                            <div className="bg-gray-50 rounded-lg p-3 max-h-32 overflow-y-auto">
                                {selectedFiles.map((file, i) => (
                                    <p key={i} className="text-sm text-gray-700 truncate">{file.name}</p>
                                ))}
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <button
                                onClick={() => {
                                    setShowUploadModal(false)
                                    setSelectedFiles([])
                                    setUploadSubject('')
                                }}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button onClick={confirmUpload} className="flex-1 btn-primary">
                                Upload Materials
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Syllabus PDF View Modal - Using PDFViewer */}
            {showSyllabusModal && classroom?.syllabus_url && (
                <div className="fixed inset-0 bg-black/90 z-50 flex flex-col">
                    <PDFViewer
                        pdfUrl={classroom.syllabus_url}
                        title={classroom.syllabus_filename || 'Class Syllabus'}
                        onClose={() => setShowSyllabusModal(false)}
                    />
                </div>
            )}

            {/* View Submissions Modal */}
            {showSubmissionsModal && viewingAssignment && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-gray-200">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Submissions</h2>
                                <p className="text-sm text-gray-500">{viewingAssignment.title} • {viewingAssignment.points || 0} points</p>
                            </div>
                            <button
                                onClick={() => {
                                    setShowSubmissionsModal(false)
                                    setViewingAssignment(null)
                                    setSelectedSubmission(null)
                                }}
                                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                            >
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="p-6">
                            {assignmentSubmissions.length === 0 ? (
                                <div className="text-center py-12 bg-gray-50 rounded-xl">
                                    <UsersIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                                    <p className="text-gray-500">No submissions yet</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {assignmentSubmissions.map(submission => (
                                        <div key={submission.id} className="border border-gray-200 rounded-xl p-4">
                                            <div className="flex items-start justify-between">
                                                {/* Student Info */}
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-medium">
                                                        {submission.student?.name?.[0]?.toUpperCase() || 'S'}
                                                    </div>
                                                    <div>
                                                        <p className="font-medium text-gray-900">{submission.student?.name || 'Student'}</p>
                                                        <p className="text-sm text-gray-500">{submission.student?.email}</p>
                                                        <p className="text-xs text-gray-400">
                                                            Submitted: {new Date(submission.submitted_at).toLocaleString()}
                                                        </p>
                                                    </div>
                                                </div>

                                                {/* Status Badge */}
                                                <div className="flex items-center gap-2">
                                                    {submission.status === 'graded' ? (
                                                        <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm font-medium">
                                                            Graded: {submission.grade}/{viewingAssignment.points}
                                                        </span>
                                                    ) : (
                                                        <span className="bg-yellow-100 text-yellow-700 px-3 py-1 rounded-full text-sm font-medium">
                                                            Pending
                                                        </span>
                                                    )}
                                                </div>
                                            </div>

                                            {/* Submitted Files */}
                                            {submission.files && submission.files.length > 0 && (
                                                <div className="mt-4 bg-gray-50 rounded-lg p-3">
                                                    <p className="text-sm font-medium text-gray-700 mb-2">Submitted Files:</p>
                                                    <div className="space-y-2">
                                                        {submission.files.map((file, idx) => (
                                                            <a
                                                                key={idx}
                                                                href={file.url}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="flex items-center gap-2 text-primary-600 hover:underline bg-white p-2 rounded"
                                                            >
                                                                {file.type === 'file' ? (
                                                                    <DocumentTextIcon className="w-4 h-4" />
                                                                ) : (
                                                                    <LinkIcon className="w-4 h-4" />
                                                                )}
                                                                <span className="text-sm">{file.filename}</span>
                                                            </a>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Feedback Display (if graded) */}
                                            {submission.feedback && (
                                                <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                                                    <p className="text-sm font-medium text-blue-700">Feedback:</p>
                                                    <p className="text-sm text-blue-600">{submission.feedback}</p>
                                                </div>
                                            )}

                                            {/* Grading Section */}
                                            {selectedSubmission?.id === submission.id ? (
                                                <div className="mt-4 p-4 bg-gray-100 rounded-lg">
                                                    <div className="grid grid-cols-2 gap-4">
                                                        <div>
                                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                                Grade (out of {viewingAssignment.points})
                                                            </label>
                                                            <input
                                                                type="number"
                                                                min="0"
                                                                max={viewingAssignment.points || 100}
                                                                value={gradeInput}
                                                                onChange={(e) => setGradeInput(e.target.value)}
                                                                className="input-field"
                                                                placeholder="Enter grade"
                                                            />
                                                        </div>
                                                        <div>
                                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                                Feedback (optional)
                                                            </label>
                                                            <input
                                                                type="text"
                                                                value={feedbackInput}
                                                                onChange={(e) => setFeedbackInput(e.target.value)}
                                                                className="input-field"
                                                                placeholder="Add feedback"
                                                            />
                                                        </div>
                                                    </div>
                                                    <div className="flex justify-end gap-2 mt-3">
                                                        <button
                                                            onClick={() => setSelectedSubmission(null)}
                                                            className="px-3 py-1.5 text-gray-600 hover:bg-gray-200 rounded-lg text-sm"
                                                        >
                                                            Cancel
                                                        </button>
                                                        <button
                                                            onClick={saveGrade}
                                                            disabled={savingGrade}
                                                            className="px-3 py-1.5 bg-green-600 text-white rounded-lg text-sm hover:bg-green-700 disabled:opacity-50"
                                                        >
                                                            {savingGrade ? 'Saving...' : 'Save Grade'}
                                                        </button>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="mt-4 flex justify-end">
                                                    <button
                                                        onClick={() => openGradeModal(submission)}
                                                        className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 flex items-center gap-2"
                                                    >
                                                        <AcademicCapIcon className="w-4 h-4" />
                                                        {submission.status === 'graded' ? 'Update Grade' : 'Grade'}
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Create Assignment Modal */}
            {showAssignmentModal && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-gray-200">
                            <div className="flex items-center gap-3">
                                <AcademicCapIcon className="w-6 h-6 text-primary-600" />
                                <h2 className="text-xl font-bold text-gray-900">Create Assignment</h2>
                            </div>
                            <button
                                onClick={() => setShowAssignmentModal(false)}
                                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                            >
                                ✕
                            </button>
                        </div>

                        {/* Form */}
                        <div className="p-6 space-y-4">
                            {/* Title */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Title *
                                </label>
                                <input
                                    type="text"
                                    value={assignmentTitle}
                                    onChange={(e) => setAssignmentTitle(e.target.value)}
                                    placeholder="Assignment title"
                                    className="input-field"
                                />
                            </div>

                            {/* Description */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Description
                                </label>
                                <textarea
                                    value={assignmentDescription}
                                    onChange={(e) => setAssignmentDescription(e.target.value)}
                                    placeholder="Assignment instructions..."
                                    className="input-field min-h-[120px] resize-none"
                                />
                            </div>

                            {/* Due Date & Points */}
                            <div className="grid md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Due Date
                                    </label>
                                    <input
                                        type="datetime-local"
                                        value={assignmentDueDate}
                                        onChange={(e) => setAssignmentDueDate(e.target.value)}
                                        className="input-field"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Points
                                    </label>
                                    <input
                                        type="number"
                                        value={assignmentPoints}
                                        onChange={(e) => setAssignmentPoints(e.target.value)}
                                        placeholder="100"
                                        className="input-field"
                                    />
                                </div>
                            </div>

                            {/* Attachments */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Attachments
                                </label>

                                {/* Upload PDF Button */}
                                <div className="flex gap-2 mb-3">
                                    <button
                                        onClick={() => assignmentFileRef.current?.click()}
                                        className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center gap-2"
                                    >
                                        <DocumentTextIcon className="w-4 h-4" />
                                        Upload PDF
                                    </button>
                                    <input
                                        ref={assignmentFileRef}
                                        type="file"
                                        accept=".pdf,application/pdf"
                                        multiple
                                        onChange={handleAssignmentFileUpload}
                                        className="hidden"
                                    />
                                </div>

                                {/* Add Link Input */}
                                <div className="flex gap-2 mb-3">
                                    <input
                                        type="url"
                                        value={linkInput}
                                        onChange={(e) => setLinkInput(e.target.value)}
                                        placeholder="https://example.com/resource"
                                        className="input-field flex-1"
                                        onKeyPress={(e) => e.key === 'Enter' && addLink()}
                                    />
                                    <button
                                        onClick={addLink}
                                        className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center gap-2"
                                    >
                                        <LinkIcon className="w-4 h-4" />
                                        Add Link
                                    </button>
                                </div>

                                {/* Attachment List */}
                                {assignmentAttachments.length > 0 && (
                                    <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                                        {assignmentAttachments.map((att, idx) => (
                                            <div key={idx} className="flex items-center justify-between bg-white p-2 rounded">
                                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                                    {att.type === 'file' ? (
                                                        <DocumentTextIcon className="w-4 h-4 text-primary-600 flex-shrink-0" />
                                                    ) : (
                                                        <LinkIcon className="w-4 h-4 text-primary-600 flex-shrink-0" />
                                                    )}
                                                    <span className="text-sm text-gray-700 truncate">{att.filename || att.url}</span>
                                                </div>
                                                <button
                                                    onClick={() => removeAttachment(idx)}
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

                        {/* Footer */}
                        <div className="p-6 border-t border-gray-100 flex justify-end gap-3">
                            <button
                                onClick={() => setShowAssignmentModal(false)}
                                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={saveAssignment}
                                disabled={savingAssignment || !assignmentTitle.trim()}
                                className="btn-primary flex items-center gap-2 disabled:opacity-50"
                            >
                                {savingAssignment ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                        Creating...
                                    </>
                                ) : (
                                    <>
                                        <PlusIcon className="w-4 h-4" />
                                        Create Assignment
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Document Viewer Modal */}
            {showDocumentViewer && viewingDocument && (
                <div className="fixed inset-0 bg-black/90 z-50 flex flex-col">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800">
                        <div className="flex items-center gap-3">
                            <DocumentTextIcon className="w-6 h-6 text-primary-400" />
                            <div>
                                <h2 className="text-white font-medium">{viewingDocument.name}</h2>
                                <p className="text-gray-400 text-sm">{formatSize(viewingDocument.size)} • {viewingDocument.subject}</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => {
                                    setShowDocumentViewer(false)
                                    setViewingDocument(null)
                                }}
                                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg"
                            >
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 overflow-hidden">
                        {viewingDocument.type.includes('pdf') ? (
                            <PDFViewer
                                pdfUrl={viewingDocument.url}
                                title={viewingDocument.name}
                                onClose={() => {
                                    setShowDocumentViewer(false)
                                    setViewingDocument(null)
                                }}
                            />
                        ) : viewingDocument.type.includes('image') ? (
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
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
