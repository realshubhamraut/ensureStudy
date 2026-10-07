'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useSession } from 'next-auth/react'
import Link from 'next/link'
import {
    ArrowLeftIcon,
    CloudArrowUpIcon,
    DocumentTextIcon,
    PhotoIcon,
    VideoCameraIcon,
    XMarkIcon,
    ArrowPathIcon,
    CheckCircleIcon,
    ExclamationTriangleIcon,
    ArrowDownTrayIcon,
    DocumentIcon,
    EyeIcon,
    TrashIcon,
    MagnifyingGlassIcon,
    DocumentMagnifyingGlassIcon
} from '@heroicons/react/24/outline'

interface NotesJob {
    id: string
    title: string
    description: string
    source_type: 'video' | 'images'
    status: 'pending' | 'processing' | 'completed' | 'failed'
    progress_percent: number
    current_step: string
    pdf_path: string | null
    created_at: string
    error_message: string
    total_pages?: number
    avg_confidence?: number
}

interface NotePage {
    id: string
    job_id: string
    page_number: number
    enhanced_image_url: string | null
    thumbnail_url: string | null
    extracted_text: string | null
    confidence_score: number | null
    status: string
}

const BACKEND_URL = `${getApiBaseUrl()}`

export default function DigitizeNotesPage() {
    const params = useParams()
    const router = useRouter()
    const { data: session } = useSession()
    const classroomId = params.id as string

    // State
    const [jobs, setJobs] = useState<NotesJob[]>([])
    const [loading, setLoading] = useState(true)
    const [uploading, setUploading] = useState(false)
    const [showUploadModal, setShowUploadModal] = useState(false)

    // PDF Viewer state
    const [viewingPdf, setViewingPdf] = useState<{ jobId: string, title: string } | null>(null)
    const [pdfBlobUrl, setPdfBlobUrl] = useState<string | null>(null)
    const [loadingPdf, setLoadingPdf] = useState(false)

    // Details modal state
    const [showDetailsModal, setShowDetailsModal] = useState(false)
    const [selectedJob, setSelectedJob] = useState<NotesJob | null>(null)
    const [jobPages, setJobPages] = useState<NotePage[]>([])
    const [selectedPage, setSelectedPage] = useState<NotePage | null>(null)
    const [loadingDetails, setLoadingDetails] = useState(false)

    // Search state
    const [searchQuery, setSearchQuery] = useState('')
    const [searchResults, setSearchResults] = useState<any[]>([])
    const [searching, setSearching] = useState(false)

    // Upload form
    const [uploadTitle, setUploadTitle] = useState('')
    const [uploadFiles, setUploadFiles] = useState<File[]>([])
    const fileInputRef = useRef<HTMLInputElement>(null)

    // Fetch jobs
    useEffect(() => {
        if (session?.accessToken) {
            fetchJobs()
        }
    }, [classroomId, session?.accessToken])

    // Poll for processing jobs
    useEffect(() => {
        const processingJobs = jobs.filter(j => j.status === 'processing' || j.status === 'pending')
        if (processingJobs.length === 0) return

        const interval = setInterval(() => {
            processingJobs.forEach(job => fetchJobStatus(job.id))
        }, 3000)

        return () => clearInterval(interval)
    }, [jobs])

    // Cleanup blob URL when modal closes
    useEffect(() => {
        if (!viewingPdf && pdfBlobUrl) {
            URL.revokeObjectURL(pdfBlobUrl)
            setPdfBlobUrl(null)
        }
    }, [viewingPdf])

    const fetchJobs = async () => {
        if (!session?.accessToken) return
        try {
            const res = await fetch(`${BACKEND_URL}/api/notes/jobs?classroom_id=${classroomId}`, {
                headers: { Authorization: `Bearer ${session.accessToken}` }
            })
            if (res.ok) {
                const data = await res.json()
                setJobs(data.jobs || [])
            }
        } catch (error) {
            console.error('Error fetching jobs:', error)
        } finally {
            setLoading(false)
        }
    }

    const fetchJobStatus = async (jobId: string) => {
        if (!session?.accessToken) return
        try {
            const res = await fetch(`${BACKEND_URL}/api/notes/jobs/${jobId}`, {
                headers: { Authorization: `Bearer ${session.accessToken}` }
            })
            if (res.ok) {
                const data = await res.json()
                setJobs(prev => prev.map(j => j.id === jobId ? data.job : j))
            }
        } catch (error) {
            console.error('Error fetching job status:', error)
        }
    }

    const handleUpload = async () => {
        if (!session?.accessToken || !uploadFiles.length || !uploadTitle) return

        setUploading(true)

        try {
            const formData = new FormData()
            formData.append('classroom_id', classroomId)
            formData.append('title', uploadTitle)
            uploadFiles.forEach(file => formData.append('file', file))

            // Use XMLHttpRequest for better large file handling
            const xhr = new XMLHttpRequest()

            const uploadPromise = new Promise<any>((resolve, reject) => {
                xhr.open('POST', `${BACKEND_URL}/api/notes/upload`, true)
                xhr.setRequestHeader('Authorization', `Bearer ${session.accessToken}`)
                xhr.timeout = 300000 // 5 minutes for large files

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const data = JSON.parse(xhr.responseText)
                            resolve(data)
                        } catch (e) {
                            reject(new Error('Invalid JSON response'))
                        }
                    } else {
                        try {
                            const error = JSON.parse(xhr.responseText)
                            reject(new Error(error.error || `Upload failed: ${xhr.status}`))
                        } catch (e) {
                            reject(new Error(`Upload failed: ${xhr.status}`))
                        }
                    }
                }

                xhr.onerror = () => reject(new Error('Network error during upload'))
                xhr.ontimeout = () => reject(new Error('Upload timed out - file may be too large'))

                xhr.send(formData)
            })

            const data = await uploadPromise
            setJobs(prev => [data.job, ...prev])
            setShowUploadModal(false)
            setUploadTitle('')
            setUploadFiles([])

        } catch (error: any) {
            console.error('Upload error:', error)
            alert(`Upload failed: ${error.message || 'Unknown error'}`)
        } finally {
            setUploading(false)
        }
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || [])
        setUploadFiles(files)

        // Auto-set title from first file name
        if (files.length > 0 && !uploadTitle) {
            const name = files[0].name.replace(/\.[^/.]+$/, '')
            setUploadTitle(name)
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed':
                return <CheckCircleIcon className="w-5 h-5 text-green-500" />
            case 'processing':
            case 'pending':
                return <ArrowPathIcon className="w-5 h-5 text-blue-500 animate-spin" />
            case 'failed':
                return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />
            default:
                return <DocumentIcon className="w-5 h-5 text-gray-400" />
        }
    }

    const downloadPdf = (jobId: string, title: string) => {
        const url = `${BACKEND_URL}/api/notes/pdf/${jobId}`
        fetch(url, {
            headers: { Authorization: `Bearer ${session?.accessToken}` }
        })
            .then(res => res.blob())
            .then(blob => {
                const blobUrl = URL.createObjectURL(blob)
                const link = document.createElement('a')
                link.href = blobUrl
                link.download = `${title}.pdf`
                link.click()
                URL.revokeObjectURL(blobUrl)
            })
    }

    const viewPdf = async (jobId: string, title: string) => {
        setViewingPdf({ jobId, title })
        setLoadingPdf(true)

        try {
            const url = `${BACKEND_URL}/api/notes/pdf/${jobId}`
            const res = await fetch(url, {
                headers: { Authorization: `Bearer ${session?.accessToken}` }
            })

            if (res.ok) {
                const blob = await res.blob()
                const blobUrl = URL.createObjectURL(blob)
                setPdfBlobUrl(blobUrl)
            } else {
                alert('Failed to load PDF')
                setViewingPdf(null)
            }
        } catch (error) {
            console.error('Error loading PDF:', error)
            alert('Failed to load PDF')
            setViewingPdf(null)
        } finally {
            setLoadingPdf(false)
        }
    }

    const closePdfViewer = () => {
        setViewingPdf(null)
        if (pdfBlobUrl) {
            URL.revokeObjectURL(pdfBlobUrl)
            setPdfBlobUrl(null)
        }
    }

    const deleteJob = async (jobId: string, title: string) => {
        if (!confirm(`Are you sure you want to delete "${title}"? This cannot be undone.`)) {
            return
        }

        try {
            const res = await fetch(`${BACKEND_URL}/api/notes/jobs/${jobId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${session?.accessToken}` }
            })

            if (res.ok) {
                // Remove from local state
                setJobs(prev => prev.filter(j => j.id !== jobId))
            } else {
                const error = await res.json()
                alert(`Failed to delete: ${error.error || 'Unknown error'}`)
            }
        } catch (error) {
            console.error('Error deleting job:', error)
            alert('Failed to delete notes')
        }
    }

    const viewDetails = async (job: NotesJob) => {
        setSelectedJob(job)
        setShowDetailsModal(true)
        setLoadingDetails(true)
        setJobPages([])
        setSelectedPage(null)

        try {
            // Fetch pages with text
            const res = await fetch(`${BACKEND_URL}/api/notes/pages/${job.id}?include_text=true`, {
                headers: { Authorization: `Bearer ${session?.accessToken}` }
            })

            if (res.ok) {
                const data = await res.json()
                setJobPages(data.pages || [])
                // Select first page by default
                if (data.pages && data.pages.length > 0) {
                    setSelectedPage(data.pages[0])
                }
            }
        } catch (error) {
            console.error('Error fetching pages:', error)
        } finally {
            setLoadingDetails(false)
        }
    }

    const searchNotes = async () => {
        if (!searchQuery.trim()) return

        setSearching(true)
        try {
            const res = await fetch(`${BACKEND_URL}/api/notes/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${session?.accessToken}`
                },
                body: JSON.stringify({
                    query: searchQuery,
                    classroom_id: classroomId,
                    limit: 10
                })
            })

            if (res.ok) {
                const data = await res.json()
                setSearchResults(data.results || [])
            }
        } catch (error) {
            console.error('Error searching notes:', error)
        } finally {
            setSearching(false)
        }
    }

    const getConfidenceColor = (score: number | null) => {
        if (score === null) return 'bg-gray-200 text-gray-600'
        if (score >= 0.8) return 'bg-green-100 text-green-700'
        if (score >= 0.5) return 'bg-yellow-100 text-yellow-700'
        return 'bg-red-100 text-red-700'
    }

    const getConfidenceLabel = (score: number | null) => {
        if (score === null) return 'N/A'
        return `${(score * 100).toFixed(0)}%`
    }

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <ArrowPathIcon className="w-8 h-8 animate-spin text-purple-600" />
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 p-6">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <Link
                            href={`/classrooms/${classroomId}`}
                            className="p-2 hover:bg-white/50 rounded-lg transition-colors"
                        >
                            <ArrowLeftIcon className="w-5 h-5" />
                        </Link>
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">
                                Digitize My Notes
                            </h1>
                            <p className="text-gray-500">
                                Convert handwritten notes to searchable PDF
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={() => setShowUploadModal(true)}
                        className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                    >
                        <CloudArrowUpIcon className="w-5 h-5" />
                        Upload Notes
                    </button>
                </div>

                {/* Empty State */}
                {jobs.length === 0 && (
                    <div className="bg-white rounded-2xl shadow-sm border p-12 text-center">
                        <DocumentTextIcon className="w-16 h-16 mx-auto text-gray-300 mb-4" />
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                            No notes yet
                        </h3>
                        <p className="text-gray-500 mb-6">
                            Upload a video or images of your handwritten notes to convert them to searchable PDF
                        </p>
                        <button
                            onClick={() => setShowUploadModal(true)}
                            className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                        >
                            <CloudArrowUpIcon className="w-5 h-5" />
                            Upload Your First Notes
                        </button>
                    </div>
                )}

                {/* Jobs List */}
                {jobs.length > 0 && (
                    <div className="space-y-4">
                        {jobs.map(job => (
                            <div
                                key={job.id}
                                className="bg-white rounded-xl shadow-sm border p-6 hover:shadow-md transition-shadow"
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex items-start gap-4">
                                        {getStatusIcon(job.status)}
                                        <div>
                                            <h3 className="font-semibold text-gray-900">
                                                {job.title}
                                            </h3>
                                            <p className="text-sm text-gray-500">
                                                {job.source_type === 'video' ? (
                                                    <span className="flex items-center gap-1">
                                                        <VideoCameraIcon className="w-4 h-4" />
                                                        Video
                                                    </span>
                                                ) : (
                                                    <span className="flex items-center gap-1">
                                                        <PhotoIcon className="w-4 h-4" />
                                                        Images
                                                    </span>
                                                )}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="text-right">
                                        {/* Status */}
                                        {job.status === 'processing' || job.status === 'pending' ? (
                                            <div>
                                                <p className="text-sm font-medium text-blue-600">
                                                    {job.current_step || 'Processing...'}
                                                </p>
                                                <div className="w-32 h-2 bg-gray-200 rounded-full mt-2">
                                                    <div
                                                        className="h-full bg-blue-500 rounded-full transition-all"
                                                        style={{ width: `${job.progress_percent}%` }}
                                                    />
                                                </div>
                                                <p className="text-xs text-gray-400 mt-1">
                                                    {job.progress_percent}%
                                                </p>
                                            </div>
                                        ) : job.status === 'completed' ? (
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <button
                                                    onClick={() => viewPdf(job.id, job.title)}
                                                    className="flex items-center gap-2 px-3 py-2 bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition-colors"
                                                >
                                                    <EyeIcon className="w-5 h-5" />
                                                    PDF
                                                </button>
                                                <button
                                                    onClick={() => viewDetails(job)}
                                                    className="flex items-center gap-2 px-3 py-2 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors"
                                                >
                                                    <DocumentMagnifyingGlassIcon className="w-5 h-5" />
                                                    Details
                                                </button>
                                                <button
                                                    onClick={() => downloadPdf(job.id, job.title)}
                                                    className="flex items-center gap-2 px-3 py-2 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors"
                                                >
                                                    <ArrowDownTrayIcon className="w-5 h-5" />
                                                    Download
                                                </button>
                                                <button
                                                    onClick={() => deleteJob(job.id, job.title)}
                                                    className="flex items-center gap-2 px-3 py-2 bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors"
                                                >
                                                    <TrashIcon className="w-5 h-5" />
                                                    Delete
                                                </button>
                                            </div>
                                        ) : job.status === 'failed' ? (
                                            <div className="flex items-center gap-2">
                                                <p className="text-sm text-red-600">
                                                    {job.error_message || 'Processing failed'}
                                                </p>
                                                <button
                                                    onClick={() => deleteJob(job.id, job.title)}
                                                    className="flex items-center gap-1 px-2 py-1 text-xs bg-red-50 text-red-700 rounded hover:bg-red-100 transition-colors"
                                                >
                                                    <TrashIcon className="w-4 h-4" />
                                                    Delete
                                                </button>
                                            </div>
                                        ) : null}
                                    </div>
                                </div>

                                {/* Date */}
                                <p className="text-xs text-gray-400 mt-4">
                                    {new Date(job.created_at).toLocaleDateString('en-US', {
                                        month: 'short',
                                        day: 'numeric',
                                        year: 'numeric',
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    })}
                                </p>
                            </div>
                        ))}
                    </div>
                )}

                {/* Upload Modal */}
                {showUploadModal && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
                        <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-xl font-bold">Upload Notes</h2>
                                <button
                                    onClick={() => setShowUploadModal(false)}
                                    className="p-2 hover:bg-gray-100 rounded-lg"
                                >
                                    <XMarkIcon className="w-5 h-5" />
                                </button>
                            </div>

                            {/* File Drop Zone */}
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-purple-400 hover:bg-purple-50/50 cursor-pointer transition-colors"
                            >
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="video/*,image/*"
                                    multiple
                                    onChange={handleFileSelect}
                                    className="hidden"
                                />
                                {uploadFiles.length > 0 ? (
                                    <div>
                                        <CheckCircleIcon className="w-12 h-12 mx-auto text-green-500 mb-2" />
                                        <p className="font-medium text-gray-900">
                                            {uploadFiles.length} file{uploadFiles.length > 1 ? 's' : ''} selected
                                        </p>
                                        <p className="text-sm text-gray-500">
                                            {uploadFiles.map(f => f.name).join(', ')}
                                        </p>
                                    </div>
                                ) : (
                                    <div>
                                        <CloudArrowUpIcon className="w-12 h-12 mx-auto text-gray-400 mb-2" />
                                        <p className="font-medium text-gray-900">
                                            Click to upload
                                        </p>
                                        <p className="text-sm text-gray-500">
                                            Video or images of your notes
                                        </p>
                                    </div>
                                )}
                            </div>

                            {/* Title Input */}
                            <div className="mt-4">
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Title
                                </label>
                                <input
                                    type="text"
                                    value={uploadTitle}
                                    onChange={e => setUploadTitle(e.target.value)}
                                    placeholder="e.g., Physics Lecture Notes"
                                    className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                />
                            </div>

                            {/* Submit Button */}
                            <button
                                onClick={handleUpload}
                                disabled={uploading || !uploadFiles.length || !uploadTitle}
                                className="w-full mt-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                                {uploading ? (
                                    <>
                                        <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                        Uploading...
                                    </>
                                ) : (
                                    <>
                                        <CloudArrowUpIcon className="w-5 h-5" />
                                        Upload & Convert to PDF
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                )}

                {/* PDF Viewer Modal */}
                {viewingPdf && (
                    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
                        <div className="bg-white rounded-2xl shadow-2xl w-full h-full max-w-6xl max-h-[90vh] m-4 flex flex-col overflow-hidden">
                            {/* Header */}
                            <div className="flex items-center justify-between p-4 border-b bg-gray-50">
                                <div className="flex items-center gap-3">
                                    <DocumentTextIcon className="w-6 h-6 text-purple-600" />
                                    <h2 className="text-lg font-semibold">{viewingPdf.title}</h2>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => downloadPdf(viewingPdf.jobId, viewingPdf.title)}
                                        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                                    >
                                        <ArrowDownTrayIcon className="w-5 h-5" />
                                        Download
                                    </button>
                                    <button
                                        onClick={closePdfViewer}
                                        className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                                    >
                                        <XMarkIcon className="w-6 h-6" />
                                    </button>
                                </div>
                            </div>

                            {/* PDF Content */}
                            <div className="flex-1 bg-gray-100">
                                {loadingPdf ? (
                                    <div className="w-full h-full flex items-center justify-center">
                                        <div className="text-center">
                                            <ArrowPathIcon className="w-12 h-12 animate-spin text-purple-600 mx-auto mb-4" />
                                            <p className="text-gray-600">Loading PDF...</p>
                                        </div>
                                    </div>
                                ) : pdfBlobUrl ? (
                                    <iframe
                                        src={pdfBlobUrl}
                                        className="w-full h-full"
                                        title={viewingPdf.title}
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center">
                                        <p className="text-gray-500">Failed to load PDF</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Details Modal */}
                {showDetailsModal && selectedJob && (
                    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
                        <div className="bg-white rounded-2xl shadow-2xl w-full h-full max-w-6xl max-h-[90vh] m-4 flex flex-col overflow-hidden">
                            {/* Header */}
                            <div className="flex items-center justify-between p-4 border-b bg-gray-50">
                                <div className="flex items-center gap-3">
                                    <DocumentMagnifyingGlassIcon className="w-6 h-6 text-blue-600" />
                                    <div>
                                        <h2 className="text-lg font-semibold">{selectedJob.title}</h2>
                                        <p className="text-sm text-gray-500">
                                            {jobPages.length} page{jobPages.length !== 1 ? 's' : ''} •
                                            {selectedJob.avg_confidence ? ` ${(selectedJob.avg_confidence * 100).toFixed(0)}% avg confidence` : ' Processing...'}
                                        </p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setShowDetailsModal(false)}
                                    className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                                >
                                    <XMarkIcon className="w-6 h-6" />
                                </button>
                            </div>

                            {loadingDetails ? (
                                <div className="flex-1 flex items-center justify-center">
                                    <div className="text-center">
                                        <ArrowPathIcon className="w-12 h-12 animate-spin text-blue-600 mx-auto mb-4" />
                                        <p className="text-gray-600">Loading pages...</p>
                                    </div>
                                </div>
                            ) : jobPages.length === 0 ? (
                                /* Fallback: Show PDF side-by-side view when no pages in DB */
                                <div className="flex-1 flex overflow-hidden">
                                    {/* Left: PDF View */}
                                    <div className="flex-1 border-r bg-gray-100">
                                        <div className="p-2 bg-white border-b flex items-center gap-2">
                                            <DocumentTextIcon className="w-5 h-5 text-purple-600" />
                                            <span className="font-medium text-sm">PDF Document</span>
                                        </div>
                                        <iframe
                                            src={`${BACKEND_URL}/api/notes/pdf/${selectedJob.id}?token=${session?.accessToken}`}
                                            className="w-full h-full"
                                            title="Notes PDF"
                                        />
                                    </div>

                                    {/* Right: Info & Search */}
                                    <div className="w-96 flex flex-col bg-white">
                                        <div className="p-4 border-b">
                                            <h3 className="font-semibold text-gray-900">Extracted Text</h3>
                                            <p className="text-sm text-gray-500 mt-1">
                                                OCR text will appear here after processing completes.
                                            </p>
                                        </div>
                                        <div className="flex-1 p-4 overflow-y-auto">
                                            <div className="text-center py-12 text-gray-400">
                                                <DocumentTextIcon className="w-12 h-12 mx-auto mb-3" />
                                                <p>Pages are being processed...</p>
                                                <p className="text-sm mt-2">OCR text extraction in progress.</p>
                                                <button
                                                    onClick={() => viewDetails(selectedJob)}
                                                    className="mt-4 px-4 py-2 text-sm bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100"
                                                >
                                                    <ArrowPathIcon className="w-4 h-4 inline mr-2" />
                                                    Refresh
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex-1 flex overflow-hidden">
                                    {/* Left: Page thumbnails */}
                                    <div className="w-48 border-r bg-gray-50 overflow-y-auto p-2 space-y-2">
                                        {jobPages.map((page, index) => (
                                            <button
                                                key={page.id}
                                                onClick={() => setSelectedPage(page)}
                                                className={`w-full p-2 rounded-lg text-left transition-all ${selectedPage?.id === page.id
                                                    ? 'bg-blue-100 ring-2 ring-blue-500'
                                                    : 'bg-white hover:bg-gray-100'
                                                    }`}
                                            >
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="text-sm font-medium">Page {page.page_number}</span>
                                                    <span className={`text-xs px-2 py-0.5 rounded-full ${getConfidenceColor(page.confidence_score)}`}>
                                                        {getConfidenceLabel(page.confidence_score)}
                                                    </span>
                                                </div>
                                                {page.enhanced_image_url && (
                                                    <img
                                                        src={`${BACKEND_URL}${page.enhanced_image_url}?token=${session?.accessToken}`}
                                                        alt={`Page ${page.page_number}`}
                                                        className="w-full h-24 object-cover rounded border"
                                                        onError={(e) => {
                                                            (e.target as HTMLImageElement).src = '/placeholder-page.png'
                                                        }}
                                                    />
                                                )}
                                            </button>
                                        ))}
                                    </div>

                                    {/* Right: Selected page content */}
                                    <div className="flex-1 flex flex-col overflow-hidden">
                                        {selectedPage ? (
                                            <>
                                                {/* Page header */}
                                                <div className="p-4 border-b bg-white flex items-center justify-between">
                                                    <div>
                                                        <h3 className="font-semibold">Page {selectedPage.page_number}</h3>
                                                        <div className="flex items-center gap-2 mt-1">
                                                            <span className={`text-sm px-2 py-0.5 rounded-full ${getConfidenceColor(selectedPage.confidence_score)}`}>
                                                                OCR Confidence: {getConfidenceLabel(selectedPage.confidence_score)}
                                                            </span>
                                                            <span className="text-sm text-gray-500">
                                                                Status: {selectedPage.status}
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Page content: Image + Text */}
                                                <div className="flex-1 overflow-y-auto p-4 grid grid-cols-2 gap-4">
                                                    {/* Image */}
                                                    <div className="bg-gray-100 rounded-lg p-2">
                                                        <h4 className="text-sm font-medium text-gray-600 mb-2">Enhanced Image</h4>
                                                        {selectedPage.enhanced_image_url ? (
                                                            <img
                                                                src={`${BACKEND_URL}${selectedPage.enhanced_image_url}?token=${session?.accessToken}`}
                                                                alt={`Page ${selectedPage.page_number}`}
                                                                className="w-full rounded border shadow-sm"
                                                            />
                                                        ) : (
                                                            <div className="h-64 flex items-center justify-center bg-gray-200 rounded">
                                                                <p className="text-gray-500">No image available</p>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {/* Extracted Text */}
                                                    <div className="bg-white border rounded-lg p-4">
                                                        <h4 className="text-sm font-medium text-gray-600 mb-2 flex items-center gap-2">
                                                            <DocumentTextIcon className="w-4 h-4" />
                                                            Extracted Text (OCR)
                                                        </h4>
                                                        {selectedPage.extracted_text ? (
                                                            <pre className="text-sm whitespace-pre-wrap font-mono bg-gray-50 p-3 rounded max-h-96 overflow-y-auto">
                                                                {selectedPage.extracted_text}
                                                            </pre>
                                                        ) : (
                                                            <div className="text-center py-8 text-gray-500">
                                                                <p>No text extracted yet</p>
                                                                <p className="text-xs mt-1">OCR may still be processing</p>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </>
                                        ) : (
                                            <div className="flex-1 flex items-center justify-center text-gray-500">
                                                Select a page to view details
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Search bar at bottom */}
                            <div className="p-4 border-t bg-gray-50">
                                <div className="flex gap-2">
                                    <div className="flex-1 relative">
                                        <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                                        <input
                                            type="text"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && searchNotes()}
                                            placeholder="Search within these notes..."
                                            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                    <button
                                        onClick={searchNotes}
                                        disabled={searching || !searchQuery.trim()}
                                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                                    >
                                        {searching ? (
                                            <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                        ) : (
                                            <MagnifyingGlassIcon className="w-5 h-5" />
                                        )}
                                        Search
                                    </button>
                                </div>

                                {/* Search results */}
                                {searchResults.length > 0 && (
                                    <div className="mt-3 max-h-40 overflow-y-auto">
                                        <p className="text-sm text-gray-500 mb-2">{searchResults.length} results found:</p>
                                        <div className="space-y-2">
                                            {searchResults.map((result, i) => (
                                                <div key={i} className="p-2 bg-white rounded border text-sm">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="font-medium">Page {result.page_number || '?'}</span>
                                                        <span className="text-xs text-gray-500">Score: {(result.score * 100).toFixed(0)}%</span>
                                                    </div>
                                                    <p className="text-gray-600 line-clamp-2">{result.chunk_text || result.text}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
