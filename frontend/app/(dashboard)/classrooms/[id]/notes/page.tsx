'use client'

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
    EyeIcon
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
}

const BACKEND_URL = 'http://localhost:8000'

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
                                            <div className="flex items-center gap-2">
                                                <button
                                                    onClick={() => viewPdf(job.id, job.title)}
                                                    className="flex items-center gap-2 px-3 py-2 bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition-colors"
                                                >
                                                    <EyeIcon className="w-5 h-5" />
                                                    View
                                                </button>
                                                <button
                                                    onClick={() => downloadPdf(job.id, job.title)}
                                                    className="flex items-center gap-2 px-3 py-2 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors"
                                                >
                                                    <ArrowDownTrayIcon className="w-5 h-5" />
                                                    Download
                                                </button>
                                            </div>
                                        ) : job.status === 'failed' ? (
                                            <p className="text-sm text-red-600">
                                                {job.error_message || 'Processing failed'}
                                            </p>
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
            </div>
        </div>
    )
}
