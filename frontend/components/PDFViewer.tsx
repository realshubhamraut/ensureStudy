'use client'

import { useState, useMemo } from 'react'
import {
    ArrowDownTrayIcon,
    XMarkIcon,
    ArrowTopRightOnSquareIcon,
    ExclamationTriangleIcon
} from '@heroicons/react/24/outline'

interface PDFViewerProps {
    pdfUrl: string
    title?: string
    onClose?: () => void
}

export default function PDFViewer({ pdfUrl, title, onClose }: PDFViewerProps) {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    // Fix URL if it points to wrong port - dynamically use current API host
    const correctedUrl = useMemo(() => {
        if (!pdfUrl) return pdfUrl
        // Replace localhost:8000 or localhost:5000 with the correct API URL
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || `http://${window.location.hostname}:9000`
        return pdfUrl.replace(/http:\/\/localhost:(8000|5000)/g, apiUrl)
    }, [pdfUrl])

    // Check if URL is valid
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

    // Show placeholder for invalid URLs (mock data)
    if (!isValidUrl) {
        return (
            <div className="flex flex-col h-full bg-gray-900">
                {/* Toolbar */}
                <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
                    <h3 className="text-white font-medium truncate max-w-[300px]">
                        {title || 'Document'}
                    </h3>
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        >
                            <XMarkIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>

                {/* Placeholder for demo/mock data */}
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-gray-800">
                    <ExclamationTriangleIcon className="w-16 h-16 text-yellow-500 mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Demo Mode</h3>
                    <p className="text-gray-400 mb-6 max-w-md">
                        This is sample/demo data. In production, actual PDF documents
                        uploaded to your classroom will be displayed here.
                    </p>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg"
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="flex flex-col h-full bg-gray-900">
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
                <div className="flex items-center gap-4">
                    <h3 className="text-white font-medium truncate max-w-[300px]">
                        {title || 'Document'}
                    </h3>
                </div>

                <div className="flex items-center gap-2">
                    {/* Open in New Tab */}
                    <button
                        onClick={handleOpenInNewTab}
                        className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5"
                        title="Open in new tab"
                    >
                        <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                        <span className="hidden sm:inline">Open in Tab</span>
                    </button>

                    {/* Download */}
                    <button
                        onClick={handleDownload}
                        className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5"
                        title="Download PDF"
                    >
                        <ArrowDownTrayIcon className="w-4 h-4" />
                        <span className="hidden sm:inline">Download</span>
                    </button>

                    {onClose && (
                        <>
                            <div className="w-px h-6 bg-gray-700 mx-1" />
                            <button
                                onClick={onClose}
                                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                            >
                                <XMarkIcon className="w-5 h-5" />
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* PDF Content - Using native browser PDF viewer */}
            <div className="flex-1 relative bg-gray-700">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
                        <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
                            <p className="text-gray-400">Loading document...</p>
                        </div>
                    </div>
                )}

                {/* Use object tag for native PDF rendering - works better than iframe */}
                <object
                    data={correctedUrl}
                    type="application/pdf"
                    className="w-full h-full"
                    onLoad={() => setLoading(false)}
                    onError={() => {
                        setLoading(false)
                        setError('Failed to load PDF')
                    }}
                >
                    {/* Fallback for browsers that don't support object PDF */}
                    <iframe
                        src={correctedUrl}
                        className="w-full h-full border-0"
                        onLoad={() => setLoading(false)}
                        title={title || 'PDF Document'}
                    />
                </object>

                {error && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800">
                        <p className="text-red-400 mb-4">{error}</p>
                        <button
                            onClick={handleOpenInNewTab}
                            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg flex items-center gap-2"
                        >
                            <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                            Open in New Tab
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}
