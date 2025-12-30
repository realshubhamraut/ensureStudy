'use client'

import { useState, useEffect, useRef } from 'react'
import {
    ArrowDownTrayIcon,
    XMarkIcon,
    ArrowTopRightOnSquareIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    MagnifyingGlassMinusIcon,
    MagnifyingGlassPlusIcon
} from '@heroicons/react/24/outline'

interface BoundingBox {
    x1: number
    y1: number
    x2: number
    y2: number
    pageNumber: number
}

interface PDFViewerWithHighlightProps {
    pdfUrl: string
    title?: string
    onClose?: () => void
    highlights?: BoundingBox[]
    initialPage?: number
}

export default function PDFViewerWithHighlight({
    pdfUrl,
    title,
    onClose,
    highlights = [],
    initialPage = 1
}: PDFViewerWithHighlightProps) {
    const [currentPage, setCurrentPage] = useState(initialPage)
    const [numPages, setNumPages] = useState(0)
    const [scale, setScale] = useState(1)
    const [loading, setLoading] = useState(true)
    const containerRef = useRef<HTMLDivElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)

    // Check if URL is valid
    const isValidUrl = pdfUrl && pdfUrl !== '#' && pdfUrl.length > 1

    // Get highlights for current page
    const currentPageHighlights = highlights.filter(h => h.pageNumber === currentPage)

    const handleDownload = () => {
        if (!isValidUrl) return
        const link = document.createElement('a')
        link.href = pdfUrl
        link.download = title ? `${title}.pdf` : 'document.pdf'
        link.target = '_blank'
        link.click()
    }

    const handleOpenInNewTab = () => {
        if (!isValidUrl) return
        window.open(pdfUrl, '_blank')
    }

    const goToPage = (page: number) => {
        if (page >= 1 && page <= numPages) {
            setCurrentPage(page)
        }
    }

    const zoomIn = () => setScale(prev => Math.min(prev + 0.25, 3))
    const zoomOut = () => setScale(prev => Math.max(prev - 0.25, 0.5))

    // Render highlights overlay
    const renderHighlights = () => {
        if (currentPageHighlights.length === 0) return null

        return (
            <div className="absolute inset-0 pointer-events-none">
                {currentPageHighlights.map((box, index) => {
                    // Convert bbox coordinates to percentages
                    // Assuming bbox is in pixels for a 612x792 PDF page (US Letter)
                    const pageWidth = 612 // Standard PDF width
                    const pageHeight = 792 // Standard PDF height

                    const left = (box.x1 / pageWidth) * 100
                    const top = (box.y1 / pageHeight) * 100
                    const width = ((box.x2 - box.x1) / pageWidth) * 100
                    const height = ((box.y2 - box.y1) / pageHeight) * 100

                    return (
                        <div
                            key={index}
                            className="absolute bg-yellow-400/30 border-2 border-yellow-500 rounded"
                            style={{
                                left: `${left}%`,
                                top: `${top}%`,
                                width: `${width}%`,
                                height: `${height}%`,
                                animation: 'pulse 2s infinite'
                            }}
                        />
                    )
                })}
            </div>
        )
    }

    // Show placeholder for invalid URLs
    if (!isValidUrl) {
        return (
            <div className="flex flex-col h-full bg-gray-900">
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
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-gray-800">
                    <p className="text-gray-400 mb-6">
                        No PDF available to display.
                    </p>
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg"
                        >
                            Close
                        </button>
                    )}
                </div>
            </div>
        )
    }

    return (
        <div className="flex flex-col h-full bg-gray-900">
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
                <div className="flex items-center gap-4">
                    <h3 className="text-white font-medium truncate max-w-[200px]">
                        {title || 'Document'}
                    </h3>

                    {/* Page Navigation */}
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => goToPage(currentPage - 1)}
                            disabled={currentPage <= 1}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <ChevronLeftIcon className="w-4 h-4" />
                        </button>
                        <span className="text-sm text-gray-300">
                            Page {currentPage} {numPages > 0 ? `of ${numPages}` : ''}
                        </span>
                        <button
                            onClick={() => goToPage(currentPage + 1)}
                            disabled={currentPage >= numPages}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <ChevronRightIcon className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Zoom Controls */}
                    <div className="flex items-center gap-1 border-l border-gray-600 pl-3">
                        <button
                            onClick={zoomOut}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        >
                            <MagnifyingGlassMinusIcon className="w-4 h-4" />
                        </button>
                        <span className="text-sm text-gray-300 w-12 text-center">
                            {Math.round(scale * 100)}%
                        </span>
                        <button
                            onClick={zoomIn}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        >
                            <MagnifyingGlassPlusIcon className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={handleOpenInNewTab}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        title="Open in new tab"
                    >
                        <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                    </button>
                    <button
                        onClick={handleDownload}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        title="Download"
                    >
                        <ArrowDownTrayIcon className="w-5 h-5" />
                    </button>
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        >
                            <XMarkIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>
            </div>

            {/* Highlight indicator */}
            {currentPageHighlights.length > 0 && (
                <div className="px-4 py-2 bg-yellow-900/30 border-b border-yellow-700">
                    <p className="text-sm text-yellow-300">
                        {currentPageHighlights.length} highlighted region{currentPageHighlights.length > 1 ? 's' : ''} on this page
                    </p>
                </div>
            )}

            {/* PDF Content */}
            <div
                ref={containerRef}
                className="flex-1 overflow-auto bg-gray-700 p-4"
            >
                <div
                    className="relative mx-auto bg-white shadow-xl"
                    style={{
                        transform: `scale(${scale})`,
                        transformOrigin: 'top center',
                        transition: 'transform 0.2s ease'
                    }}
                >
                    {/* PDF iframe with page parameter */}
                    <iframe
                        src={`${pdfUrl}#page=${currentPage}`}
                        className="w-full min-h-[800px]"
                        style={{ border: 'none' }}
                        onLoad={() => setLoading(false)}
                    />

                    {/* Highlight Overlay */}
                    {renderHighlights()}
                </div>
            </div>

            {/* Page quick jump for highlights */}
            {highlights.length > 0 && (
                <div className="px-4 py-2 bg-gray-800 border-t border-gray-700">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm text-gray-400">Jump to:</span>
                        {[...new Set(highlights.map(h => h.pageNumber))].sort((a, b) => a - b).map(page => (
                            <button
                                key={page}
                                onClick={() => goToPage(page)}
                                className={`px-2 py-1 text-xs rounded ${currentPage === page
                                        ? 'bg-purple-600 text-white'
                                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                    }`}
                            >
                                Page {page}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            <style jsx>{`
                @keyframes pulse {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 0.9; }
                }
            `}</style>
        </div>
    )
}
