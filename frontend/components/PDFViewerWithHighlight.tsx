'use client'

import { getApiBaseUrl } from '@/utils/api'

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
    onLoad?: () => void
    onError?: (error: string) => void
}

// Helper to resolve PDF URL (handles relative paths from API)
// Files are served by Core Service (port 8000), not AI Service (port 8001)
function resolvePdfUrl(url: string): string {
    if (!url) return ''
    // Already a full URL
    if (url.startsWith('http://') || url.startsWith('https://')) {
        return url
    }
    // Relative path from API (like /api/files/web/...)
    if (url.startsWith('/api/files/')) {
        return `${getApiBaseUrl()}${url}`
    }
    // Other relative paths
    if (url.startsWith('/')) {
        return `${getApiBaseUrl()}${url}`
    }
    return url
}

export default function PDFViewerWithHighlight({
    pdfUrl,
    title,
    onClose,
    highlights = [],
    initialPage = 1,
    onLoad,
    onError
}: PDFViewerWithHighlightProps) {
    // Resolve the PDF URL (handles relative paths from API)
    const resolvedUrl = resolvePdfUrl(pdfUrl)

    // Check if URL is valid
    const isValidUrl = resolvedUrl && resolvedUrl !== '#' && resolvedUrl.length > 1

    // Show placeholder for invalid URLs
    if (!isValidUrl) {
        return (
            <div className="flex flex-col h-full items-center justify-center bg-gray-100 p-8">
                <p className="text-gray-500 mb-4">No PDF available to display.</p>
                {onClose && (
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg"
                    >
                        Close
                    </button>
                )}
            </div>
        )
    }

    // Clean, minimal PDF viewer - just the iframe with browser's native controls
    return (
        <iframe
            src={`${resolvedUrl}#page=${initialPage}`}
            className="w-full h-full border-0"
            title={title || 'PDF Document'}
            onLoad={() => onLoad?.()}
            onError={() => onError?.('Failed to load PDF')}
        />
    )
}
