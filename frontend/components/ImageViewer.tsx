'use client'

import { useState } from 'react'
import {
    ArrowDownTrayIcon,
    XMarkIcon,
    ArrowTopRightOnSquareIcon,
    MagnifyingGlassPlusIcon,
    MagnifyingGlassMinusIcon,
    ArrowsPointingOutIcon,
    ExclamationTriangleIcon
} from '@heroicons/react/24/outline'

interface ImageViewerProps {
    imageUrl: string
    title?: string
    onClose?: () => void
}

export default function ImageViewer({ imageUrl, title, onClose }: ImageViewerProps) {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [scale, setScale] = useState(1)

    // Check if URL is valid
    const isValidUrl = imageUrl && imageUrl !== '#' && imageUrl.length > 1

    const handleDownload = () => {
        if (!isValidUrl) return
        const link = document.createElement('a')
        link.href = imageUrl
        link.download = title || 'image'
        link.target = '_blank'
        link.click()
    }

    const handleOpenInNewTab = () => {
        if (!isValidUrl) return
        window.open(imageUrl, '_blank')
    }

    const zoomIn = () => setScale(prev => Math.min(prev + 0.25, 3))
    const zoomOut = () => setScale(prev => Math.max(prev - 0.25, 0.5))
    const resetZoom = () => setScale(1)

    // Show placeholder for invalid URLs (mock data)
    if (!isValidUrl) {
        return (
            <div className="flex flex-col h-full bg-gray-900">
                {/* Toolbar */}
                <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
                    <h3 className="text-white font-medium truncate max-w-[300px]">
                        {title || 'Image'}
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
                        This is sample/demo data. In production, actual images
                        uploaded to your classroom will be displayed here.
                    </p>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg"
                    >
                        Close
                    </button>
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
                        {title || 'Image'}
                    </h3>
                </div>

                <div className="flex items-center gap-1">
                    {/* Zoom Controls */}
                    <button
                        onClick={zoomOut}
                        disabled={scale <= 0.5}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Zoom out"
                    >
                        <MagnifyingGlassMinusIcon className="w-5 h-5" />
                    </button>
                    <span className="text-gray-400 text-sm min-w-[50px] text-center">
                        {Math.round(scale * 100)}%
                    </span>
                    <button
                        onClick={zoomIn}
                        disabled={scale >= 3}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Zoom in"
                    >
                        <MagnifyingGlassPlusIcon className="w-5 h-5" />
                    </button>
                    <button
                        onClick={resetZoom}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                        title="Reset zoom"
                    >
                        <ArrowsPointingOutIcon className="w-5 h-5" />
                    </button>

                    <div className="w-px h-6 bg-gray-700 mx-2" />

                    {/* Open in New Tab */}
                    <button
                        onClick={handleOpenInNewTab}
                        className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5"
                        title="Open in new tab"
                    >
                        <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                        <span className="hidden sm:inline">Open</span>
                    </button>

                    {/* Download */}
                    <button
                        onClick={handleDownload}
                        className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded flex items-center gap-1.5"
                        title="Download"
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

            {/* Image Content */}
            <div className="flex-1 relative overflow-auto bg-gray-800 flex items-center justify-center p-8">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
                        <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
                            <p className="text-gray-400">Loading image...</p>
                        </div>
                    </div>
                )}

                {error ? (
                    <div className="text-center">
                        <p className="text-red-400 mb-4">{error}</p>
                        <button
                            onClick={handleOpenInNewTab}
                            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg flex items-center gap-2"
                        >
                            <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                            Open in New Tab
                        </button>
                    </div>
                ) : (
                    <img
                        src={imageUrl}
                        alt={title || 'Image'}
                        className="max-w-full max-h-full object-contain rounded-lg shadow-2xl transition-transform duration-200"
                        style={{ transform: `scale(${scale})` }}
                        onLoad={() => setLoading(false)}
                        onError={() => {
                            setLoading(false)
                            setError('Failed to load image')
                        }}
                    />
                )}
            </div>
        </div>
    )
}
