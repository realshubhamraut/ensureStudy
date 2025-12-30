'use client'

import { useState, useEffect } from 'react'
import {
    DocumentTextIcon,
    ChevronRightIcon,
    XMarkIcon,
    MagnifyingGlassIcon,
    DocumentMagnifyingGlassIcon
} from '@heroicons/react/24/outline'

interface SidebarMatch {
    chunk_id: string
    page_number: number
    bbox?: number[]
    text_snippet: string
    similarity: number
    ocr_confidence?: number
}

interface DocumentSidebarProps {
    docId: string
    classId: string
    query?: string
    onClose?: () => void
    onPageSelect?: (pageNumber: number, bbox?: number[]) => void
}

interface SidebarData {
    doc_id: string
    title: string
    top_matches: SidebarMatch[]
    preview_summary: string
    pdf_url: string
    version: number
}

export default function DocumentSidebar({
    docId,
    classId,
    query,
    onClose,
    onPageSelect
}: DocumentSidebarProps) {
    const [data, setData] = useState<SidebarData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [searchQuery, setSearchQuery] = useState(query || '')
    const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)

    const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

    // Fetch sidebar data
    useEffect(() => {
        fetchSidebarData()
    }, [docId, query])

    const fetchSidebarData = async (customQuery?: string) => {
        setLoading(true)
        setError(null)

        try {
            const q = customQuery || searchQuery || query || ''
            const url = `${AI_SERVICE_URL}/api/ai-tutor/documents/${docId}/sidebar?query=${encodeURIComponent(q)}&top_k=5`

            const response = await fetch(url)
            if (!response.ok) {
                throw new Error(`Failed to fetch: ${response.status}`)
            }

            const result = await response.json()
            setData(result)
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load document')
        } finally {
            setLoading(false)
        }
    }

    const handleSearch = () => {
        fetchSidebarData(searchQuery)
    }

    const handleMatchClick = (match: SidebarMatch) => {
        setSelectedChunkId(match.chunk_id)
        if (onPageSelect) {
            onPageSelect(match.page_number, match.bbox)
        }
    }

    const getSimilarityColor = (similarity: number) => {
        if (similarity >= 0.8) return 'text-green-400'
        if (similarity >= 0.6) return 'text-yellow-400'
        return 'text-gray-400'
    }

    if (loading) {
        return (
            <div className="w-80 bg-gray-900 border-l border-gray-700 flex flex-col">
                <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                    <span className="text-white font-medium">Document Context</span>
                    {onClose && (
                        <button onClick={onClose} className="text-gray-400 hover:text-white">
                            <XMarkIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>
                <div className="flex-1 flex items-center justify-center">
                    <div className="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full" />
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="w-80 bg-gray-900 border-l border-gray-700 flex flex-col">
                <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                    <span className="text-white font-medium">Document Context</span>
                    {onClose && (
                        <button onClick={onClose} className="text-gray-400 hover:text-white">
                            <XMarkIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>
                <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
                    <DocumentMagnifyingGlassIcon className="w-12 h-12 text-gray-500 mb-3" />
                    <p className="text-gray-400 text-sm">{error}</p>
                    <button
                        onClick={() => fetchSidebarData()}
                        className="mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm"
                    >
                        Retry
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className="w-80 bg-gray-900 border-l border-gray-700 flex flex-col h-full">
            {/* Header */}
            <div className="p-4 border-b border-gray-700">
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <DocumentTextIcon className="w-5 h-5 text-purple-400" />
                        <span className="text-white font-medium truncate max-w-[180px]">
                            {data?.title || 'Document'}
                        </span>
                    </div>
                    {onClose && (
                        <button onClick={onClose} className="text-gray-400 hover:text-white">
                            <XMarkIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>

                {/* Search */}
                <div className="flex gap-2">
                    <div className="relative flex-1">
                        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            placeholder="Search in document..."
                            className="w-full pl-9 pr-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
                        />
                    </div>
                    <button
                        onClick={handleSearch}
                        className="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm"
                    >
                        Search
                    </button>
                </div>
            </div>

            {/* Summary */}
            {data?.preview_summary && (
                <div className="px-4 py-3 bg-gray-800/50 border-b border-gray-700">
                    <p className="text-sm text-gray-300">{data.preview_summary}</p>
                </div>
            )}

            {/* Matches */}
            <div className="flex-1 overflow-y-auto">
                {data?.top_matches && data.top_matches.length > 0 ? (
                    <div className="p-2 space-y-2">
                        {data.top_matches.map((match, index) => (
                            <button
                                key={match.chunk_id || index}
                                onClick={() => handleMatchClick(match)}
                                className={`w-full text-left p-3 rounded-lg transition-colors ${selectedChunkId === match.chunk_id
                                        ? 'bg-purple-600/30 border border-purple-500'
                                        : 'bg-gray-800 hover:bg-gray-700 border border-transparent'
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs text-purple-400 font-medium">
                                        Page {match.page_number}
                                    </span>
                                    <span className={`text-xs ${getSimilarityColor(match.similarity)}`}>
                                        {Math.round(match.similarity * 100)}% match
                                    </span>
                                </div>
                                <p className="text-sm text-gray-300 line-clamp-3">
                                    {match.text_snippet}
                                </p>
                                <div className="flex items-center justify-end mt-2 text-xs text-gray-500">
                                    <span>Click to view</span>
                                    <ChevronRightIcon className="w-3 h-3 ml-1" />
                                </div>
                            </button>
                        ))}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full p-4 text-center">
                        <DocumentMagnifyingGlassIcon className="w-12 h-12 text-gray-500 mb-3" />
                        <p className="text-gray-400 text-sm">
                            {searchQuery
                                ? 'No matches found for your search'
                                : 'Search to find relevant content'}
                        </p>
                    </div>
                )}
            </div>

            {/* PDF Link */}
            {data?.pdf_url && (
                <div className="p-3 border-t border-gray-700">
                    <a
                        href={data.pdf_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-center gap-2 w-full py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm transition-colors"
                    >
                        <DocumentTextIcon className="w-4 h-4" />
                        Open Full PDF
                    </a>
                </div>
            )}
        </div>
    )
}
