'use client'

import { useState } from 'react'
import DocumentSidebar from './DocumentSidebar'
import PDFViewerWithHighlight from './PDFViewerWithHighlight'
import {
    DocumentTextIcon,
    XMarkIcon,
    ChevronDoubleRightIcon,
    ChevronDoubleLeftIcon
} from '@heroicons/react/24/outline'

interface DocumentSource {
    document_id: string
    title: string
    page_number?: number
    similarity?: number
    preview_text?: string
    bbox?: number[]
}

interface DocumentContextPanelProps {
    classId: string
    sources: DocumentSource[]
    query?: string
    onClose?: () => void
    isOpen?: boolean
}

interface BoundingBox {
    x1: number
    y1: number
    x2: number
    y2: number
    pageNumber: number
}

export default function DocumentContextPanel({
    classId,
    sources,
    query,
    onClose,
    isOpen = true
}: DocumentContextPanelProps) {
    const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
    const [pdfUrl, setPdfUrl] = useState<string>('')
    const [highlights, setHighlights] = useState<BoundingBox[]>([])
    const [currentPage, setCurrentPage] = useState(1)
    const [showSidebar, setShowSidebar] = useState(true)
    const [expanded, setExpanded] = useState(false)

    const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

    const handleDocumentSelect = async (docId: string) => {
        setSelectedDocId(docId)

        // Fetch document details to get PDF URL
        try {
            const response = await fetch(`${AI_SERVICE_URL}/api/ai-tutor/documents/${docId}/sidebar?top_k=1`)
            if (response.ok) {
                const data = await response.json()
                setPdfUrl(data.pdf_url || '')
            }
        } catch (error) {
            console.error('Failed to fetch document:', error)
        }
    }

    const handlePageSelect = (pageNumber: number, bbox?: number[]) => {
        setCurrentPage(pageNumber)

        if (bbox && bbox.length === 4) {
            setHighlights([{
                x1: bbox[0],
                y1: bbox[1],
                x2: bbox[2],
                y2: bbox[3],
                pageNumber
            }])
        }
    }

    if (!isOpen) return null

    // Unique documents from sources
    const uniqueDocs = sources.reduce((acc, source) => {
        if (!acc.find(d => d.document_id === source.document_id)) {
            acc.push(source)
        }
        return acc
    }, [] as DocumentSource[])

    return (
        <div className={`fixed right-0 top-0 h-full bg-gray-900 shadow-2xl transition-all duration-300 z-50 flex ${expanded ? 'w-[80vw]' : 'w-96'
            }`}>
            {/* Document List / Sidebar */}
            {showSidebar && (
                <div className={`${selectedDocId && expanded ? 'w-80' : 'w-full'} flex flex-col border-r border-gray-700`}>
                    {/* Header */}
                    <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <DocumentTextIcon className="w-5 h-5 text-purple-400" />
                            <h3 className="text-white font-semibold">Document Sources</h3>
                        </div>
                        <div className="flex items-center gap-1">
                            {selectedDocId && (
                                <button
                                    onClick={() => setExpanded(!expanded)}
                                    className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                                    title={expanded ? 'Collapse' : 'Expand'}
                                >
                                    {expanded ? (
                                        <ChevronDoubleLeftIcon className="w-4 h-4" />
                                    ) : (
                                        <ChevronDoubleRightIcon className="w-4 h-4" />
                                    )}
                                </button>
                            )}
                            {onClose && (
                                <button
                                    onClick={onClose}
                                    className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                                >
                                    <XMarkIcon className="w-5 h-5" />
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Document List */}
                    {!selectedDocId ? (
                        <div className="flex-1 overflow-y-auto p-3 space-y-2">
                            {uniqueDocs.length === 0 ? (
                                <div className="text-center py-8">
                                    <DocumentTextIcon className="w-12 h-12 text-gray-500 mx-auto mb-3" />
                                    <p className="text-gray-400 text-sm">No document sources found</p>
                                </div>
                            ) : (
                                uniqueDocs.map((doc, index) => (
                                    <button
                                        key={doc.document_id || index}
                                        onClick={() => handleDocumentSelect(doc.document_id)}
                                        className="w-full text-left p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
                                    >
                                        <div className="flex items-start justify-between">
                                            <span className="text-white font-medium line-clamp-1">
                                                {doc.title || 'Untitled Document'}
                                            </span>
                                            {doc.similarity && (
                                                <span className={`text-xs ml-2 ${doc.similarity >= 0.8 ? 'text-green-400' :
                                                        doc.similarity >= 0.6 ? 'text-yellow-400' : 'text-gray-400'
                                                    }`}>
                                                    {Math.round(doc.similarity * 100)}%
                                                </span>
                                            )}
                                        </div>
                                        {doc.preview_text && (
                                            <p className="text-sm text-gray-400 mt-1 line-clamp-2">
                                                {doc.preview_text}
                                            </p>
                                        )}
                                        {doc.page_number && (
                                            <span className="text-xs text-purple-400 mt-2 block">
                                                Page {doc.page_number}
                                            </span>
                                        )}
                                    </button>
                                ))
                            )}
                        </div>
                    ) : (
                        <DocumentSidebar
                            docId={selectedDocId}
                            classId={classId}
                            query={query}
                            onClose={() => {
                                setSelectedDocId(null)
                                setHighlights([])
                                setExpanded(false)
                            }}
                            onPageSelect={handlePageSelect}
                        />
                    )}
                </div>
            )}

            {/* PDF Viewer */}
            {selectedDocId && expanded && pdfUrl && (
                <div className="flex-1">
                    <PDFViewerWithHighlight
                        pdfUrl={pdfUrl}
                        title={uniqueDocs.find(d => d.document_id === selectedDocId)?.title}
                        highlights={highlights}
                        initialPage={currentPage}
                    />
                </div>
            )}
        </div>
    )
}
