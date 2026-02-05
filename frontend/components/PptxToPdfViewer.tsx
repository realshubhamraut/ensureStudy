'use client'

import { useState, useEffect } from 'react'
import { getAiServiceUrl } from '@/utils/api'
import PDFViewer from './PDFViewer'
import { DocumentTextIcon, ArrowPathIcon, ExclamationCircleIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline'

interface PptxToPdfViewerProps {
    pptxUrl: string
    title: string
    onClose: () => void
}

/**
 * Component that converts PPTX to PDF on-demand and displays it.
 * Shows loading state during conversion and error state if conversion fails.
 */
export default function PptxToPdfViewer({ pptxUrl, title, onClose }: PptxToPdfViewerProps) {
    const [status, setStatus] = useState<'converting' | 'ready' | 'error'>('converting')
    const [pdfUrl, setPdfUrl] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const convertToPdf = async () => {
            try {
                setStatus('converting')

                const response = await fetch(
                    `${getAiServiceUrl()}/api/convert/pptx-to-pdf?pptx_url=${encodeURIComponent(pptxUrl)}`,
                    { method: 'POST' }
                )

                const data = await response.json()

                if (data.success && data.pdf_url) {
                    // Construct full URL
                    const fullPdfUrl = data.pdf_url.startsWith('http')
                        ? data.pdf_url
                        : `${getAiServiceUrl()}${data.pdf_url}`
                    setPdfUrl(fullPdfUrl)
                    setStatus('ready')
                } else {
                    setError(data.error || 'Conversion failed')
                    setStatus('error')
                }
            } catch (err) {
                console.error('[PptxToPdfViewer] Conversion error:', err)
                setError('Failed to connect to conversion service')
                setStatus('error')
            }
        }

        convertToPdf()
    }, [pptxUrl])

    // Loading state - converting
    if (status === 'converting') {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-gradient-to-br from-orange-900 to-amber-900">
                <div className="bg-gradient-to-br from-orange-500 to-amber-600 p-8 rounded-3xl mb-8 shadow-2xl animate-pulse">
                    <DocumentTextIcon className="w-20 h-20 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">{title}</h3>
                <div className="flex items-center gap-3 text-orange-300 mb-6">
                    <ArrowPathIcon className="w-6 h-6 animate-spin" />
                    <span className="text-lg">Converting to PDF for viewing...</span>
                </div>
                <p className="text-gray-400 text-sm max-w-md">
                    This may take a few seconds depending on the presentation size.
                </p>
                <button
                    onClick={onClose}
                    className="mt-8 px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-medium"
                >
                    Cancel
                </button>
            </div>
        )
    }

    // Error state
    if (status === 'error') {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-gradient-to-br from-red-900 to-orange-900">
                <div className="bg-gradient-to-br from-red-500 to-orange-600 p-8 rounded-3xl mb-8 shadow-2xl">
                    <ExclamationCircleIcon className="w-20 h-20 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">{title}</h3>
                <p className="text-red-300 mb-4 text-lg">Conversion Failed</p>
                <p className="text-gray-400 text-sm mb-8 max-w-md">
                    {error || 'Unable to convert presentation to PDF.'}
                    <br />You can still download and view it locally.
                </p>
                <div className="flex gap-4">
                    <a
                        href={pptxUrl}
                        download
                        className="px-6 py-3 bg-gradient-to-r from-orange-600 to-amber-600 hover:from-orange-700 hover:to-amber-700 text-white rounded-xl flex items-center gap-2 font-medium shadow-lg"
                    >
                        <ArrowDownTrayIcon className="w-5 h-5" />
                        Download PPTX
                    </a>
                    <button
                        onClick={onClose}
                        className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-medium"
                    >
                        Close
                    </button>
                </div>
            </div>
        )
    }

    // Ready state - show PDF viewer
    if (status === 'ready' && pdfUrl) {
        return (
            <PDFViewer
                pdfUrl={pdfUrl}
                title={title + ' (Converted)'}
                onClose={onClose}
            />
        )
    }

    return null
}
