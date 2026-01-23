'use client'

import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'

interface MindmapViewerProps {
    code: string
    title?: string
}

export function MindmapViewer({ code, title }: MindmapViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null)
    const [error, setError] = useState<string | null>(null)
    const [zoom, setZoom] = useState(225)

    // Pan/drag state
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [isDragging, setIsDragging] = useState(false)
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

    useEffect(() => {
        if (!containerRef.current || !code) return

        // Initialize mermaid with theme
        mermaid.initialize({
            startOnLoad: false,
            theme: 'default',
            securityLevel: 'loose',
            mindmap: {
                padding: 20,
                useMaxWidth: false
            }
        })

        // Render the diagram
        const renderDiagram = async () => {
            try {
                const id = `mindmap-${Date.now()}`
                const { svg } = await mermaid.render(id, code)
                if (containerRef.current) {
                    containerRef.current.innerHTML = svg
                }
                setError(null)
            } catch (err) {
                console.error('Mermaid render error:', err)
                setError('Failed to render flowchart')
            }
        }

        renderDiagram()
    }, [code])

    // Handle mouse drag for panning
    const handleMouseDown = (e: React.MouseEvent) => {
        if (zoom > 100) {
            setIsDragging(true)
            setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
        }
    }

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging && zoom > 100) {
            setPan({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y
            })
        }
    }

    const handleMouseUp = () => setIsDragging(false)
    const handleMouseLeave = () => setIsDragging(false)

    const resetView = () => {
        setZoom(100)
        setPan({ x: 0, y: 0 })
    }

    return (
        <div className="flex-1 flex flex-col bg-gradient-to-br from-slate-50 to-slate-100 overflow-hidden relative">
            {/* Header */}
            {title && (
                <div className="p-3 border-b border-slate-200 bg-white">
                    <h3 className="text-sm font-medium text-slate-700">{title}</h3>
                </div>
            )}

            {/* Mindmap Container - Draggable */}
            <div
                className={`flex-1 overflow-hidden p-6 flex items-center justify-center ${zoom > 100 ? 'cursor-grab' : ''} ${isDragging ? 'cursor-grabbing' : ''}`}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
            >
                {error ? (
                    <div className="text-red-500 text-sm bg-red-50 p-4 rounded-lg">
                        {error}
                    </div>
                ) : (
                    <div
                        ref={containerRef}
                        className={`select-none ${!isDragging ? 'transition-transform duration-200' : ''}`}
                        style={{
                            transform: `scale(${zoom / 100}) translate(${pan.x / (zoom / 100)}px, ${pan.y / (zoom / 100)}px)`,
                            willChange: 'transform',
                            backfaceVisibility: 'hidden'
                        }}
                    />
                )}
            </div>

            {/* Zoom Controls */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-white rounded-full px-4 py-2 shadow-lg border border-slate-200">
                <button
                    onClick={() => setZoom(z => Math.max(25, z - 25))}
                    className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-100 text-slate-600 transition-colors"
                    title="Zoom out"
                >
                    <span className="text-xl font-bold">−</span>
                </button>
                <span className="text-slate-600 text-sm font-medium w-14 text-center">{zoom}%</span>
                <button
                    onClick={() => setZoom(z => Math.min(400, z + 25))}
                    className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-100 text-slate-600 transition-colors"
                    title="Zoom in"
                >
                    <span className="text-xl font-bold">+</span>
                </button>
                <div className="w-px h-6 bg-slate-200 mx-1" />
                <button
                    onClick={resetView}
                    className="px-3 py-1 text-slate-600 text-sm hover:bg-slate-100 rounded-full transition-colors"
                    title="Reset zoom and pan"
                >
                    Reset
                </button>
            </div>
        </div>
    )
}

export default MindmapViewer
