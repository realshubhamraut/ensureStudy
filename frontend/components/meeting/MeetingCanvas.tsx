'use client'

import { useRef, useState, useEffect, useCallback } from 'react'
import {
    PencilIcon,
    TrashIcon,
    ArrowUturnLeftIcon,
    ArrowUturnRightIcon,
    XMarkIcon,
    ArrowsPointingOutIcon,
    ArrowsPointingInIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Point {
    x: number
    y: number
}

interface DrawStroke {
    points: Point[]
    color: string
    width: number
    tool: 'pen' | 'eraser'
}

interface MeetingCanvasProps {
    isHost: boolean
    onClose: () => void
    isFullscreen?: boolean
    onToggleFullscreen?: () => void
}

const COLORS = [
    '#000000', // Black
    '#EF4444', // Red
    '#F97316', // Orange
    '#EAB308', // Yellow
    '#22C55E', // Green
    '#3B82F6', // Blue
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#FFFFFF', // White
]

const BRUSH_SIZES = [2, 4, 8, 12, 20]

/**
 * Collaborative canvas/whiteboard for meetings
 * Supports drawing, erasing, and real-time sync (when connected)
 */
export function MeetingCanvas({
    isHost,
    onClose,
    isFullscreen = false,
    onToggleFullscreen
}: MeetingCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [tool, setTool] = useState<'pen' | 'eraser'>('pen')
    const [color, setColor] = useState('#000000')
    const [brushSize, setBrushSize] = useState(4)
    const [history, setHistory] = useState<DrawStroke[]>([])
    const [historyIndex, setHistoryIndex] = useState(-1)
    const currentStrokeRef = useRef<DrawStroke | null>(null)
    const lastPointRef = useRef<Point | null>(null)

    // Initialize canvas
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Set canvas size
        const container = containerRef.current
        if (container) {
            canvas.width = container.clientWidth
            canvas.height = container.clientHeight
        }

        // Set white background
        ctx.fillStyle = '#FFFFFF'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    }, [isFullscreen])

    // Resize canvas on window resize
    useEffect(() => {
        const handleResize = () => {
            const canvas = canvasRef.current
            const container = containerRef.current
            if (!canvas || !container) return

            // Save current image
            const ctx = canvas.getContext('2d')
            if (!ctx) return
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

            // Resize
            canvas.width = container.clientWidth
            canvas.height = container.clientHeight

            // Restore image
            ctx.fillStyle = '#FFFFFF'
            ctx.fillRect(0, 0, canvas.width, canvas.height)
            ctx.putImageData(imageData, 0, 0)
        }

        window.addEventListener('resize', handleResize)
        return () => window.removeEventListener('resize', handleResize)
    }, [])

    // Get point from mouse/touch event
    const getPoint = useCallback((e: React.MouseEvent | React.TouchEvent): Point => {
        const canvas = canvasRef.current
        if (!canvas) return { x: 0, y: 0 }

        const rect = canvas.getBoundingClientRect()
        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height

        if ('touches' in e) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            }
        }

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        }
    }, [])

    // Draw a stroke segment
    const drawSegment = useCallback((from: Point, to: Point, strokeColor: string, strokeWidth: number, strokeTool: 'pen' | 'eraser') => {
        const canvas = canvasRef.current
        const ctx = canvas?.getContext('2d')
        if (!ctx) return

        ctx.beginPath()
        ctx.moveTo(from.x, from.y)
        ctx.lineTo(to.x, to.y)
        ctx.strokeStyle = strokeTool === 'eraser' ? '#FFFFFF' : strokeColor
        ctx.lineWidth = strokeTool === 'eraser' ? strokeWidth * 3 : strokeWidth
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.stroke()
    }, [])

    // Start drawing
    const handlePointerDown = useCallback((e: React.MouseEvent | React.TouchEvent) => {
        e.preventDefault()
        const point = getPoint(e)

        setIsDrawing(true)
        lastPointRef.current = point
        currentStrokeRef.current = {
            points: [point],
            color,
            width: brushSize,
            tool
        }
    }, [getPoint, color, brushSize, tool])

    // Continue drawing
    const handlePointerMove = useCallback((e: React.MouseEvent | React.TouchEvent) => {
        if (!isDrawing || !lastPointRef.current || !currentStrokeRef.current) return
        e.preventDefault()

        const point = getPoint(e)
        drawSegment(lastPointRef.current, point, color, brushSize, tool)

        currentStrokeRef.current.points.push(point)
        lastPointRef.current = point
    }, [isDrawing, getPoint, drawSegment, color, brushSize, tool])

    // End drawing
    const handlePointerUp = useCallback(() => {
        if (!isDrawing || !currentStrokeRef.current) return

        // Add to history
        const newHistory = history.slice(0, historyIndex + 1)
        newHistory.push(currentStrokeRef.current)
        setHistory(newHistory)
        setHistoryIndex(newHistory.length - 1)

        setIsDrawing(false)
        lastPointRef.current = null
        currentStrokeRef.current = null
    }, [isDrawing, history, historyIndex])

    // Redraw canvas from history
    const redrawCanvas = useCallback((upToIndex: number) => {
        const canvas = canvasRef.current
        const ctx = canvas?.getContext('2d')
        if (!ctx || !canvas) return

        // Clear canvas
        ctx.fillStyle = '#FFFFFF'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        // Redraw all strokes up to index
        history.slice(0, upToIndex + 1).forEach(stroke => {
            if (stroke.points.length < 2) return

            for (let i = 1; i < stroke.points.length; i++) {
                drawSegment(
                    stroke.points[i - 1],
                    stroke.points[i],
                    stroke.color,
                    stroke.width,
                    stroke.tool
                )
            }
        })
    }, [history, drawSegment])

    // Undo
    const handleUndo = useCallback(() => {
        if (historyIndex < 0) return
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)
        redrawCanvas(newIndex)
    }, [historyIndex, redrawCanvas])

    // Redo
    const handleRedo = useCallback(() => {
        if (historyIndex >= history.length - 1) return
        const newIndex = historyIndex + 1
        setHistoryIndex(newIndex)
        redrawCanvas(newIndex)
    }, [historyIndex, history.length, redrawCanvas])

    // Clear canvas
    const handleClear = useCallback(() => {
        const canvas = canvasRef.current
        const ctx = canvas?.getContext('2d')
        if (!ctx || !canvas) return

        ctx.fillStyle = '#FFFFFF'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        setHistory([])
        setHistoryIndex(-1)
    }, [])

    return (
        <div
            ref={containerRef}
            className={clsx(
                'flex flex-col bg-gray-900 rounded-xl overflow-hidden',
                isFullscreen ? 'fixed inset-0 z-50' : 'relative w-full h-full min-h-[300px]'
            )}
            style={!isFullscreen ? { minHeight: '300px' } : undefined}
        >
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
                <div className="flex items-center gap-4">
                    {/* Tools */}
                    <div className="flex items-center gap-1 bg-gray-700 rounded-lg p-1">
                        <button
                            onClick={() => setTool('pen')}
                            className={clsx(
                                'p-2 rounded-md transition-colors',
                                tool === 'pen' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                            )}
                            title="Pen"
                        >
                            <PencilIcon className="w-5 h-5" />
                        </button>
                        <button
                            onClick={() => setTool('eraser')}
                            className={clsx(
                                'p-2 rounded-md transition-colors',
                                tool === 'eraser' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                            )}
                            title="Eraser"
                        >
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M16.24 3.56l4.95 4.94c.78.79.78 2.05 0 2.84L12 20.53a4.008 4.008 0 01-5.66 0L2.81 17c-.78-.79-.78-2.05 0-2.84l10.6-10.6c.79-.78 2.05-.78 2.83 0zM4.22 15.58l3.54 3.53c.78.79 2.04.79 2.83 0L16 13.7 10.3 8 4.22 14.07c-.39.39-.39 1.02 0 1.41z" />
                            </svg>
                        </button>
                    </div>

                    {/* Colors */}
                    <div className="flex items-center gap-1">
                        {COLORS.map((c) => (
                            <button
                                key={c}
                                onClick={() => setColor(c)}
                                className={clsx(
                                    'w-6 h-6 rounded-full border-2 transition-transform',
                                    color === c ? 'border-white scale-110' : 'border-gray-600 hover:scale-105',
                                    c === '#FFFFFF' && 'ring-1 ring-gray-500'
                                )}
                                style={{ backgroundColor: c }}
                                title={c}
                            />
                        ))}
                    </div>

                    {/* Brush sizes */}
                    <div className="flex items-center gap-1">
                        {BRUSH_SIZES.map((size) => (
                            <button
                                key={size}
                                onClick={() => setBrushSize(size)}
                                className={clsx(
                                    'w-8 h-8 rounded-md flex items-center justify-center transition-colors',
                                    brushSize === size ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                                )}
                                title={`${size}px`}
                            >
                                <div
                                    className="rounded-full bg-white"
                                    style={{ width: Math.min(size, 16), height: Math.min(size, 16) }}
                                />
                            </button>
                        ))}
                    </div>

                    {/* History controls */}
                    <div className="flex items-center gap-1 bg-gray-700 rounded-lg p-1">
                        <button
                            onClick={handleUndo}
                            disabled={historyIndex < 0}
                            className="p-2 rounded-md text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Undo"
                        >
                            <ArrowUturnLeftIcon className="w-5 h-5" />
                        </button>
                        <button
                            onClick={handleRedo}
                            disabled={historyIndex >= history.length - 1}
                            className="p-2 rounded-md text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Redo"
                        >
                            <ArrowUturnRightIcon className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Clear button (host only) */}
                    {isHost && (
                        <button
                            onClick={handleClear}
                            className="p-2 rounded-md text-red-400 hover:text-red-300 hover:bg-gray-700"
                            title="Clear canvas"
                        >
                            <TrashIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    {/* Fullscreen toggle */}
                    {onToggleFullscreen && (
                        <button
                            onClick={onToggleFullscreen}
                            className="p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700"
                            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
                        >
                            {isFullscreen ? (
                                <ArrowsPointingInIcon className="w-5 h-5" />
                            ) : (
                                <ArrowsPointingOutIcon className="w-5 h-5" />
                            )}
                        </button>
                    )}

                    {/* Close button */}
                    <button
                        onClick={onClose}
                        className="p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700"
                        title="Close canvas"
                    >
                        <XMarkIcon className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Canvas Drawing Area */}
            <div className="flex-1 relative bg-white cursor-crosshair min-h-[250px]">
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full"
                    onMouseDown={handlePointerDown}
                    onMouseMove={handlePointerMove}
                    onMouseUp={handlePointerUp}
                    onMouseLeave={handlePointerUp}
                    onTouchStart={handlePointerDown}
                    onTouchMove={handlePointerMove}
                    onTouchEnd={handlePointerUp}
                />
            </div>
        </div>
    )
}
