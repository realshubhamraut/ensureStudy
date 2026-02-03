'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { getApiBaseUrl } from '@/utils/api'
import {
    BookOpenIcon,
    FunnelIcon
} from '@heroicons/react/24/outline'

// Dynamic imports for components
const TopicsSidebar = dynamic(() => import('@/components/curriculum/TopicsSidebar'), { ssr: false })
const StudyCalendar = dynamic(() => import('@/components/curriculum/StudyCalendar'), { ssr: false })

// ============================================================================
// Types
// ============================================================================

interface ClassroomOption {
    id: string
    name: string
    subject: string
}

// ============================================================================
// Main Component  
// ============================================================================

export default function CurriculumPage() {
    const [mounted, setMounted] = useState(false)
    const [classrooms, setClassrooms] = useState<ClassroomOption[]>([])
    const [selectedClassroomIds, setSelectedClassroomIds] = useState<Set<string>>(new Set()) // Empty = All

    // Resizable panel state
    const [sidebarWidth, setSidebarWidth] = useState(340) // Default width in pixels
    const containerRef = useRef<HTMLDivElement>(null)
    const isResizing = useRef(false)
    const startX = useRef(0)
    const startWidth = useRef(0)

    // Resize handlers
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        isResizing.current = true
        startX.current = e.clientX
        startWidth.current = sidebarWidth
        document.body.style.cursor = 'col-resize'
        document.body.style.userSelect = 'none'
    }, [sidebarWidth])

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!isResizing.current) return
        const delta = startX.current - e.clientX // Flip sign since moving left increases sidebar
        const newWidth = Math.max(250, Math.min(600, startWidth.current + delta))
        setSidebarWidth(newWidth)
    }, [])

    const handleMouseUp = useCallback(() => {
        isResizing.current = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
    }, [])

    // Add global mouse listeners for resize
    useEffect(() => {
        document.addEventListener('mousemove', handleMouseMove)
        document.addEventListener('mouseup', handleMouseUp)
        return () => {
            document.removeEventListener('mousemove', handleMouseMove)
            document.removeEventListener('mouseup', handleMouseUp)
        }
    }, [handleMouseMove, handleMouseUp])

    useEffect(() => {
        setMounted(true)
        fetchClassrooms()
    }, [])

    const fetchClassrooms = async () => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/curriculum/enrolled-topics`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                const options: ClassroomOption[] = (data.classrooms || []).map((c: any) => ({
                    id: c.classroom_id,
                    name: c.classroom_name,
                    subject: c.subject || c.classroom_name
                }))
                setClassrooms(options)
            }
        } catch (error) {
            console.error('Failed to fetch classrooms:', error)
        }
    }

    const toggleClassroom = (classroomId: string) => {
        setSelectedClassroomIds(prev => {
            const newSet = new Set(prev)
            if (newSet.has(classroomId)) {
                newSet.delete(classroomId)
            } else {
                newSet.add(classroomId)
            }
            return newSet
        })
    }

    const selectAll = () => {
        setSelectedClassroomIds(new Set()) // Empty set means all
    }

    const isAllSelected = selectedClassroomIds.size === 0

    // Get selected classroom names for calendar filtering
    const selectedClassroomNames = isAllSelected
        ? []
        : classrooms.filter(c => selectedClassroomIds.has(c.id)).map(c => c.name)

    // Prevent hydration mismatch
    if (!mounted) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
            </div>
        )
    }

    return (
        <div className="h-[calc(100vh-80px)] p-4">
            {/* Page Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
                        <BookOpenIcon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-gray-900">Study Planner</h1>
                        <p className="text-sm text-gray-500">Plan your learning schedule</p>
                    </div>
                </div>

                {/* Subject Filters */}
                <div className="flex items-center gap-2">
                    <FunnelIcon className="w-4 h-4 text-gray-400" />
                    <div className="flex flex-wrap gap-2">
                        {/* All Button */}
                        <button
                            onClick={selectAll}
                            className={`
                                px-3 py-1.5 text-sm font-medium rounded-full transition-all
                                ${isAllSelected
                                    ? 'bg-indigo-600 text-white shadow-md'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }
                            `}
                        >
                            All
                        </button>

                        {/* Subject/Classroom Buttons */}
                        {classrooms.map(classroom => {
                            const isSelected = selectedClassroomIds.has(classroom.id)
                            return (
                                <button
                                    key={classroom.id}
                                    onClick={() => toggleClassroom(classroom.id)}
                                    className={`
                                        px-3 py-1.5 text-sm font-medium rounded-full transition-all
                                        ${isSelected
                                            ? 'bg-indigo-600 text-white shadow-md'
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                        }
                                    `}
                                >
                                    {classroom.name}
                                </button>
                            )
                        })}
                    </div>
                </div>
            </div>

            {/* Two Panel Layout - Resizable */}
            <div ref={containerRef} className="flex gap-0 h-[calc(100%-60px)]">
                {/* Left: Calendar - takes remaining space */}
                <div className="flex-1 min-w-[300px]">
                    <StudyCalendar
                        selectedClassroomIds={Array.from(selectedClassroomIds)}
                        selectedClassroomNames={selectedClassroomNames}
                    />
                </div>

                {/* Resizable Divider */}
                <div
                    onMouseDown={handleMouseDown}
                    className="w-1.5 bg-gray-200 hover:bg-indigo-400 active:bg-indigo-500 cursor-col-resize transition-colors flex-shrink-0 rounded-full mx-1.5 group relative"
                >
                    {/* Drag handle indicator */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-8 flex flex-col items-center justify-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                        <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                        <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                    </div>
                </div>

                {/* Right: Topics Sidebar - resizable width */}
                <div style={{ width: sidebarWidth }} className="flex-shrink-0 min-w-[250px] max-w-[600px]">
                    <TopicsSidebar selectedClassroomIds={Array.from(selectedClassroomIds)} />
                </div>
            </div>
        </div>
    )
}
