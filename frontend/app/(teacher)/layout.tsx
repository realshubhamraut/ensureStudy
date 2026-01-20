'use client'

import { useSession, signOut } from 'next-auth/react'
import { useRouter, usePathname } from 'next/navigation'
import Link from 'next/link'
import { useEffect, useState, useCallback, useRef } from 'react'
import {
    UsersIcon,
    AcademicCapIcon,
    DocumentTextIcon,
    Cog6ToothIcon,
    ArrowRightOnRectangleIcon,
    Bars3Icon,
    XMarkIcon,
    ClipboardDocumentCheckIcon,
    SparklesIcon,
    ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

const navigation = [
    { name: 'AI Assistant', href: '/teacher/dashboard', icon: SparklesIcon },
    { name: 'Classrooms', href: '/teacher/classrooms', icon: AcademicCapIcon },
    { name: 'My Students', href: '/teacher/students', icon: UsersIcon },
    { name: 'Interact', href: '/teacher/interact', icon: ChatBubbleLeftRightIcon },
    { name: 'Exam Evaluations', href: '/teacher/scan', icon: ClipboardDocumentCheckIcon },
    { name: 'Assessments', href: '/teacher/assessments', icon: DocumentTextIcon },
    { name: 'Settings', href: '/teacher/settings', icon: Cog6ToothIcon },
]

export default function TeacherLayout({
    children,
}: {
    children: React.ReactNode
}) {
    const { data: session, status } = useSession()
    const router = useRouter()
    const pathname = usePathname()
    const [sidebarOpen, setSidebarOpen] = useState(false)
    const [sidebarExpanded, setSidebarExpanded] = useState(false)
    const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null)

    // Handle mouse enter with small delay to prevent flickering
    const handleMouseEnter = useCallback(() => {
        if (hoverTimeoutRef.current) {
            clearTimeout(hoverTimeoutRef.current)
        }
        setSidebarExpanded(true)
    }, [])

    // Handle mouse leave with delay
    const handleMouseLeave = useCallback(() => {
        hoverTimeoutRef.current = setTimeout(() => {
            setSidebarExpanded(false)
        }, 150)
    }, [])

    useEffect(() => {
        if (status === 'unauthenticated') {
            router.push('/auth/signin')
        }
    }, [status, router])

    if (status === 'loading') {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="spinner w-8 h-8" />
            </div>
        )
    }

    // Redirect if not teacher
    if (status === 'authenticated' && session?.user?.role !== 'teacher') {
        router.push('/dashboard')
        return null
    }

    const isExpanded = sidebarOpen || sidebarExpanded

    return (
        <div className="h-screen bg-gray-50 flex overflow-hidden">
            {/* Mobile sidebar backdrop */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* Sidebar - Expands on hover */}
            <aside
                className={clsx(
                    'fixed inset-y-0 left-0 z-50 bg-white border-r border-gray-200 transform transition-all duration-200 ease-out',
                    sidebarOpen ? 'w-64 translate-x-0' : '-translate-x-full lg:translate-x-0',
                    !sidebarOpen && (sidebarExpanded ? 'lg:w-64' : 'lg:w-16')
                )}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                <div className="flex flex-col h-screen">
                    {/* Logo */}
                    <div className="flex items-center px-3 py-4 border-b border-gray-100">
                        <Link href="/teacher/dashboard" className="flex items-center">
                            <AcademicCapIcon className="w-8 h-8 text-purple-600 flex-shrink-0" />
                            <span className={clsx(
                                'ml-3 text-xl font-bold text-purple-600 whitespace-nowrap overflow-hidden transition-all duration-200',
                                isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                            )}>
                                Teacher
                            </span>
                        </Link>
                        {sidebarOpen && (
                            <button
                                className="ml-auto lg:hidden"
                                onClick={() => setSidebarOpen(false)}
                            >
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        )}
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto overflow-x-hidden">
                        {navigation.map((item) => {
                            const isActive = pathname === item.href || pathname?.startsWith(item.href + '/')
                            return (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={clsx(
                                        'flex items-center p-3 rounded-xl font-medium transition-all duration-200',
                                        isActive
                                            ? 'bg-purple-50 text-purple-700'
                                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                                    )}
                                >
                                    <item.icon className="w-6 h-6 flex-shrink-0" />
                                    <span className={clsx(
                                        'ml-3 whitespace-nowrap overflow-hidden transition-all duration-200',
                                        isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                                    )}>
                                        {item.name}
                                    </span>
                                </Link>
                            )
                        })}
                    </nav>

                    {/* Logout */}
                    <div className="p-2 border-t border-gray-100">
                        <button
                            onClick={() => signOut({ callbackUrl: '/' })}
                            className="w-full flex items-center p-3 rounded-xl text-red-500 hover:bg-red-50 transition-colors"
                        >
                            <ArrowRightOnRectangleIcon className="w-6 h-6 flex-shrink-0" />
                            <span className={clsx(
                                'ml-3 whitespace-nowrap overflow-hidden transition-all duration-200',
                                isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                            )}>
                                Logout
                            </span>
                        </button>
                    </div>
                </div>
            </aside>

            {/* Main content */}
            <div className="flex-1 flex flex-col min-w-0 lg:ml-16">
                {/* Top bar - hidden on dashboard */}
                {pathname !== '/teacher/dashboard' && (
                    <header className="sticky top-0 z-30 bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4">
                        <button
                            className="lg:hidden"
                            onClick={() => setSidebarOpen(true)}
                        >
                            <Bars3Icon className="w-6 h-6" />
                        </button>
                        <h1 className="text-xl font-bold text-gray-900">
                            {navigation.find(n => pathname?.startsWith(n.href))?.name || 'Teacher Portal'}
                        </h1>
                    </header>
                )}

                {/* Mobile menu button for dashboard */}
                {pathname === '/teacher/dashboard' && (
                    <button
                        className="lg:hidden fixed top-4 left-4 z-30 p-2 bg-white rounded-lg shadow-md border border-gray-200"
                        onClick={() => setSidebarOpen(true)}
                    >
                        <Bars3Icon className="w-6 h-6" />
                    </button>
                )}

                {/* Page content */}
                <main className="flex-1 p-6 overflow-y-auto">
                    {children}
                </main>
            </div>
        </div>
    )
}
