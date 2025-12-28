'use client'

import { ReactNode } from 'react'
import { useSession, signOut } from 'next-auth/react'
import { useRouter, usePathname } from 'next/navigation'
import Link from 'next/link'
import {
    AcademicCapIcon,
    HomeIcon,
    ChatBubbleLeftRightIcon,
    BookOpenIcon,
    ClipboardDocumentListIcon,
    TrophyIcon,
    ChartBarIcon,
    Cog6ToothIcon,
    ArrowRightOnRectangleIcon,
    Bars3Icon,
    XMarkIcon,
    FolderIcon,
    UserGroupIcon
} from '@heroicons/react/24/outline'
import { useState, useCallback, useRef } from 'react'
import clsx from 'clsx'

const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'My Classrooms', href: '/classrooms', icon: FolderIcon },
    { name: 'AI Tutor', href: '/chat', icon: ChatBubbleLeftRightIcon },
    { name: 'Soft Skills AI', href: '/softskills', icon: UserGroupIcon },
    { name: 'Assessments', href: '/assessments', icon: ClipboardDocumentListIcon },
    { name: 'Progress', href: '/progress', icon: ChartBarIcon },
    { name: 'Leaderboard', href: '/leaderboard', icon: TrophyIcon },
]

export default function DashboardLayout({ children }: { children: ReactNode }) {
    const { data: session, status } = useSession()
    const router = useRouter()
    const pathname = usePathname()
    const [sidebarOpen, setSidebarOpen] = useState(false) // Mobile sidebar
    const [sidebarExpanded, setSidebarExpanded] = useState(false) // Desktop hover expand
    const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null)

    // Handle mouse enter with small delay to prevent flickering
    const handleMouseEnter = useCallback(() => {
        if (hoverTimeoutRef.current) {
            clearTimeout(hoverTimeoutRef.current)
        }
        setSidebarExpanded(true)
    }, [])

    // Handle mouse leave with delay to allow moving to submenu items
    const handleMouseLeave = useCallback(() => {
        hoverTimeoutRef.current = setTimeout(() => {
            setSidebarExpanded(false)
        }, 150) // Small delay to prevent accidental closes
    }, [])

    // Redirect if not authenticated
    if (status === 'unauthenticated') {
        router.push('/auth/signin')
        return null
    }

    // Redirect non-students to their appropriate dashboard
    if (status === 'authenticated' && session?.user?.role) {
        const role = session.user.role as string
        if (role === 'admin') {
            router.push('/admin/dashboard')
            return null
        } else if (role === 'teacher') {
            router.push('/teacher/dashboard')
            return null
        } else if (role === 'parent') {
            router.push('/parent/dashboard')
            return null
        }
    }

    if (status === 'loading') {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="spinner w-8 h-8" />
            </div>
        )
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

            {/* Sidebar - Expands on hover (desktop) or click (mobile) */}
            <aside
                className={clsx(
                    'fixed inset-y-0 left-0 z-50 bg-white border-r border-gray-200 transform transition-all duration-200 ease-out',
                    sidebarOpen ? 'w-64 translate-x-0' : '-translate-x-full lg:translate-x-0',
                    // Desktop width controlled by JS state
                    !sidebarOpen && (sidebarExpanded ? 'lg:w-64' : 'lg:w-16')
                )}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                <div className="flex flex-col h-screen">
                    {/* Logo */}
                    <div className="flex items-center px-3 py-3 border-b border-gray-100 flex-shrink-0">
                        <Link href="/dashboard" className="flex items-center">
                            <AcademicCapIcon className="w-8 h-8 text-primary-600 flex-shrink-0" />
                            <span className={clsx(
                                'ml-3 text-xl font-bold gradient-text whitespace-nowrap overflow-hidden transition-all duration-200',
                                isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                            )}>
                                ensureStudy
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

                    {/* Navigation - scrollable if needed but constrained */}
                    <nav className="flex-1 px-2 py-3 space-y-1 overflow-y-auto overflow-x-hidden min-h-0">
                        {navigation.map((item) => {
                            const isActive = pathname === item.href || pathname?.startsWith(item.href + '/')
                            return (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={clsx(
                                        'flex items-center p-2.5 rounded-xl font-medium transition-all duration-200',
                                        isActive
                                            ? 'bg-primary-50 text-primary-700'
                                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                                    )}
                                >
                                    <item.icon className="w-5 h-5 flex-shrink-0" />
                                    <span className={clsx(
                                        'ml-3 whitespace-nowrap overflow-hidden transition-all duration-200 text-sm',
                                        isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                                    )}>
                                        {item.name}
                                    </span>
                                </Link>
                            )
                        })}
                    </nav>

                    {/* User section - always visible at bottom */}
                    <div className="p-2 border-t border-gray-100 flex-shrink-0 bg-white">
                        <Link
                            href="/settings"
                            className="flex items-center p-2.5 rounded-xl text-gray-600 hover:bg-gray-100 transition-colors"
                        >
                            <Cog6ToothIcon className="w-5 h-5 flex-shrink-0" />
                            <span className={clsx(
                                'ml-3 whitespace-nowrap overflow-hidden transition-all duration-200 text-sm',
                                isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                            )}>
                                Settings
                            </span>
                        </Link>
                        <button
                            onClick={() => signOut({ callbackUrl: '/' })}
                            className="w-full flex items-center p-2.5 rounded-xl text-danger-500 hover:bg-danger-50 transition-colors"
                        >
                            <ArrowRightOnRectangleIcon className="w-5 h-5 flex-shrink-0" />
                            <span className={clsx(
                                'ml-3 whitespace-nowrap overflow-hidden transition-all duration-200 text-sm',
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
                {/* Top bar - Hidden on chat page for cleaner UI */}
                {!pathname?.startsWith('/chat') && (
                    <header className="sticky top-0 z-30 bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4">
                        <button
                            className="lg:hidden"
                            onClick={() => setSidebarOpen(true)}
                        >
                            <Bars3Icon className="w-6 h-6" />
                        </button>
                        <h1 className="text-xl font-bold text-gray-900">
                            {navigation.find(n => pathname?.startsWith(n.href))?.name || 'Dashboard'}
                        </h1>
                    </header>
                )}

                {/* Page content */}
                <main className={`flex-1 overflow-y-auto ${pathname?.startsWith('/chat') ? 'p-0' : 'p-6'}`}>
                    {children}
                </main>
            </div>
        </div>
    )
}
