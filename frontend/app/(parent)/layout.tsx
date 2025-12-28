'use client'

import { ReactNode, useState, useEffect, useCallback, useRef } from 'react'
import { useSession, signOut } from 'next-auth/react'
import { useRouter, usePathname } from 'next/navigation'
import Link from 'next/link'
import {
    UserGroupIcon,
    HomeIcon,
    ChartBarIcon,
    DocumentTextIcon,
    Cog6ToothIcon,
    ArrowRightOnRectangleIcon,
    Bars3Icon,
    XMarkIcon,
    AcademicCapIcon,
    BellIcon,
    PlusIcon,
    ChevronDownIcon,
    ChevronUpIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface LinkedChild {
    id: string
    student_id: string
    name: string
    email: string
}

const navigation = [
    { name: 'Dashboard', href: '/parent/dashboard', icon: HomeIcon },
    { name: 'Progress', href: '/parent/progress', icon: ChartBarIcon },
    { name: 'Reports', href: '/parent/reports', icon: DocumentTextIcon },
    { name: 'Notifications', href: '/parent/notifications', icon: BellIcon },
]

export default function ParentLayout({ children }: { children: ReactNode }) {
    const { data: session, status } = useSession()
    const router = useRouter()
    const pathname = usePathname()
    const [sidebarOpen, setSidebarOpen] = useState(false)
    const [sidebarExpanded, setSidebarExpanded] = useState(false)
    const [linkedChildren, setLinkedChildren] = useState<LinkedChild[]>([])
    const [childrenExpanded, setChildrenExpanded] = useState(true)
    const [showLinkModal, setShowLinkModal] = useState(false)
    const [linkCode, setLinkCode] = useState('')
    const [linking, setLinking] = useState(false)
    const [linkError, setLinkError] = useState('')
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
        if (status === 'authenticated') {
            fetchLinkedChildren()
        }
    }, [status])

    const fetchLinkedChildren = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/students/linked-children', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setLinkedChildren(data.children || [])
            }
        } catch (error) {
            console.error('Failed to fetch linked children:', error)
        }
    }

    const handleLinkChild = async () => {
        if (!linkCode.trim()) {
            setLinkError('Please enter a link code')
            return
        }

        setLinking(true)
        setLinkError('')

        try {
            const res = await fetch('http://localhost:8000/api/students/link-by-code', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    link_code: linkCode,
                    relationship_type: 'parent'
                })
            })

            if (res.ok) {
                setLinkCode('')
                setShowLinkModal(false)
                fetchLinkedChildren()
            } else {
                const data = await res.json()
                setLinkError(data.error || 'Invalid link code')
            }
        } catch (error) {
            setLinkError('Failed to link. Please try again.')
        } finally {
            setLinking(false)
        }
    }

    // Redirect if not authenticated
    if (status === 'unauthenticated') {
        router.push('/auth/signin')
        return null
    }

    // Redirect non-parents
    if (status === 'authenticated' && session?.user?.role) {
        const role = session.user.role as string
        if (role !== 'parent') {
            if (role === 'admin') router.push('/admin/dashboard')
            else if (role === 'teacher') router.push('/teacher/dashboard')
            else if (role === 'student') router.push('/dashboard')
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

            {/* Link Child Modal */}
            {showLinkModal && (
                <div className="fixed inset-0 bg-black/50 z-[100] flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl p-6 w-full max-w-md shadow-2xl">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">🔗 Link a Child</h3>
                        <p className="text-sm text-gray-600 mb-4">
                            Enter your child's link code from their Settings page.
                        </p>
                        <input
                            type="text"
                            value={linkCode}
                            onChange={(e) => setLinkCode(e.target.value.toUpperCase())}
                            placeholder="Enter 8-character code"
                            className="input-field font-mono text-lg tracking-widest uppercase mb-2"
                            maxLength={8}
                        />
                        {linkError && (
                            <p className="text-sm text-red-500 mb-3">{linkError}</p>
                        )}
                        <div className="flex gap-3 mt-4">
                            <button
                                onClick={() => { setShowLinkModal(false); setLinkError(''); setLinkCode(''); }}
                                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleLinkChild}
                                disabled={linking || !linkCode.trim()}
                                className="flex-1 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50"
                            >
                                {linking ? 'Linking...' : 'Link Child'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Sidebar */}
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
                    <div className="flex items-center px-3 py-3 border-b border-gray-100 flex-shrink-0">
                        <Link href="/parent/dashboard" className="flex items-center">
                            <UserGroupIcon className="w-8 h-8 text-orange-600 flex-shrink-0" />
                            <span className={clsx(
                                'ml-3 text-xl font-bold text-orange-600 whitespace-nowrap overflow-hidden transition-all duration-200',
                                isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                            )}>
                                Parent Portal
                            </span>
                        </Link>
                        {sidebarOpen && (
                            <button className="ml-auto lg:hidden" onClick={() => setSidebarOpen(false)}>
                                <XMarkIcon className="w-6 h-6" />
                            </button>
                        )}
                    </div>

                    {/* Navigation */}
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
                                            ? 'bg-orange-50 text-orange-700'
                                            : 'text-gray-600 hover:bg-orange-50 hover:text-orange-600'
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

                    {/* Children Section - Above Settings/Logout */}
                    <div className="px-2 py-2 border-t border-gray-100 flex-shrink-0 bg-white">
                        <button
                            onClick={() => setChildrenExpanded(!childrenExpanded)}
                            className="w-full flex items-center p-2.5 rounded-xl text-gray-600 hover:bg-orange-50 transition-colors"
                        >
                            <AcademicCapIcon className="w-5 h-5 flex-shrink-0 text-orange-600" />
                            <span className={clsx(
                                'ml-3 whitespace-nowrap overflow-hidden transition-all duration-200 text-sm font-medium flex-1 text-left',
                                isExpanded ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0'
                            )}>
                                My Children ({linkedChildren.length})
                            </span>
                            <span className={clsx(
                                'transition-all duration-200',
                                isExpanded ? 'opacity-100' : 'opacity-0'
                            )}>
                                {childrenExpanded ? (
                                    <ChevronUpIcon className="w-4 h-4" />
                                ) : (
                                    <ChevronDownIcon className="w-4 h-4" />
                                )}
                            </span>
                        </button>

                        {/* Children List - Collapsible */}
                        <div className={clsx(
                            'overflow-hidden transition-all duration-200',
                            childrenExpanded && isExpanded ? 'max-h-48' : 'max-h-0'
                        )}>
                            <div className="pl-2 space-y-1 mt-1">
                                {linkedChildren.map((child) => (
                                    <Link
                                        key={child.id}
                                        href={`/parent/children/${child.student_id}`}
                                        className="flex items-center gap-2 p-2 rounded-lg text-sm text-gray-600 hover:bg-orange-50 hover:text-orange-600 transition-colors"
                                    >
                                        <div className="w-6 h-6 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0">
                                            <span className="text-orange-600 text-xs font-semibold">
                                                {child.name?.[0]?.toUpperCase() || 'S'}
                                            </span>
                                        </div>
                                        <span className="truncate">{child.name}</span>
                                    </Link>
                                ))}

                                {/* Add Child Button */}
                                <button
                                    onClick={() => setShowLinkModal(true)}
                                    className="w-full flex items-center gap-2 p-2 rounded-lg text-sm text-orange-600 hover:bg-orange-50 transition-colors"
                                >
                                    <div className="w-6 h-6 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0">
                                        <PlusIcon className="w-4 h-4 text-orange-600" />
                                    </div>
                                    <span>Add Child</span>
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* User section - always visible at bottom */}
                    <div className="p-2 border-t border-gray-100 flex-shrink-0 bg-white">
                        <Link
                            href="/parent/settings"
                            className="flex items-center p-2.5 rounded-xl text-gray-600 hover:bg-orange-50 hover:text-orange-600 transition-colors"
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
                <header className="sticky top-0 z-30 bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4">
                    <button className="lg:hidden" onClick={() => setSidebarOpen(true)}>
                        <Bars3Icon className="w-6 h-6" />
                    </button>
                    <h1 className="text-xl font-bold text-gray-900">
                        {navigation.find(n => pathname?.startsWith(n.href))?.name || 'Dashboard'}
                    </h1>
                </header>

                <main className="flex-1 p-6 overflow-y-auto">
                    {children}
                </main>
            </div>
        </div>
    )
}
