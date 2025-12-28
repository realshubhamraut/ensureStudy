'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    UsersIcon,
    AcademicCapIcon,
    TicketIcon,
    ChartBarIcon,
    ClipboardDocumentIcon,
    ArrowTrendingUpIcon
} from '@heroicons/react/24/outline'

interface DashboardStats {
    license_count: number
    used_licenses: number
    available_licenses: number
    total_teachers: number
    total_students: number
    total_parents: number
    total_users: number
}

export default function AdminDashboard() {
    const { data: session } = useSession()
    const [stats, setStats] = useState<DashboardStats | null>(null)
    const [accessToken, setAccessToken] = useState('')
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchDashboard = async () => {
            try {
                const res = await fetch('http://localhost:8000/api/admin/dashboard', {
                    headers: {
                        'Authorization': `Bearer ${(session as any)?.accessToken}`
                    }
                })
                if (res.ok) {
                    const data = await res.json()
                    setStats(data.stats)
                    setAccessToken(data.access_token)
                }
            } catch (error) {
                console.error('Failed to fetch dashboard:', error)
            } finally {
                setLoading(false)
            }
        }

        if (session) {
            fetchDashboard()
        }
    }, [session])

    const statCards = stats ? [
        {
            name: 'Total Licenses',
            value: stats.license_count,
            icon: TicketIcon,
            color: 'from-blue-500 to-cyan-500',
            subtext: `${stats.available_licenses} available`
        },
        {
            name: 'Students',
            value: stats.total_students,
            icon: AcademicCapIcon,
            color: 'from-green-500 to-emerald-500',
            subtext: `Using ${stats.used_licenses} licenses`
        },
        {
            name: 'Teachers',
            value: stats.total_teachers,
            icon: UsersIcon,
            color: 'from-purple-500 to-pink-500',
            subtext: 'Active teachers'
        },
        {
            name: 'Total Users',
            value: stats.total_users,
            icon: ChartBarIcon,
            color: 'from-orange-500 to-red-500',
            subtext: 'All registered users'
        },
    ] : []

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
                <p className="text-gray-600">Welcome back! Here's an overview of your organization.</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {statCards.map((stat) => (
                    <div key={stat.name} className="card-hover">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-sm font-medium text-gray-500">{stat.name}</p>
                                <p className="mt-2 text-3xl font-bold text-gray-900">{stat.value}</p>
                                <p className="mt-1 text-sm text-gray-500">{stat.subtext}</p>
                            </div>
                            <div className={`p-3 rounded-xl bg-gradient-to-br ${stat.color}`}>
                                <stat.icon className="w-6 h-6 text-white" />
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Access Token Section */}
            <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Registration Access Token</h2>
                <p className="text-gray-600 mb-4">
                    Share this token with teachers and students to register under your organization.
                </p>

                <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
                    <div className="flex-1">
                        <code className="text-lg font-mono font-bold text-primary-600">
                            {accessToken}
                        </code>
                    </div>
                    <button
                        onClick={() => {
                            navigator.clipboard.writeText(accessToken)
                            alert('Token copied to clipboard!')
                        }}
                        className="btn-primary flex items-center gap-2"
                    >
                        <ClipboardDocumentIcon className="w-5 h-5" />
                        Copy Token
                    </button>
                </div>

                <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <h3 className="font-medium text-blue-900 mb-2">How to use:</h3>
                    <ol className="list-decimal list-inside text-sm text-blue-800 space-y-1">
                        <li>Share this token with teachers and students</li>
                        <li>They enter this token during registration</li>
                        <li>They will be automatically linked to your organization</li>
                        <li>Students use 1 license each; teachers are free</li>
                    </ol>
                </div>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <a href="/admin/teachers" className="card-hover flex items-center gap-4 group">
                    <div className="p-3 rounded-xl bg-purple-100 group-hover:bg-purple-200 transition-colors">
                        <UsersIcon className="w-6 h-6 text-purple-600" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-900">Manage Teachers</h3>
                        <p className="text-sm text-gray-500">View and manage teacher accounts</p>
                    </div>
                    <ArrowTrendingUpIcon className="w-5 h-5 text-gray-400 ml-auto" />
                </a>

                <a href="/admin/students" className="card-hover flex items-center gap-4 group">
                    <div className="p-3 rounded-xl bg-green-100 group-hover:bg-green-200 transition-colors">
                        <AcademicCapIcon className="w-6 h-6 text-green-600" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-900">Manage Students</h3>
                        <p className="text-sm text-gray-500">View and manage student accounts</p>
                    </div>
                    <ArrowTrendingUpIcon className="w-5 h-5 text-gray-400 ml-auto" />
                </a>

                <a href="/admin/billing" className="card-hover flex items-center gap-4 group">
                    <div className="p-3 rounded-xl bg-blue-100 group-hover:bg-blue-200 transition-colors">
                        <TicketIcon className="w-6 h-6 text-blue-600" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-900">Buy Licenses</h3>
                        <p className="text-sm text-gray-500">Purchase more student licenses</p>
                    </div>
                    <ArrowTrendingUpIcon className="w-5 h-5 text-gray-400 ml-auto" />
                </a>
            </div>
        </div>
    )
}
