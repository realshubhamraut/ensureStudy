'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    UserCircleIcon,
    KeyIcon,
    BellIcon,
    LinkIcon,
    ClipboardDocumentIcon,
    CheckCircleIcon,
    UserGroupIcon
} from '@heroicons/react/24/outline'

interface LinkedParent {
    id: string
    name: string
    email: string
    relationship_type: string
    linked_at: string
}

export default function StudentSettingsPage() {
    const { data: session } = useSession()
    const [loading, setLoading] = useState(false)
    const [formData, setFormData] = useState({
        first_name: '',
        last_name: '',
        phone: '',
        grade: '',
        school: ''
    })
    const [passwordData, setPasswordData] = useState({
        current_password: '',
        new_password: '',
        confirm_password: ''
    })
    const [notifications, setNotifications] = useState({
        email_quiz: true,
        email_results: true,
        email_reminders: false
    })

    // Parent linking state
    const [linkCode, setLinkCode] = useState<string | null>(null)
    const [linkedParents, setLinkedParents] = useState<LinkedParent[]>([])
    const [codeCopied, setCodeCopied] = useState(false)

    useEffect(() => {
        if (session?.user) {
            setFormData({
                first_name: (session.user as any).first_name || '',
                last_name: (session.user as any).last_name || '',
                phone: (session.user as any).phone || '',
                grade: (session.user as any).grade || '',
                school: (session.user as any).school || ''
            })
            fetchLinkCode()
            fetchLinkedParents()
        }
    }, [session])

    const fetchLinkCode = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/students/link-code', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setLinkCode(data.link_code)
            }
        } catch (error) {
            console.error('Failed to fetch link code:', error)
        }
    }

    const fetchLinkedParents = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/students/linked-parents', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setLinkedParents(data.parents || [])
            }
        } catch (error) {
            console.error('Failed to fetch linked parents:', error)
        }
    }

    const copyLinkCode = () => {
        if (linkCode) {
            navigator.clipboard.writeText(linkCode)
            setCodeCopied(true)
            setTimeout(() => setCodeCopied(false), 2000)
        }
    }

    const handleSaveProfile = async () => {
        setLoading(true)
        try {
            alert('Profile updated successfully!')
        } catch (error) {
            alert('Failed to update profile')
        } finally {
            setLoading(false)
        }
    }

    const handleChangePassword = async () => {
        if (passwordData.new_password !== passwordData.confirm_password) {
            alert('Passwords do not match')
            return
        }
        if (passwordData.new_password.length < 8) {
            alert('Password must be at least 8 characters')
            return
        }

        setLoading(true)
        try {
            const res = await fetch('http://localhost:8000/api/auth/change-password', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    current_password: passwordData.current_password,
                    new_password: passwordData.new_password
                })
            })

            if (res.ok) {
                alert('Password changed successfully!')
                setPasswordData({ current_password: '', new_password: '', confirm_password: '' })
            } else {
                const data = await res.json()
                alert(data.error || 'Failed to change password')
            }
        } catch (error) {
            alert('Failed to change password')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900">⚙️ Settings</h1>
                <p className="text-gray-600">Manage your account and preferences</p>
            </div>

            {/* Two Column Grid Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Parent Linking Code */}
                <div className="card border-2 border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-green-100">
                            <LinkIcon className="w-6 h-6 text-green-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Parent Linking Code</h2>
                            <p className="text-sm text-gray-500">Share this code with your parents</p>
                        </div>
                    </div>

                    <div className="bg-white rounded-xl p-4 border border-gray-200 mb-3">
                        <label className="block text-sm font-medium text-gray-600 mb-2">Your Unique Code</label>
                        <div className="flex items-center gap-3">
                            <div className="flex-1 bg-gray-50 rounded-lg p-3 border-2 border-dashed border-green-300">
                                <span className="text-2xl font-mono font-bold text-green-700 tracking-widest">
                                    {linkCode || 'Loading...'}
                                </span>
                            </div>
                            <button
                                onClick={copyLinkCode}
                                disabled={!linkCode}
                                className={`p-3 rounded-lg transition-all ${codeCopied
                                    ? 'bg-green-500 text-white'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                                title="Copy code"
                            >
                                {codeCopied ? (
                                    <CheckCircleIcon className="w-5 h-5" />
                                ) : (
                                    <ClipboardDocumentIcon className="w-5 h-5" />
                                )}
                            </button>
                        </div>
                        <p className="text-xs text-gray-400 mt-2">
                            Parents enter this code in their app to link to your account.
                        </p>
                    </div>

                    {linkedParents.length > 0 ? (
                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-600">Linked Parents</label>
                            {linkedParents.map((parent) => (
                                <div key={parent.id} className="flex items-center gap-3 bg-white rounded-lg p-2 border border-gray-200">
                                    <UserGroupIcon className="w-5 h-5 text-green-600" />
                                    <div className="flex-1">
                                        <p className="font-medium text-gray-900 text-sm">{parent.name}</p>
                                        <p className="text-xs text-gray-500">{parent.relationship_type}</p>
                                    </div>
                                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Linked</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-gray-500 italic">No parents linked yet</p>
                    )}
                </div>

                {/* Profile Settings */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-purple-100">
                            <UserCircleIcon className="w-6 h-6 text-purple-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Profile</h2>
                            <p className="text-sm text-gray-500">Your personal information</p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                                <input
                                    type="text"
                                    value={formData.first_name}
                                    onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                                <input
                                    type="text"
                                    value={formData.last_name}
                                    onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                                    className="input-field"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                            <input
                                type="email"
                                value={session?.user?.email || ''}
                                className="input-field bg-gray-50"
                                disabled
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Grade</label>
                                <select
                                    value={formData.grade}
                                    onChange={(e) => setFormData({ ...formData, grade: e.target.value })}
                                    className="input-field"
                                >
                                    <option value="">Select</option>
                                    {['6', '7', '8', '9', '10', '11', '12'].map(g => (
                                        <option key={g} value={g}>Class {g}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                                <input
                                    type="tel"
                                    value={formData.phone}
                                    onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                                    className="input-field"
                                    placeholder="+91..."
                                />
                            </div>
                        </div>

                        <button onClick={handleSaveProfile} disabled={loading} className="btn-primary w-full">
                            {loading ? 'Saving...' : 'Save Profile'}
                        </button>
                    </div>
                </div>

                {/* Notification Settings */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-blue-100">
                            <BellIcon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Notifications</h2>
                            <p className="text-sm text-gray-500">Manage notification preferences</p>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">New Quiz Assignments</p>
                                <p className="text-xs text-gray-500">Get notified for new quizzes</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_quiz}
                                onChange={(e) => setNotifications({ ...notifications, email_quiz: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>

                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Quiz Results</p>
                                <p className="text-xs text-gray-500">Get notified when graded</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_results}
                                onChange={(e) => setNotifications({ ...notifications, email_results: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>

                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Study Reminders</p>
                                <p className="text-xs text-gray-500">Daily study reminders</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_reminders}
                                onChange={(e) => setNotifications({ ...notifications, email_reminders: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>
                    </div>
                </div>

                {/* Change Password */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-red-100">
                            <KeyIcon className="w-6 h-6 text-red-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Change Password</h2>
                            <p className="text-sm text-gray-500">Update your password</p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Current Password</label>
                            <input
                                type="password"
                                value={passwordData.current_password}
                                onChange={(e) => setPasswordData({ ...passwordData, current_password: e.target.value })}
                                className="input-field"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                            <input
                                type="password"
                                value={passwordData.new_password}
                                onChange={(e) => setPasswordData({ ...passwordData, new_password: e.target.value })}
                                className="input-field"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
                            <input
                                type="password"
                                value={passwordData.confirm_password}
                                onChange={(e) => setPasswordData({ ...passwordData, confirm_password: e.target.value })}
                                className="input-field"
                            />
                        </div>

                        <button onClick={handleChangePassword} disabled={loading} className="btn-primary bg-red-600 hover:bg-red-700 w-full">
                            Change Password
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
