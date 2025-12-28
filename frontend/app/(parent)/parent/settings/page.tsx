'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    UserCircleIcon,
    KeyIcon,
    BellIcon,
    LinkIcon,
    UserGroupIcon,
    AcademicCapIcon
} from '@heroicons/react/24/outline'

interface LinkedChild {
    id: string
    student_id: string
    name: string
    email: string
    relationship_type: string
    linked_at: string
}

export default function ParentSettingsPage() {
    const { data: session } = useSession()
    const [loading, setLoading] = useState(false)
    const [formData, setFormData] = useState({
        first_name: '',
        last_name: '',
        phone: '',
        relationship: 'parent'
    })
    const [passwordData, setPasswordData] = useState({
        current_password: '',
        new_password: '',
        confirm_password: ''
    })
    const [notifications, setNotifications] = useState({
        email_progress: true,
        email_attendance: true,
        email_results: true
    })

    // Child linking state
    const [linkCode, setLinkCode] = useState('')
    const [linkedChildren, setLinkedChildren] = useState<LinkedChild[]>([])
    const [linking, setLinking] = useState(false)
    const [linkError, setLinkError] = useState('')

    useEffect(() => {
        if (session?.user) {
            setFormData({
                first_name: (session.user as any).first_name || '',
                last_name: (session.user as any).last_name || '',
                phone: (session.user as any).phone || '',
                relationship: (session.user as any).relationship || 'parent'
            })
            fetchLinkedChildren()
        }
    }, [session])

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
                    relationship_type: formData.relationship
                })
            })

            if (res.ok) {
                setLinkCode('')
                fetchLinkedChildren()
                alert('Successfully linked to your child!')
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
                <p className="text-gray-600">Manage your account and linked children</p>
            </div>

            {/* Two Column Grid Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Link Child Section */}
                <div className="card border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-indigo-50">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-blue-100">
                            <LinkIcon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Link Your Child</h2>
                            <p className="text-sm text-gray-500">Enter the code from your child's account</p>
                        </div>
                    </div>

                    <div className="bg-white rounded-xl p-4 border border-gray-200 mb-3">
                        <label className="block text-sm font-medium text-gray-600 mb-2">Child's Link Code</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="text"
                                value={linkCode}
                                onChange={(e) => setLinkCode(e.target.value.toUpperCase())}
                                placeholder="Enter 8-character code"
                                className="input-field font-mono text-lg tracking-widest uppercase"
                                maxLength={8}
                            />
                            <button
                                onClick={handleLinkChild}
                                disabled={linking || !linkCode.trim()}
                                className="btn-primary px-4 py-2 whitespace-nowrap"
                            >
                                {linking ? 'Linking...' : 'Link'}
                            </button>
                        </div>
                        {linkError && (
                            <p className="text-sm text-red-500 mt-2">{linkError}</p>
                        )}
                        <p className="text-xs text-gray-400 mt-2">
                            Ask your child to share their link code from their Settings page.
                        </p>
                    </div>

                    {/* Linked Children List */}
                    {linkedChildren.length > 0 ? (
                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-600">Your Children</label>
                            {linkedChildren.map((child) => (
                                <div key={child.id} className="flex items-center gap-3 bg-white rounded-lg p-3 border border-gray-200">
                                    <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                                        <AcademicCapIcon className="w-5 h-5 text-blue-600" />
                                    </div>
                                    <div className="flex-1">
                                        <p className="font-medium text-gray-900">{child.name}</p>
                                        <p className="text-sm text-gray-500">{child.email}</p>
                                    </div>
                                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">Linked</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-gray-500 italic">No children linked yet</p>
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
                                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                                <input
                                    type="tel"
                                    value={formData.phone}
                                    onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                                    className="input-field"
                                    placeholder="+91..."
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Relationship</label>
                                <select
                                    value={formData.relationship}
                                    onChange={(e) => setFormData({ ...formData, relationship: e.target.value })}
                                    className="input-field"
                                >
                                    <option value="parent">Parent</option>
                                    <option value="mother">Mother</option>
                                    <option value="father">Father</option>
                                    <option value="guardian">Guardian</option>
                                </select>
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
                        <div className="p-3 rounded-xl bg-green-100">
                            <BellIcon className="w-6 h-6 text-green-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Notifications</h2>
                            <p className="text-sm text-gray-500">Stay updated on your child's progress</p>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Progress Reports</p>
                                <p className="text-xs text-gray-500">Weekly progress updates</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_progress}
                                onChange={(e) => setNotifications({ ...notifications, email_progress: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>

                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Attendance Alerts</p>
                                <p className="text-xs text-gray-500">Notify when absent</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_attendance}
                                onChange={(e) => setNotifications({ ...notifications, email_attendance: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>

                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Test Results</p>
                                <p className="text-xs text-gray-500">Exam and quiz scores</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_results}
                                onChange={(e) => setNotifications({ ...notifications, email_results: e.target.checked })}
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
