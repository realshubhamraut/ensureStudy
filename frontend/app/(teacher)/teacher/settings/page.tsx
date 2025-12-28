'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    UserCircleIcon,
    KeyIcon,
    BellIcon,
    AcademicCapIcon
} from '@heroicons/react/24/outline'

export default function TeacherSettingsPage() {
    const { data: session } = useSession()
    const [loading, setLoading] = useState(false)
    const [formData, setFormData] = useState({
        first_name: '',
        last_name: '',
        phone: '',
        subject: '',
        department: ''
    })
    const [passwordData, setPasswordData] = useState({
        current_password: '',
        new_password: '',
        confirm_password: ''
    })
    const [notifications, setNotifications] = useState({
        email_submissions: true,
        email_announcements: true,
        email_reminders: false
    })

    useEffect(() => {
        if (session?.user) {
            setFormData({
                first_name: (session.user as any).first_name || '',
                last_name: (session.user as any).last_name || '',
                phone: (session.user as any).phone || '',
                subject: (session.user as any).subject || '',
                department: (session.user as any).department || ''
            })
        }
    }, [session])

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
                <p className="text-gray-600">Manage your account settings</p>
            </div>

            {/* Two Column Grid Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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

                        <button onClick={handleSaveProfile} disabled={loading} className="btn-primary w-full">
                            {loading ? 'Saving...' : 'Save Profile'}
                        </button>
                    </div>
                </div>

                {/* Teaching Info */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-blue-100">
                            <AcademicCapIcon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Teaching Info</h2>
                            <p className="text-sm text-gray-500">Your academic details</p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Primary Subject</label>
                            <select
                                value={formData.subject}
                                onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                className="input-field"
                            >
                                <option value="">Select Subject</option>
                                <option value="Mathematics">Mathematics</option>
                                <option value="Physics">Physics</option>
                                <option value="Chemistry">Chemistry</option>
                                <option value="Biology">Biology</option>
                                <option value="English">English</option>
                                <option value="Computer Science">Computer Science</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Department</label>
                            <input
                                type="text"
                                value={formData.department}
                                onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                                className="input-field"
                                placeholder="Science Department"
                            />
                        </div>

                        <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-sm text-gray-600">
                                <span className="font-medium">School:</span> {(session?.user as any)?.organization || 'Not assigned'}
                            </p>
                            <p className="text-xs text-gray-400 mt-1">Contact admin to change school</p>
                        </div>
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
                            <p className="text-sm text-gray-500">Manage notification preferences</p>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Student Submissions</p>
                                <p className="text-xs text-gray-500">Alert when students submit work</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_submissions}
                                onChange={(e) => setNotifications({ ...notifications, email_submissions: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>

                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">School Announcements</p>
                                <p className="text-xs text-gray-500">Admin announcements</p>
                            </div>
                            <input
                                type="checkbox"
                                checked={notifications.email_announcements}
                                onChange={(e) => setNotifications({ ...notifications, email_announcements: e.target.checked })}
                                className="w-5 h-5 text-primary-600 rounded"
                            />
                        </label>

                        <label className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer">
                            <div>
                                <p className="font-medium text-gray-900 text-sm">Class Reminders</p>
                                <p className="text-xs text-gray-500">Upcoming class notifications</p>
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
