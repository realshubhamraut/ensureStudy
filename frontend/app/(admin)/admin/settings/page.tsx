'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    BuildingOfficeIcon,
    KeyIcon,
    BellIcon,
    ShieldCheckIcon,
    ArrowPathIcon,
    LightBulbIcon
} from '@heroicons/react/24/outline'

export default function SettingsPage() {
    const { data: session } = useSession()
    const [organization, setOrganization] = useState<any>(null)
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [formData, setFormData] = useState({
        name: '',
        phone: '',
        address: '',
        city: '',
        state: ''
    })

    useEffect(() => {
        const fetchOrg = async () => {
            try {
                const res = await fetch(`${getApiBaseUrl()}/api/admin/organization`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                })
                if (res.ok) {
                    const data = await res.json()
                    setOrganization(data.organization)
                    setFormData({
                        name: data.organization.name || '',
                        phone: data.organization.phone || '',
                        address: data.organization.address || '',
                        city: data.organization.city || '',
                        state: data.organization.state || ''
                    })
                }
            } catch (error) {
                console.error('Failed to fetch organization:', error)
            } finally {
                setLoading(false)
            }
        }

        fetchOrg()
    }, [])

    const handleSave = async () => {
        setSaving(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/admin/organization`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            if (res.ok) {
                const data = await res.json()
                setOrganization(data.organization)
                alert('Settings saved successfully!')
            }
        } catch (error) {
            console.error('Failed to save:', error)
            alert('Failed to save settings')
        } finally {
            setSaving(false)
        }
    }

    const regenerateToken = async () => {
        if (!confirm('Are you sure? Existing registration links will stop working.')) return

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/admin/organization/regenerate-token`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            })
            if (res.ok) {
                const data = await res.json()
                setOrganization({ ...organization, access_token: data.access_token })
                alert('Token regenerated!')
            }
        } catch (error) {
            console.error('Failed to regenerate token:', error)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900">⚙️ Settings</h1>
                <p className="text-gray-600">Manage your organization settings</p>
            </div>

            {/* Two Column Grid Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Organization Info */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-blue-100">
                            <BuildingOfficeIcon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Organization</h2>
                            <p className="text-sm text-gray-500">Your school information</p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Organization Name</label>
                            <input
                                type="text"
                                value={formData.name}
                                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                className="input-field"
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

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Address</label>
                            <input
                                type="text"
                                value={formData.address}
                                onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                                className="input-field"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">City</label>
                                <input
                                    type="text"
                                    value={formData.city}
                                    onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">State</label>
                                <input
                                    type="text"
                                    value={formData.state}
                                    onChange={(e) => setFormData({ ...formData, state: e.target.value })}
                                    className="input-field"
                                />
                            </div>
                        </div>

                        <button onClick={handleSave} disabled={saving} className="btn-primary w-full">
                            {saving ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>
                </div>

                {/* Access Token */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-purple-100">
                            <KeyIcon className="w-6 h-6 text-purple-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Access Token</h2>
                            <p className="text-sm text-gray-500">Share to register teachers/students</p>
                        </div>
                    </div>

                    <div className="p-4 bg-gray-50 rounded-xl">
                        <div className="flex items-center gap-3">
                            <code className="flex-1 text-lg font-mono font-bold text-primary-600 bg-white px-3 py-2 rounded-lg border truncate">
                                {organization?.access_token || 'Loading...'}
                            </code>
                            <button
                                onClick={() => {
                                    navigator.clipboard.writeText(organization?.access_token || '')
                                    alert('Token copied!')
                                }}
                                className="btn-primary px-3 py-2"
                            >
                                Copy
                            </button>
                        </div>

                        <div className="mt-3 flex items-center justify-between">
                            <p className="text-xs text-gray-500">Need a new token?</p>
                            <button
                                onClick={regenerateToken}
                                className="flex items-center gap-1 text-xs text-red-600 hover:text-red-700"
                            >
                                <ArrowPathIcon className="w-3 h-3" />
                                Regenerate
                            </button>
                        </div>
                    </div>
                </div>

                {/* Admission Window Control */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className={`p-3 rounded-xl ${organization?.admission_open ? 'bg-green-100' : 'bg-red-100'}`}>
                            <BellIcon className={`w-6 h-6 ${organization?.admission_open ? 'text-green-600' : 'text-red-600'}`} />
                        </div>
                        <div className="flex-1">
                            <h2 className="text-lg font-semibold text-gray-900">Admission Window</h2>
                            <p className="text-sm text-gray-500">Control new registrations</p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${organization?.admission_open
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'
                            }`}>
                            {organization?.admission_open ? 'Open' : 'Closed'}
                        </span>
                    </div>

                    <div className="p-4 bg-gray-50 rounded-xl">
                        <p className="font-medium text-gray-900 text-sm">
                            {organization?.admission_open
                                ? '🟢 Admissions are OPEN'
                                : '🔴 Admissions are CLOSED'
                            }
                        </p>
                        <p className="text-xs text-gray-500 mt-1 mb-3">
                            {organization?.admission_open
                                ? 'New users can register with your token.'
                                : 'No new registrations allowed.'
                            }
                        </p>
                        <button
                            onClick={async () => {
                                const newState = !organization?.admission_open
                                const confirmMsg = newState
                                    ? 'Open admissions?'
                                    : 'Close admissions?'

                                if (!confirm(confirmMsg)) return

                                try {
                                    const res = await fetch(`${getApiBaseUrl()}/api/admin/admission/toggle`, {
                                        method: 'POST',
                                        headers: {
                                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                                            'Content-Type': 'application/json'
                                        },
                                        body: JSON.stringify({ admission_open: newState })
                                    })
                                    if (res.ok) {
                                        setOrganization({ ...organization, admission_open: newState })
                                    }
                                } catch (error) {
                                    console.error('Failed to toggle admission:', error)
                                }
                            }}
                            className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${organization?.admission_open
                                ? 'bg-red-600 text-white hover:bg-red-700'
                                : 'bg-green-600 text-white hover:bg-green-700'
                                }`}
                        >
                            {organization?.admission_open ? 'Close Admissions' : 'Open Admissions'}
                        </button>
                    </div>

                    <p className="text-xs text-gray-400 mt-3 flex items-center gap-1">
                        <LightBulbIcon className="w-4 h-4" /> Close after registration is complete.
                    </p>
                </div>

                {/* Subscription Info */}
                <div className="card">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-3 rounded-xl bg-green-100">
                            <ShieldCheckIcon className="w-6 h-6 text-green-600" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-gray-900">Subscription</h2>
                            <p className="text-sm text-gray-500">Your current plan</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Status</p>
                            <p className="text-lg font-semibold capitalize">{organization?.subscription_status}</p>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Licenses</p>
                            <p className="text-lg font-semibold">{organization?.license_count}</p>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Used</p>
                            <p className="text-lg font-semibold">{organization?.used_licenses || 0}</p>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Available</p>
                            <p className="text-lg font-semibold text-green-600">
                                {organization?.license_count - (organization?.used_licenses || 0)}
                            </p>
                        </div>
                    </div>

                    <a href="/admin/billing" className="inline-block mt-3 text-primary-600 hover:underline text-sm">
                        Manage billing →
                    </a>
                </div>
            </div>
        </div>
    )
}
