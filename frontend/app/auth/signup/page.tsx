'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import {
    AcademicCapIcon,
    BuildingOfficeIcon,
    UsersIcon,
    UserIcon,
    ArrowRightIcon,
    CurrencyRupeeIcon,
    GiftIcon
} from '@heroicons/react/24/outline'

type Role = 'admin' | 'teacher' | 'student' | 'parent'

const roles = [
    { id: 'admin' as Role, name: 'School', icon: BuildingOfficeIcon },
    { id: 'teacher' as Role, name: 'Teacher', icon: UsersIcon },
    { id: 'student' as Role, name: 'Student', icon: AcademicCapIcon },
    { id: 'parent' as Role, name: 'Parent', icon: UserIcon },
]

export default function SignUpPage() {
    const router = useRouter()
    const searchParams = useSearchParams()
    const initialRole = (searchParams.get('role') as Role) || 'student'

    const [selectedRole, setSelectedRole] = useState<Role>(initialRole)

    // Form state
    const [email, setEmail] = useState('')
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [firstName, setFirstName] = useState('')
    const [lastName, setLastName] = useState('')

    // Role-specific fields
    const [organizationName, setOrganizationName] = useState('')
    const [accessToken, setAccessToken] = useState('')
    const [studentLinkCode, setStudentLinkCode] = useState('')

    // Student profile
    const [gradeLevel, setGradeLevel] = useState('')
    const [board, setBoard] = useState('')
    const [targetExams, setTargetExams] = useState<string[]>([])

    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [tokenValid, setTokenValid] = useState<boolean | null>(null)
    const [tokenOrg, setTokenOrg] = useState('')

    const validateToken = async (token: string) => {
        if (!token) {
            setTokenValid(null)
            return
        }
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/auth/validate-token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ access_token: token })
            })
            const data = await res.json()
            setTokenValid(data.valid)
            if (data.valid) setTokenOrg(data.organization.name)
        } catch {
            setTokenValid(false)
        }
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')

        if (password !== confirmPassword) {
            setError('Passwords do not match')
            return
        }

        if (password.length < 8) {
            setError('Password must be at least 8 characters')
            return
        }

        setLoading(true)

        try {
            const body: any = {
                email,
                username,
                password,
                first_name: firstName,
                last_name: lastName,
                role: selectedRole
            }

            if (selectedRole === 'admin') {
                body.organization_name = organizationName
            } else if (selectedRole === 'teacher') {
                body.access_token = accessToken
            } else if (selectedRole === 'parent') {
                body.student_link_code = studentLinkCode
            }

            if (selectedRole === 'student') {
                body.profile = { grade_level: gradeLevel, board, target_exams: targetExams }
            }

            const res = await fetch(`${getApiBaseUrl()}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            })

            const data = await res.json()

            if (!res.ok) throw new Error(data.error || 'Registration failed')

            localStorage.setItem('accessToken', data.access_token)

            switch (selectedRole) {
                case 'admin': router.push('/admin/dashboard'); break
                case 'teacher': router.push('/teacher/dashboard'); break
                case 'parent': router.push('/parent/dashboard'); break
                default: router.push('/join-classroom')
            }
        } catch (err: any) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-white to-secondary-50 p-4">
            <div className="w-full max-w-lg">
                {/* Logo */}
                <div className="text-center mb-8">
                    <Link href="/" className="inline-flex items-center gap-2">
                        <AcademicCapIcon className="w-10 h-10 text-primary-600" />
                        <span className="text-3xl font-bold gradient-text">ensureStudy</span>
                    </Link>
                </div>

                <div className="card">
                    {/* Role Toggle Tabs */}
                    <div className="flex rounded-xl bg-gray-100 p-1 mb-6">
                        {roles.map((role) => (
                            <button
                                key={role.id}
                                onClick={() => setSelectedRole(role.id)}
                                className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 px-2 rounded-lg text-sm font-medium transition-all ${selectedRole === role.id
                                    ? 'bg-white text-primary-600 shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700'
                                    }`}
                            >
                                <role.icon className="w-4 h-4" />
                                <span className="hidden sm:inline">{role.name}</span>
                            </button>
                        ))}
                    </div>

                    {/* Sign Up Heading */}
                    <h1 className="text-xl font-bold text-gray-900 text-center mb-6">
                        Create {roles.find(r => r.id === selectedRole)?.name} Account
                    </h1>

                    {error && (
                        <div className="p-3 bg-red-50 text-red-700 rounded-lg mb-4 text-sm">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* Admin: Pricing Info */}
                        {selectedRole === 'admin' && (
                            <div className="p-3 bg-blue-50 rounded-lg border border-blue-200 text-sm flex items-center justify-between">
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <CurrencyRupeeIcon className="w-4 h-4 text-blue-600" />
                                        <span className="font-medium text-blue-900">₹29/student/year • Teachers FREE</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-blue-600">
                                        <GiftIcon className="w-4 h-4" />
                                        <span>5 free trial licenses included</span>
                                    </div>
                                </div>
                                <Link
                                    href="/pricing"
                                    className="px-3 py-1.5 bg-blue-600 text-white text-xs font-medium rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap"
                                >
                                    Upgrade
                                </Link>
                            </div>
                        )}

                        {/* Admin: Organization Name */}
                        {selectedRole === 'admin' && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">School/Organization Name *</label>
                                <input
                                    type="text"
                                    value={organizationName}
                                    onChange={(e) => setOrganizationName(e.target.value)}
                                    className="input-field"
                                    placeholder="ABC International School"
                                    required
                                />
                            </div>
                        )}

                        {/* Teacher: Access Token */}
                        {selectedRole === 'teacher' && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">School Access Token *</label>
                                <input
                                    type="text"
                                    value={accessToken}
                                    onChange={(e) => { setAccessToken(e.target.value); validateToken(e.target.value) }}
                                    className="input-field"
                                    placeholder="Get this from your school admin"
                                    required
                                />
                                {tokenValid === true && <p className="text-green-600 text-xs mt-1">✓ Valid token for {tokenOrg}</p>}
                                {tokenValid === false && <p className="text-red-600 text-xs mt-1">✗ Invalid token</p>}
                            </div>
                        )}

                        {/* Student: Info */}
                        {selectedRole === 'student' && (
                            <div className="p-3 bg-blue-50 rounded-lg border border-blue-200 text-sm text-blue-700">
                                After signup, you'll join your class using a code from your teacher.
                            </div>
                        )}

                        {/* Parent: Info & Optional Link Code */}
                        {selectedRole === 'parent' && (
                            <div className="space-y-3">
                                <div className="p-3 bg-orange-50 rounded-lg border border-orange-200 text-sm text-orange-700">
                                    <p className="font-medium mb-1">👨‍👩‍👧‍👦 Parent Account</p>
                                    <p>You can link your children after signup from your dashboard. Each child has a unique code in their Settings.</p>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Student Link Code (Optional)</label>
                                    <input
                                        type="text"
                                        value={studentLinkCode}
                                        onChange={(e) => setStudentLinkCode(e.target.value.toUpperCase())}
                                        className="input-field"
                                        placeholder="Link now or add later from dashboard"
                                    />
                                    <p className="text-xs text-gray-500 mt-1">You can add multiple children from Settings after signup.</p>
                                </div>
                            </div>
                        )}

                        {/* Name Fields */}
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                                <input type="text" value={firstName} onChange={(e) => setFirstName(e.target.value)} className="input-field" />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                                <input type="text" value={lastName} onChange={(e) => setLastName(e.target.value)} className="input-field" />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
                            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="input-field" required />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Username *</label>
                            <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} className="input-field" required />
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Password *</label>
                                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="input-field" required />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Confirm *</label>
                                <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="input-field" required />
                            </div>
                        </div>

                        {/* Student: Profile */}
                        {selectedRole === 'student' && (
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Grade</label>
                                    <select value={gradeLevel} onChange={(e) => setGradeLevel(e.target.value)} className="input-field">
                                        <option value="">Select</option>
                                        {['9', '10', '11', '12'].map(g => <option key={g} value={g}>{g}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Board</label>
                                    <select value={board} onChange={(e) => setBoard(e.target.value)} className="input-field">
                                        <option value="">Select</option>
                                        {['CBSE', 'ICSE', 'State Board', 'IB'].map(b => <option key={b} value={b}>{b}</option>)}
                                    </select>
                                </div>
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading || (selectedRole === 'teacher' && tokenValid !== true)}
                            className="w-full btn-primary flex items-center justify-center gap-2"
                        >
                            {loading ? <div className="spinner"></div> : <>Create Account <ArrowRightIcon className="w-5 h-5" /></>}
                        </button>
                    </form>

                    {/* Footer */}
                    <p className="text-center text-sm text-gray-500 mt-6">
                        Already have an account?{' '}
                        <Link href={`/auth/signin?role=${selectedRole}`} className="text-primary-600 font-medium hover:underline">
                            Sign in
                        </Link>
                    </p>
                </div>
            </div>
        </main>
    )
}
