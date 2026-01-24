'use client'
import { getApiBaseUrl } from '@/utils/api'


import { useState } from 'react'
import { signIn } from 'next-auth/react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import {
    AcademicCapIcon,
    BuildingOfficeIcon,
    UsersIcon,
    UserIcon,
    ArrowRightIcon
} from '@heroicons/react/24/outline'

type Role = 'admin' | 'teacher' | 'student' | 'parent'

const roles = [
    { id: 'admin' as Role, name: 'School', icon: BuildingOfficeIcon },
    { id: 'teacher' as Role, name: 'Teacher', icon: UsersIcon },
    { id: 'student' as Role, name: 'Student', icon: AcademicCapIcon },
    { id: 'parent' as Role, name: 'Parent', icon: UserIcon },
]

export default function SignInPage() {
    const router = useRouter()
    const searchParams = useSearchParams()
    const initialRole = (searchParams.get('role') as Role) || 'student'

    const [selectedRole, setSelectedRole] = useState<Role>(initialRole)
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')
        setLoading(true)

        try {
            // Authenticate with backend API
            const loginUrl = `${getApiBaseUrl()}/api/auth/login`
            console.log('[Login] Calling API:', loginUrl)
            console.log('[Login] From:', window.location.origin)

            const res = await fetch(loginUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            })

            const data = await res.json()

            if (!res.ok) {
                throw new Error(data.error || 'Login failed')
            }

            // Store token and user info
            localStorage.setItem('accessToken', data.access_token)

            // Store user info for role-based features (like recording)
            if (data.user) {
                localStorage.setItem('userId', data.user.id)
                localStorage.setItem('userRole', data.user.role || selectedRole)
                localStorage.setItem('userName', data.user.name || email.split('@')[0])
                console.log('[Auth] Stored user info:', {
                    userId: data.user.id,
                    role: data.user.role || selectedRole,
                    name: data.user.name
                })
            } else {
                // Fallback: use selected role
                localStorage.setItem('userRole', selectedRole)
                console.log('[Auth] No user in response, using selected role:', selectedRole)
            }

            // Sign in with NextAuth
            const result = await signIn('credentials', {
                email,
                password,
                redirect: false
            })

            if (result?.error) {
                throw new Error(result.error)
            }

            // Redirect based on role
            const userRole = data.user?.role || selectedRole
            switch (userRole) {
                case 'admin':
                    router.push('/admin/dashboard')
                    break
                case 'teacher':
                    router.push('/teacher/dashboard')
                    break
                case 'parent':
                    router.push('/parent/dashboard')
                    break
                default:
                    router.push('/dashboard')
            }
        } catch (err: any) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-white to-secondary-50 p-4">
            <div className="w-full max-w-md">
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

                    {/* Sign In Heading */}
                    <h1 className="text-xl font-bold text-gray-900 text-center mb-6">
                        Sign in as {roles.find(r => r.id === selectedRole)?.name}
                    </h1>

                    {error && (
                        <div className="p-3 bg-red-50 text-red-700 rounded-lg mb-4 text-sm">
                            {error}
                        </div>
                    )}

                    {/* Login Form */}
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Email
                            </label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="input-field"
                                placeholder="you@example.com"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Password
                            </label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="input-field"
                                placeholder="••••••••"
                                required
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full btn-primary flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <div className="spinner"></div>
                            ) : (
                                <>
                                    Sign In
                                    <ArrowRightIcon className="w-5 h-5" />
                                </>
                            )}
                        </button>
                    </form>

                    {/* Footer */}
                    <p className="text-center text-sm text-gray-500 mt-6">
                        Don't have an account?{' '}
                        <Link
                            href={`/auth/signup?role=${selectedRole}`}
                            className="text-primary-600 font-medium hover:underline"
                        >
                            Create one
                        </Link>
                    </p>
                </div>
            </div>
        </main>
    )
}
