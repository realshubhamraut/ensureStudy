'use client'

import Link from 'next/link'
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'

export default function AuthError() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 to-secondary-50">
            <div className="text-center p-8 bg-white rounded-2xl shadow-xl max-w-md">
                <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-red-100 flex items-center justify-center">
                    <ExclamationTriangleIcon className="w-8 h-8 text-red-500" />
                </div>
                <h1 className="text-2xl font-bold text-gray-900 mb-2">Authentication Error</h1>
                <p className="text-gray-600 mb-6">
                    There was a problem signing you in. Please try again.
                </p>
                <div className="space-y-3">
                    <Link
                        href="/auth/signin"
                        className="block w-full px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium"
                    >
                        Try Again
                    </Link>
                    <Link
                        href="/"
                        className="block w-full px-6 py-3 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                        Go Home
                    </Link>
                </div>
            </div>
        </div>
    )
}
