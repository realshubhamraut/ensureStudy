'use client'

import { redirect } from 'next/navigation'

export default function AuthPage() {
    // Default to signin page
    redirect('/auth/signin')
}
