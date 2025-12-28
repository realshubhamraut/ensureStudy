import type { Metadata } from 'next'
import { Inter, Arimo } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/Providers'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const arimo = Arimo({ subsets: ['latin'], variable: '--font-arimo' })

export const metadata: Metadata = {
    title: 'ensureStudy - AI Learning Platform',
    description: 'AI-first learning platform with RAG, adaptive assessments, and personalized study plans',
    keywords: ['education', 'AI tutor', 'learning', 'study', 'assessments'],
    icons: {
        icon: '/favicon.svg',
        apple: '/apple-touch-icon.svg',
    },
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body className={`${inter.variable} ${arimo.variable} font-sans antialiased`}>
                <Providers>
                    {children}
                    <Toaster
                        position="top-right"
                        toastOptions={{
                            duration: 4000,
                            style: {
                                background: '#333',
                                color: '#fff',
                            },
                        }}
                    />
                </Providers>
            </body>
        </html>
    )
}

