'use client'

import Link from 'next/link'
import {
    AcademicCapIcon,
    ChatBubbleLeftRightIcon,
    ChartBarIcon,
    BookOpenIcon,
    TrophyIcon,
    SparklesIcon,
    ArrowRightIcon
} from '@heroicons/react/24/outline'

export default function Home() {
    return (
        <main className="min-h-screen bg-[#FDFBF7] relative noise-bg">
            {/* Grain texture overlay - GPU accelerated */}
            <div
                className="fixed inset-0 pointer-events-none opacity-[0.15]"
                style={{
                    backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.5' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
                    transform: 'translateZ(0)',
                    backfaceVisibility: 'hidden',
                }}
            />

            {/* Navigation - Sticky Header */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-[#FDFBF7]">
                <div className="w-full h-px bg-gray-300 absolute bottom-0 left-0" />
                <div className="flex items-center justify-between px-8 py-5 max-w-6xl mx-auto">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-600 to-violet-600 flex items-center justify-center">
                            <AcademicCapIcon className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-xl font-semibold text-gray-900">ensureStudy</span>
                    </div>
                    <div className="flex items-center gap-6">
                        <Link
                            href="#features"
                            className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors"
                        >
                            Features
                        </Link>
                        <Link
                            href="/auth/signup"
                            className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors"
                        >
                            Sign Up
                        </Link>
                        <Link
                            href="/auth/signin"
                            className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors shadow-sm"
                        >
                            Sign In
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative z-10 px-8 pt-32 pb-24 max-w-6xl mx-auto">
                <div className="max-w-3xl">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-emerald-100/80 border-2 border-emerald-200 rounded-full mb-6">
                        <SparklesIcon className="w-4 h-4 text-emerald-600" />
                        <span className="text-emerald-800 text-sm font-medium">AI-Powered Education Platform</span>
                    </div>

                    <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
                        Learn smarter with
                        <br />
                        <span className="text-indigo-600">AI-powered tutoring</span>
                    </h1>

                    <p className="text-xl text-gray-600 mb-10 max-w-2xl leading-relaxed">
                        AI tutoring, automated answer evaluation, and personalized learning paths.
                        Everything teachers and students need in one place.
                    </p>

                    <div className="flex flex-col sm:flex-row items-start gap-4">
                        <Link
                            href="/auth/signin"
                            className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors shadow-sm"
                        >
                            Get started free
                            <ArrowRightIcon className="w-4 h-4" />
                        </Link>
                        <Link
                            href="#features"
                            className="inline-flex items-center gap-2 px-6 py-3 text-gray-900 font-medium border-2 border-gray-900 rounded-lg bg-white/50 hover:bg-white/70 hover:border-black transition-colors"
                        >
                            See how it works
                        </Link>
                    </div>
                </div>

                {/* Decorative elements */}
                <div className="absolute top-20 right-20 w-72 h-72 bg-amber-100/50 rounded-full" />
                <div className="absolute bottom-10 right-40 w-48 h-48 bg-indigo-100/40 rounded-full" />
            </section>


            {/* Features Section */}
            <section id="features" className="relative z-10 py-20 px-8 max-w-6xl mx-auto">
                <div className="mb-16">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">
                        Everything you need to <span className="text-indigo-600">Excel</span>
                    </h2>
                    <p className="text-gray-600 max-w-2xl text-lg">
                        Our platform adapts to your learning style and helps you achieve your goals.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <FeatureCard
                        icon={<ChatBubbleLeftRightIcon className="w-6 h-6" />}
                        title="AI Tutor Chat"
                        description="Stuck on a problem? Ask our AI and get step-by-step explanations with sources from your textbooks."
                        color="indigo"
                    />
                    <FeatureCard
                        icon={<BookOpenIcon className="w-6 h-6" />}
                        title="Smart Answer Evaluation"
                        description="Submit handwritten answers. Our AI reads, scores, and gives feedback instantly."
                        color="violet"
                    />
                    <FeatureCard
                        icon={<ChartBarIcon className="w-6 h-6" />}
                        title="AI Progress Insights"
                        description="Every quiz is analyzed. AI spots your weak topics and shows exactly where to focus."
                        color="amber"
                    />
                    <FeatureCard
                        icon={<AcademicCapIcon className="w-6 h-6" />}
                        title="Adaptive Quizzes"
                        description="AI generates practice questions based on what you got wrong. Practice makes perfect."
                        color="emerald"
                    />
                    <FeatureCard
                        icon={<TrophyIcon className="w-6 h-6" />}
                        title="Virtual Classrooms"
                        description="Join classrooms, access materials, submit homework, and track grades—all in one place."
                        color="rose"
                    />
                    <FeatureCard
                        icon={<SparklesIcon className="w-6 h-6" />}
                        title="Personalized Learning"
                        description="AI builds your study plan based on performance, exams, and available time."
                        color="cyan"
                    />
                </div>
            </section>

            {/* Gumroad-style Separator */}
            <div className="relative z-10 w-full h-px bg-gray-300" />

            {/* CTA Section + Footer - Full Width */}
            <section className="relative z-10 py-16 bg-white overflow-hidden">
                {/* Decorative elements - no blur for performance */}
                <div className="absolute top-10 right-20 w-64 h-64 bg-indigo-50 rounded-full" />
                <div className="absolute -bottom-20 left-20 w-72 h-72 bg-amber-50 rounded-full" />

                <div className="max-w-6xl mx-auto px-8 relative">
                    <div className="max-w-2xl">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">
                            Ready to transform your learning?
                        </h2>
                        <p className="text-gray-600 text-lg mb-8">
                            Join thousands of students already using ensureStudy to ace their exams.
                        </p>
                        <Link
                            href="/auth/signup"
                            className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors shadow-sm"
                        >
                            Get started for free
                            <ArrowRightIcon className="w-4 h-4" />
                        </Link>
                    </div>
                </div>

                {/* Divider */}
                <div className="max-w-6xl mx-auto px-8 relative mt-16 mb-8">
                    <div className="w-full h-px bg-gray-300" />
                </div>

                {/* Footer */}
                <div className="max-w-6xl mx-auto px-8 relative">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-600 to-violet-600 flex items-center justify-center">
                                <AcademicCapIcon className="w-4 h-4 text-white" />
                            </div>
                            <span className="text-gray-900 font-semibold">ensureStudy</span>
                        </div>
                        <p className="text-gray-500 text-sm">© 2026 ensureStudy. All rights reserved.</p>
                    </div>
                </div>
            </section>
        </main>
    )
}

function FeatureCard({
    icon,
    title,
    description,
    color
}: {
    icon: React.ReactNode
    title: string
    description: string
    color: 'indigo' | 'violet' | 'amber' | 'emerald' | 'rose' | 'cyan'
}) {
    const colorStyles = {
        indigo: 'bg-indigo-100 text-indigo-600 group-hover:bg-indigo-200',
        violet: 'bg-violet-100 text-violet-600 group-hover:bg-violet-200',
        amber: 'bg-amber-100 text-amber-600 group-hover:bg-amber-200',
        emerald: 'bg-emerald-100 text-emerald-600 group-hover:bg-emerald-200',
        rose: 'bg-rose-100 text-rose-600 group-hover:bg-rose-200',
        cyan: 'bg-cyan-100 text-cyan-600 group-hover:bg-cyan-200',
    }

    return (
        <div className="group p-6 rounded-xl border border-gray-200 bg-white hover:shadow-lg hover:border-gray-300 transition-all">
            <div className={`w-11 h-11 rounded-xl ${colorStyles[color]} flex items-center justify-center mb-4 transition-colors`}>
                {icon}
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
            <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
        </div>
    )
}
