import Link from 'next/link'
import {
    AcademicCapIcon,
    ChatBubbleLeftRightIcon,
    ChartBarIcon,
    BookOpenIcon,
    TrophyIcon,
    SparklesIcon
} from '@heroicons/react/24/outline'

export default function Home() {
    return (
        <main className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
            {/* Hero Section */}
            <div className="relative overflow-hidden">
                <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

                <nav className="relative z-10 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
                    <div className="flex items-center gap-2">
                        <AcademicCapIcon className="w-8 h-8 text-primary-600" />
                        <span className="text-2xl font-bold gradient-text">ensureStudy</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link
                            href="/auth/signup"
                            className="text-gray-600 hover:text-primary-600 font-medium transition-colors"
                        >
                            Sign Up
                        </Link>
                        <Link
                            href="/auth/signin"
                            className="btn-primary"
                        >
                            Sign In
                        </Link>
                    </div>
                </nav>

                <div className="relative z-10 max-w-7xl mx-auto px-8 py-20 text-center">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-100 rounded-full mb-8">
                        <SparklesIcon className="w-5 h-5 text-primary-600" />
                        <span className="text-primary-700 font-medium">AI-Powered Education Platform</span>
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-8 leading-snug pb-2">
                        Your Complete
                        <span className="block gradient-text">Smart Classroom</span>
                    </h1>

                    <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-12">
                        AI-powered tutoring, automated answer evaluation,
                        and personalized learning paths. Everything teachers and students need in one place.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link
                            href="/auth/signin"
                            className="btn-primary text-lg px-8 py-3 flex items-center gap-2"
                        >
                            <SparklesIcon className="w-5 h-5" />
                            Start Learning Free
                        </Link>
                        <Link
                            href="#features"
                            className="btn-secondary text-lg px-8 py-3"
                        >
                            See Features
                        </Link>
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <section id="features" className="py-20 px-8 max-w-7xl mx-auto">
                <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
                    Everything You Need to <span className="gradient-text">Excel</span>
                </h2>
                <p className="text-gray-600 text-center max-w-2xl mx-auto mb-16">
                    Our AI-powered platform adapts to your learning style and helps you achieve your goals.
                </p>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {/* Feature Cards */}
                    <FeatureCard
                        icon={<ChatBubbleLeftRightIcon className="w-8 h-8" />}
                        title="AI Tutor Chat"
                        description="Stuck on a problem? Ask our AI and get step-by-step explanations with sources from your textbooks—instantly."
                        gradient="from-blue-500 to-cyan-500"
                    />
                    <FeatureCard
                        icon={<BookOpenIcon className="w-8 h-8" />}
                        title="Smart Answer Evaluation"
                        description="Submit handwritten answers. Our AI reads, scores, and gives feedback—so teachers review faster, students learn better."
                        gradient="from-purple-500 to-pink-500"
                    />
                    <FeatureCard
                        icon={<ChartBarIcon className="w-8 h-8" />}
                        title="AI Progress Insights"
                        description="Every quiz, every answer is analyzed. AI spots your weak topics before you do and shows exactly where to focus."
                        gradient="from-orange-500 to-red-500"
                    />
                    <FeatureCard
                        icon={<AcademicCapIcon className="w-8 h-8" />}
                        title="Adaptive Quizzes"
                        description="AI generates practice questions based on what you got wrong. The more you practice, the smarter it gets."
                        gradient="from-green-500 to-emerald-500"
                    />
                    <FeatureCard
                        icon={<TrophyIcon className="w-8 h-8" />}
                        title="Virtual Classrooms"
                        description="Join your teacher's classroom, access materials, submit homework, and track grades—all in one place."
                        gradient="from-yellow-500 to-orange-500"
                    />
                    <FeatureCard
                        icon={<SparklesIcon className="w-8 h-8" />}
                        title="Personalized Learning Path"
                        description="AI builds your study plan based on your performance, upcoming exams, and available time. Study smarter, not harder."
                        gradient="from-indigo-500 to-purple-500"
                    />
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-20 px-8 bg-gradient-to-r from-primary-600 to-secondary-600">
                <div className="max-w-4xl mx-auto text-center text-white">
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">
                        Ready to Transform Your Learning?
                    </h2>
                    <p className="text-xl opacity-90 mb-8">
                        Join thousands of students already using ensureStudy to ace their exams.
                    </p>
                    <Link
                        href="/auth/signup"
                        className="inline-flex items-center gap-2 px-8 py-4 bg-white text-primary-600 
                       rounded-lg font-bold text-lg hover:bg-gray-100 transition-colors"
                    >
                        <AcademicCapIcon className="w-6 h-6" />
                        Get Started for Free
                    </Link>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-12 px-8 bg-gray-900 text-gray-400">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <AcademicCapIcon className="w-6 h-6 text-primary-500" />
                        <span className="text-white font-bold">ensureStudy</span>
                    </div>
                    <p>© 2025 ensureStudy. All rights reserved.</p>
                </div>
            </footer>
        </main>
    )
}

function FeatureCard({
    icon,
    title,
    description,
    gradient
}: {
    icon: React.ReactNode
    title: string
    description: string
    gradient: string
}) {
    return (
        <div className="card-hover group">
            <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${gradient} 
                       flex items-center justify-center text-white mb-4
                       group-hover:scale-110 transition-transform duration-300`}>
                {icon}
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">{title}</h3>
            <p className="text-gray-600">{description}</p>
        </div>
    )
}
