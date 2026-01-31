'use client'

/**
 * Chat page loading skeleton
 * Shows instantly while the main chat page compiles
 */
export default function ChatLoading() {
    return (
        <div className="h-screen flex bg-gray-50 animate-pulse">
            {/* Left sidebar skeleton */}
            <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
                {/* Header */}
                <div className="p-4 border-b border-gray-100">
                    <div className="h-10 bg-gray-100 rounded-lg" />
                </div>

                {/* Conversation list skeleton */}
                <div className="flex-1 p-3 space-y-2">
                    {[1, 2, 3, 4, 5].map((i) => (
                        <div key={i} className="p-3 rounded-lg bg-gray-50">
                            <div className="h-4 bg-gray-100 rounded w-3/4 mb-2" />
                            <div className="h-3 bg-gray-100 rounded w-1/2" />
                        </div>
                    ))}
                </div>
            </div>

            {/* Main content area skeleton */}
            <div className="flex-1 flex flex-col">
                {/* Chat header */}
                <div className="h-14 border-b border-gray-200 bg-white flex items-center px-4">
                    <div className="h-6 bg-gray-100 rounded w-48" />
                </div>

                {/* Messages area */}
                <div className="flex-1 p-6 space-y-4 overflow-hidden">
                    {/* Welcome message skeleton */}
                    <div className="max-w-2xl mx-auto text-center mt-20">
                        <div className="w-16 h-16 bg-gradient-to-br from-purple-100 to-blue-100 rounded-2xl mx-auto mb-4" />
                        <div className="h-8 bg-gray-100 rounded-lg w-48 mx-auto mb-2" />
                        <div className="h-4 bg-gray-100 rounded w-64 mx-auto" />
                    </div>
                </div>

                {/* Input area skeleton */}
                <div className="p-4 bg-white border-t border-gray-200">
                    <div className="max-w-3xl mx-auto">
                        <div className="h-12 bg-gray-100 rounded-xl" />
                    </div>
                </div>
            </div>

            {/* Right sidebar skeleton (resources) */}
            <div className="w-80 bg-white border-l border-gray-200 hidden lg:block">
                <div className="p-4 border-b border-gray-100">
                    <div className="h-6 bg-gray-100 rounded w-24" />
                </div>
                <div className="p-4 space-y-3">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="p-3 rounded-lg bg-gray-50">
                            <div className="h-4 bg-gray-100 rounded w-full mb-2" />
                            <div className="h-3 bg-gray-100 rounded w-2/3" />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
