'use client'

import { useState, useEffect } from 'react'
import {
    DocumentTextIcon,
    ArrowDownTrayIcon,
    CalendarIcon
} from '@heroicons/react/24/outline'

export default function ParentReportsPage() {
    const [loading, setLoading] = useState(true)
    const [reports, setReports] = useState<any[]>([])

    useEffect(() => {
        // Simulate loading
        setTimeout(() => setLoading(false), 500)
    }, [])

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900">📄 Reports</h1>
                <p className="text-gray-600">Download progress reports for your children</p>
            </div>

            {reports.length === 0 ? (
                <div className="card text-center py-12">
                    <DocumentTextIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No Reports Available</h3>
                    <p className="text-gray-500 max-w-md mx-auto">
                        Reports will be generated as your children complete assessments and make progress.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {reports.map((report, index) => (
                        <div key={index} className="card hover:shadow-lg transition-shadow">
                            <div className="flex items-start gap-4">
                                <div className="p-3 rounded-lg bg-orange-100">
                                    <DocumentTextIcon className="w-6 h-6 text-orange-600" />
                                </div>
                                <div className="flex-1">
                                    <h3 className="font-semibold text-gray-900">{report.title}</h3>
                                    <div className="flex items-center gap-2 text-sm text-gray-500 mt-1">
                                        <CalendarIcon className="w-4 h-4" />
                                        <span>{report.date}</span>
                                    </div>
                                </div>
                                <button className="p-2 text-orange-600 hover:bg-orange-50 rounded-lg transition-colors">
                                    <ArrowDownTrayIcon className="w-5 h-5" />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Coming Soon */}
            <div className="card bg-gradient-to-br from-orange-50 to-amber-50 border-orange-200">
                <h3 className="font-semibold text-orange-900 mb-2">📈 Coming Soon</h3>
                <ul className="text-sm text-orange-700 space-y-1">
                    <li>• Weekly progress summaries</li>
                    <li>• Monthly achievement reports</li>
                    <li>• Subject-wise analysis</li>
                    <li>• Downloadable PDF reports</li>
                </ul>
            </div>
        </div>
    )
}
