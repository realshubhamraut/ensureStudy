'use client'

import Link from 'next/link'
import {
    AcademicCapIcon,
    CheckCircleIcon,
    CurrencyRupeeIcon
} from '@heroicons/react/24/outline'

const plans = [
    {
        name: 'Trial',
        price: 'Free',
        period: '',
        description: 'Get started with 5 free student licenses',
        features: [
            '5 student licenses',
            'Unlimited teachers',
            'AI-powered assessments',
            'Basic analytics',
            'Email support'
        ],
        cta: 'Start Free Trial',
        ctaLink: '/auth/signup?role=admin',
        popular: false
    },
    {
        name: 'School',
        price: '₹29',
        period: '/student/year',
        description: 'Best value for schools and coaching centers',
        features: [
            'Unlimited student licenses',
            'Unlimited teachers',
            'AI quiz generation',
            'Advanced analytics',
            'Parent portal access',
            'Priority support',
            'Custom branding'
        ],
        cta: 'Contact Sales',
        ctaLink: 'mailto:sales@ensurestudy.com',
        popular: true
    },
    {
        name: 'Enterprise',
        price: 'Custom',
        period: '',
        description: 'For large institutions with custom needs',
        features: [
            'Everything in School',
            'Dedicated account manager',
            'SLA guarantee',
            'API access',
            'On-premise option',
            'Custom integrations'
        ],
        cta: 'Contact Us',
        ctaLink: 'mailto:enterprise@ensurestudy.com',
        popular: false
    }
]

export default function PricingPage() {
    return (
        <main className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
            {/* Header */}
            <div className="py-8 px-4">
                <div className="max-w-6xl mx-auto">
                    <Link href="/" className="inline-flex items-center gap-2">
                        <AcademicCapIcon className="w-8 h-8 text-primary-600" />
                        <span className="text-2xl font-bold gradient-text">ensureStudy</span>
                    </Link>
                </div>
            </div>

            {/* Pricing Section */}
            <div className="max-w-6xl mx-auto px-4 pb-16">
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-gray-900 mb-4">
                        Simple, Transparent Pricing
                    </h1>
                    <p className="text-xl text-gray-600">
                        Only pay for students. Teachers and parents are always free.
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-8">
                    {plans.map((plan) => (
                        <div
                            key={plan.name}
                            className={`relative card ${plan.popular ? 'ring-2 ring-primary-500' : ''}`}
                        >
                            {plan.popular && (
                                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                                    <span className="bg-primary-600 text-white text-sm font-medium px-4 py-1 rounded-full">
                                        Most Popular
                                    </span>
                                </div>
                            )}

                            <div className="text-center mb-6">
                                <h2 className="text-xl font-semibold text-gray-900">{plan.name}</h2>
                                <div className="mt-4">
                                    <span className="text-4xl font-bold text-gray-900">{plan.price}</span>
                                    <span className="text-gray-500">{plan.period}</span>
                                </div>
                                <p className="text-gray-600 mt-2 text-sm">{plan.description}</p>
                            </div>

                            <ul className="space-y-3 mb-8">
                                {plan.features.map((feature) => (
                                    <li key={feature} className="flex items-center gap-2 text-sm">
                                        <CheckCircleIcon className="w-5 h-5 text-green-500 flex-shrink-0" />
                                        <span className="text-gray-700">{feature}</span>
                                    </li>
                                ))}
                            </ul>

                            <Link
                                href={plan.ctaLink}
                                className={`block w-full text-center py-3 rounded-xl font-medium transition-colors ${plan.popular
                                        ? 'bg-primary-600 text-white hover:bg-primary-700'
                                        : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                                    }`}
                            >
                                {plan.cta}
                            </Link>
                        </div>
                    ))}
                </div>

                {/* FAQ */}
                <div className="mt-16 text-center">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Questions?</h2>
                    <p className="text-gray-600">
                        Contact us at{' '}
                        <a href="mailto:support@ensurestudy.com" className="text-primary-600 hover:underline">
                            support@ensurestudy.com
                        </a>
                    </p>
                </div>
            </div>
        </main>
    )
}
