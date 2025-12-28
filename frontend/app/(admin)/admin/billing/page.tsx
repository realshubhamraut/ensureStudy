'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect } from 'react'
import {
    TicketIcon,
    CreditCardIcon,
    CheckCircleIcon,
    ClockIcon
} from '@heroicons/react/24/outline'

interface Purchase {
    id: string
    quantity: number
    total_amount: number
    payment_status: string
    created_at: string
}

export default function BillingPage() {
    const { data: session } = useSession()
    const [licenseCount, setLicenseCount] = useState(0)
    const [usedLicenses, setUsedLicenses] = useState(0)
    const [purchases, setPurchases] = useState<Purchase[]>([])
    const [quantity, setQuantity] = useState(10)
    const [loading, setLoading] = useState(true)
    const [purchasing, setPurchasing] = useState(false)

    const PRICE_PER_STUDENT = 29 // INR

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Fetch dashboard for license info
                const dashRes = await fetch('http://localhost:8000/api/admin/dashboard', {
                    headers: {
                        'Authorization': `Bearer ${(session as any)?.accessToken}`
                    }
                })
                if (dashRes.ok) {
                    const data = await dashRes.json()
                    setLicenseCount(data.stats.license_count)
                    setUsedLicenses(data.stats.used_licenses)
                }

                // Fetch purchase history
                const histRes = await fetch('http://localhost:8000/api/admin/licenses/history', {
                    headers: {
                        'Authorization': `Bearer ${(session as any)?.accessToken}`
                    }
                })
                if (histRes.ok) {
                    const data = await histRes.json()
                    setPurchases(data.purchases)
                }
            } catch (error) {
                console.error('Failed to fetch billing data:', error)
            } finally {
                setLoading(false)
            }
        }

        if (session) {
            fetchData()
        }
    }, [session])

    const handlePurchase = async () => {
        setPurchasing(true)
        try {
            // Step 1: Initiate purchase
            const initRes = await fetch('http://localhost:8000/api/admin/licenses/purchase', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${(session as any)?.accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ quantity })
            })

            if (!initRes.ok) {
                throw new Error('Failed to initiate purchase')
            }

            const initData = await initRes.json()

            // Step 2: In production, open Razorpay checkout here
            // For demo, we'll simulate payment confirmation
            const confirmRes = await fetch('http://localhost:8000/api/admin/licenses/confirm', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${(session as any)?.accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    purchase_id: initData.purchase_id,
                    payment_id: 'demo_' + Date.now()
                })
            })

            if (confirmRes.ok) {
                const confirmData = await confirmRes.json()
                setLicenseCount(prev => prev + quantity)
                alert(`Successfully purchased ${quantity} licenses!`)
                // Refresh page
                window.location.reload()
            }
        } catch (error) {
            console.error('Purchase failed:', error)
            alert('Purchase failed. Please try again.')
        } finally {
            setPurchasing(false)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="spinner"></div>
            </div>
        )
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900">Billing & Licenses</h1>
                <p className="text-gray-600">Manage your student licenses</p>
            </div>

            {/* Current License Status */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="card bg-gradient-to-br from-blue-500 to-cyan-500 text-white">
                    <TicketIcon className="w-10 h-10 opacity-80 mb-4" />
                    <p className="text-blue-100">Total Licenses</p>
                    <p className="text-4xl font-bold">{licenseCount}</p>
                </div>

                <div className="card bg-gradient-to-br from-green-500 to-emerald-500 text-white">
                    <CheckCircleIcon className="w-10 h-10 opacity-80 mb-4" />
                    <p className="text-green-100">Used</p>
                    <p className="text-4xl font-bold">{usedLicenses}</p>
                </div>

                <div className="card bg-gradient-to-br from-purple-500 to-pink-500 text-white">
                    <ClockIcon className="w-10 h-10 opacity-80 mb-4" />
                    <p className="text-purple-100">Available</p>
                    <p className="text-4xl font-bold">{licenseCount - usedLicenses}</p>
                </div>
            </div>

            {/* Purchase Licenses */}
            <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-6">Purchase More Licenses</h2>

                <div className="flex flex-col md:flex-row gap-6">
                    <div className="flex-1">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Number of Licenses
                        </label>
                        <div className="flex items-center gap-4">
                            <input
                                type="number"
                                min="1"
                                value={quantity}
                                onChange={(e) => setQuantity(Math.max(1, Number(e.target.value)))}
                                className="input-field w-32"
                            />
                            <div className="flex gap-2">
                                {[10, 25, 50, 100].map(num => (
                                    <button
                                        key={num}
                                        onClick={() => setQuantity(num)}
                                        className={`px-3 py-1 rounded-lg text-sm ${quantity === num
                                                ? 'bg-primary-600 text-white'
                                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                    >
                                        {num}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="flex-1 bg-gray-50 rounded-xl p-6">
                        <div className="flex justify-between mb-2">
                            <span className="text-gray-600">Price per student</span>
                            <span className="font-medium">₹{PRICE_PER_STUDENT}</span>
                        </div>
                        <div className="flex justify-between mb-2">
                            <span className="text-gray-600">Quantity</span>
                            <span className="font-medium">{quantity}</span>
                        </div>
                        <hr className="my-3" />
                        <div className="flex justify-between text-lg">
                            <span className="font-semibold">Total</span>
                            <span className="font-bold text-primary-600">
                                ₹{(quantity * PRICE_PER_STUDENT).toLocaleString()}
                            </span>
                        </div>

                        <button
                            onClick={handlePurchase}
                            disabled={purchasing}
                            className="w-full mt-4 btn-primary flex items-center justify-center gap-2"
                        >
                            {purchasing ? (
                                <div className="spinner w-5 h-5 border-white"></div>
                            ) : (
                                <>
                                    <CreditCardIcon className="w-5 h-5" />
                                    Pay Now
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Purchase History */}
            <div className="card">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Purchase History</h2>

                {purchases.length === 0 ? (
                    <p className="text-gray-500 text-center py-8">No purchases yet</p>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead>
                                <tr>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Licenses</th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200">
                                {purchases.map((purchase) => (
                                    <tr key={purchase.id}>
                                        <td className="px-4 py-3 text-sm">
                                            {new Date(purchase.created_at).toLocaleDateString()}
                                        </td>
                                        <td className="px-4 py-3 text-sm font-medium">
                                            {purchase.quantity}
                                        </td>
                                        <td className="px-4 py-3 text-sm">
                                            ₹{purchase.total_amount.toLocaleString()}
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 text-xs rounded-full ${purchase.payment_status === 'completed'
                                                    ? 'bg-green-100 text-green-700'
                                                    : 'bg-yellow-100 text-yellow-700'
                                                }`}>
                                                {purchase.payment_status}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    )
}
