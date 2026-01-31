/**
 * useBehaviorAnalysis Hook
 * 
 * Provides real-time behavior analysis for softskills sessions.
 * Tracks engagement, attention, and generates behavioral reports.
 */

import { useState, useCallback, useRef, useEffect } from 'react'

const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001'

interface BehaviorMetrics {
    engagementScore: number
    attentionScore: number
    consistencyScore: number
    eyeContactAvg: number
    gazeStability: number
    postureConsistency: number
    gestureExpressiveness: number
    focusDurationRatio: number
    distractionEvents: number
    recoverySpeed: number
}

interface TemporalMetrics {
    currentAttention: number
    avgAttention: number
    attentionStability: number
    attentionTrend: 'improving' | 'stable' | 'declining' | 'insufficient_data'
    focusRatio: number
    distractionEvents: number
    avgRecoveryTime: number
}

interface BehaviorReport {
    sessionId: string
    durationSeconds: number
    totalFrames: number
    metrics: BehaviorMetrics
    overallBehaviorScore: number
    engagementLevel: 'high' | 'medium' | 'low'
    attentionPattern: 'consistent' | 'variable' | 'declining'
    topStrengths: string[]
    areasForImprovement: string[]
    timestamp: string
}

interface FrameAnalysisResult {
    frameNumber: number
    attentionScore: number
    isFocused: boolean
    currentWarnings: string[]
    consecutiveLowAttention: number
    distractionEvents: number
}

interface UseBehaviorAnalysis {
    // State
    isActive: boolean
    framesAnalyzed: number
    currentAttention: number
    warnings: string[]
    temporalMetrics: TemporalMetrics | null

    // Actions
    startSession: (sessionId: string) => void
    analyzeFrame: (
        gazeScore: number,
        postureScore: number,
        gestureScore: number,
        isLookingAtCamera: boolean,
        isUpright: boolean,
        handsVisible: boolean
    ) => Promise<FrameAnalysisResult | null>
    getTemporalMetrics: () => Promise<TemporalMetrics | null>
    getReport: () => Promise<BehaviorReport | null>
    endSession: () => Promise<BehaviorReport | null>
}

export function useBehaviorAnalysis(): UseBehaviorAnalysis {
    const [isActive, setIsActive] = useState(false)
    const [framesAnalyzed, setFramesAnalyzed] = useState(0)
    const [currentAttention, setCurrentAttention] = useState(75)
    const [warnings, setWarnings] = useState<string[]>([])
    const [temporalMetrics, setTemporalMetrics] = useState<TemporalMetrics | null>(null)

    const sessionIdRef = useRef<string>('')

    // Start tracking session
    const startSession = useCallback((sessionId: string) => {
        sessionIdRef.current = sessionId
        setIsActive(true)
        setFramesAnalyzed(0)
        setCurrentAttention(75)
        setWarnings([])
        setTemporalMetrics(null)
        console.log('[BehaviorAnalysis] Session started:', sessionId)
    }, [])

    // Analyze a single frame
    const analyzeFrame = useCallback(async (
        gazeScore: number,
        postureScore: number,
        gestureScore: number,
        isLookingAtCamera: boolean,
        isUpright: boolean,
        handsVisible: boolean
    ): Promise<FrameAnalysisResult | null> => {
        if (!isActive || !sessionIdRef.current) return null

        try {
            const response = await fetch(`${AI_SERVICE_URL}/api/softskills/behavior/analyze-frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionIdRef.current,
                    gaze_score: gazeScore,
                    posture_score: postureScore,
                    gesture_score: gestureScore,
                    is_looking_at_camera: isLookingAtCamera,
                    is_upright: isUpright,
                    hands_visible: handsVisible
                })
            })

            if (response.ok) {
                const result = await response.json()

                setFramesAnalyzed(result.frame_number)
                setCurrentAttention(result.attention_score)
                setWarnings(result.current_warnings || [])

                return {
                    frameNumber: result.frame_number,
                    attentionScore: result.attention_score,
                    isFocused: result.is_focused,
                    currentWarnings: result.current_warnings || [],
                    consecutiveLowAttention: result.consecutive_low_attention,
                    distractionEvents: result.distraction_events
                }
            }
        } catch (error) {
            console.error('[BehaviorAnalysis] Frame analysis error:', error)
        }

        return null
    }, [isActive])

    // Get temporal metrics
    const getTemporalMetrics = useCallback(async (): Promise<TemporalMetrics | null> => {
        if (!sessionIdRef.current) return null

        try {
            const response = await fetch(
                `${AI_SERVICE_URL}/api/softskills/behavior/temporal-metrics/${sessionIdRef.current}`
            )

            if (response.ok) {
                const data = await response.json()

                const metrics: TemporalMetrics = {
                    currentAttention: data.current_attention || 0,
                    avgAttention: data.avg_attention || 0,
                    attentionStability: data.attention_stability || 0,
                    attentionTrend: data.attention_trend || 'insufficient_data',
                    focusRatio: data.focus_ratio || 0,
                    distractionEvents: data.distraction_events || 0,
                    avgRecoveryTime: data.avg_recovery_time || 0
                }

                setTemporalMetrics(metrics)
                return metrics
            }
        } catch (error) {
            console.error('[BehaviorAnalysis] Temporal metrics error:', error)
        }

        return null
    }, [])

    // Get behavior report
    const getReport = useCallback(async (): Promise<BehaviorReport | null> => {
        if (!sessionIdRef.current) return null

        try {
            const response = await fetch(
                `${AI_SERVICE_URL}/api/softskills/behavior/report/${sessionIdRef.current}`
            )

            if (response.ok) {
                const data = await response.json()

                return {
                    sessionId: data.session_id,
                    durationSeconds: data.duration_seconds,
                    totalFrames: data.total_frames,
                    metrics: {
                        engagementScore: data.metrics.engagement_score,
                        attentionScore: data.metrics.attention_score,
                        consistencyScore: data.metrics.consistency_score,
                        eyeContactAvg: data.metrics.eye_contact_avg,
                        gazeStability: data.metrics.gaze_stability,
                        postureConsistency: data.metrics.posture_consistency,
                        gestureExpressiveness: data.metrics.gesture_expressiveness,
                        focusDurationRatio: data.metrics.focus_duration_ratio,
                        distractionEvents: data.metrics.distraction_events,
                        recoverySpeed: data.metrics.recovery_speed
                    },
                    overallBehaviorScore: data.overall_behavior_score,
                    engagementLevel: data.engagement_level,
                    attentionPattern: data.attention_pattern,
                    topStrengths: data.top_strengths,
                    areasForImprovement: data.areas_for_improvement,
                    timestamp: data.timestamp
                }
            }
        } catch (error) {
            console.error('[BehaviorAnalysis] Report error:', error)
        }

        return null
    }, [])

    // End session
    const endSession = useCallback(async (): Promise<BehaviorReport | null> => {
        if (!sessionIdRef.current) return null

        try {
            const response = await fetch(
                `${AI_SERVICE_URL}/api/softskills/behavior/end-session/${sessionIdRef.current}`,
                { method: 'POST' }
            )

            if (response.ok) {
                const data = await response.json()

                setIsActive(false)

                if (data.report) {
                    return {
                        sessionId: data.report.session_id,
                        durationSeconds: data.report.duration_seconds,
                        totalFrames: data.report.total_frames,
                        metrics: {
                            engagementScore: data.report.metrics.engagement_score,
                            attentionScore: data.report.metrics.attention_score,
                            consistencyScore: data.report.metrics.consistency_score,
                            eyeContactAvg: data.report.metrics.eye_contact_avg,
                            gazeStability: data.report.metrics.gaze_stability,
                            postureConsistency: data.report.metrics.posture_consistency,
                            gestureExpressiveness: data.report.metrics.gesture_expressiveness,
                            focusDurationRatio: data.report.metrics.focus_duration_ratio,
                            distractionEvents: data.report.metrics.distraction_events,
                            recoverySpeed: data.report.metrics.recovery_speed
                        },
                        overallBehaviorScore: data.report.overall_behavior_score,
                        engagementLevel: data.report.engagement_level,
                        attentionPattern: data.report.attention_pattern,
                        topStrengths: data.report.top_strengths,
                        areasForImprovement: data.report.areas_for_improvement,
                        timestamp: data.report.timestamp
                    }
                }
            }
        } catch (error) {
            console.error('[BehaviorAnalysis] End session error:', error)
        }

        setIsActive(false)
        return null
    }, [])

    return {
        isActive,
        framesAnalyzed,
        currentAttention,
        warnings,
        temporalMetrics,
        startSession,
        analyzeFrame,
        getTemporalMetrics,
        getReport,
        endSession
    }
}

export default useBehaviorAnalysis
