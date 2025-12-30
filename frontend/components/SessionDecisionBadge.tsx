'use client';

import React from 'react';

/**
 * Topic Anchor metadata from API response
 */
interface TopicAnchor {
    id: string;
    canonical_title: string;
    subject_scope?: string[];
    source?: string;
}

/**
 * Session Decision Badge Props
 */
interface SessionDecisionBadgeProps {
    /** Decision from API response: "related" or "new_topic" */
    decision: 'related' | 'new_topic';
    /** Max similarity score (0-1) */
    maxSimilarity?: number;
    /** Active topic anchor from TAL */
    topicAnchor?: TopicAnchor | null;
    /** Callback when user clicks "Continue previous topic" */
    onForceSessionPrioritize?: () => void;
    /** Callback when user clicks "Start new topic" */
    onResetTopic?: () => void;
    /** Whether to show the override button for new_topic */
    showOverrideButton?: boolean;
}

/**
 * SessionDecisionBadge - Displays context routing decision with TAL info
 * 
 * Shows whether the system continued previous context or started fresh.
 * Displays active topic anchor when available.
 * Provides override buttons for topic management.
 * 
 * Usage:
 * ```tsx
 * <SessionDecisionBadge
 *   decision={response.session_context_decision}
 *   topicAnchor={response.active_topic_anchor}
 *   maxSimilarity={response.max_similarity}
 *   onForceSessionPrioritize={() => retryQuery(true)}
 *   onResetTopic={() => resetTopic()}
 * />
 * ```
 */
export function SessionDecisionBadge({
    decision,
    maxSimilarity,
    topicAnchor,
    onForceSessionPrioritize,
    onResetTopic,
    showOverrideButton = true,
}: SessionDecisionBadgeProps) {
    if (decision === 'related' && topicAnchor) {
        return (
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-full text-xs font-medium text-blue-700">
                <span className="text-blue-500">🔗</span>
                <span>Continuing: <strong>{topicAnchor.canonical_title}</strong></span>
                {maxSimilarity !== undefined && (
                    <span className="opacity-70 text-[10px]">({(maxSimilarity * 100).toFixed(0)}%)</span>
                )}
                {onResetTopic && (
                    <button
                        onClick={onResetTopic}
                        className="ml-1 px-1.5 py-0.5 text-[10px] bg-white border border-blue-300 rounded hover:bg-blue-50 transition-colors"
                        title="End this topic and start fresh"
                    >
                        End topic
                    </button>
                )}
            </div>
        );
    }

    if (decision === 'related') {
        return (
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-full text-xs font-medium text-blue-700">
                <span className="text-blue-500">🔗</span>
                <span>Continuing previous topic</span>
                {maxSimilarity !== undefined && (
                    <span className="opacity-70 text-[10px]">({(maxSimilarity * 100).toFixed(0)}% match)</span>
                )}
            </div>
        );
    }

    // New topic
    return (
        <div className="inline-flex flex-col gap-1.5 px-3 py-2 bg-gradient-to-r from-amber-50 to-yellow-100 border border-amber-200 rounded-lg text-xs font-medium text-amber-800">
            <div className="flex items-center gap-2">
                <span className="text-amber-500">🆕</span>
                <span>
                    {topicAnchor
                        ? <>New topic: <strong>{topicAnchor.canonical_title}</strong></>
                        : 'New topic started'}
                </span>
            </div>
            {showOverrideButton && onForceSessionPrioritize && (
                <button
                    onClick={onForceSessionPrioritize}
                    className="self-start px-2 py-1 text-[10px] bg-white border border-amber-400 rounded hover:bg-amber-50 transition-colors"
                    title="Re-run with previous session context"
                >
                    ← Continue previous topic instead
                </button>
            )}
        </div>
    );
}

/**
 * Inline component for minimal badge (just icon + tooltip)
 */
export function SessionDecisionIcon({
    decision,
    maxSimilarity,
}: Pick<SessionDecisionBadgeProps, 'decision' | 'maxSimilarity'>) {
    const tooltip = decision === 'related'
        ? `Continuing context (${((maxSimilarity || 0) * 100).toFixed(0)}% match)`
        : 'New topic started';

    return (
        <span
            className={`session-icon ${decision === 'related' ? 'related' : 'new-topic'}`}
            title={tooltip}
        >
            {decision === 'related' ? '🔗' : '🆕'}
        </span>
    );
}

/**
 * CSS for Session Decision Badges
 * Add to your global CSS or CSS module:
 * 
 * ```css
 * .session-decision-badge {
 *   display: inline-flex;
 *   align-items: center;
 *   gap: 0.5rem;
 *   padding: 0.375rem 0.75rem;
 *   border-radius: 9999px;
 *   font-size: 0.75rem;
 *   font-weight: 500;
 * }
 * 
 * .session-decision-related {
 *   background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
 *   color: #1e40af;
 *   border: 1px solid #93c5fd;
 * }
 * 
 * .session-decision-new-topic {
 *   background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
 *   color: #92400e;
 *   border: 1px solid #fcd34d;
 *   flex-direction: column;
 *   gap: 0.5rem;
 * }
 * 
 * .badge-content {
 *   display: flex;
 *   align-items: center;
 *   gap: 0.5rem;
 * }
 * 
 * .badge-score {
 *   opacity: 0.7;
 *   font-size: 0.625rem;
 * }
 * 
 * .override-button {
 *   padding: 0.25rem 0.5rem;
 *   background: white;
 *   border: 1px solid #d97706;
 *   border-radius: 0.25rem;
 *   font-size: 0.625rem;
 *   cursor: pointer;
 *   transition: all 0.2s;
 * }
 * 
 * .override-button:hover {
 *   background: #fef3c7;
 * }
 * 
 * .session-icon {
 *   cursor: help;
 * }
 * 
 * .session-icon.related {
 *   color: #3b82f6;
 * }
 * 
 * .session-icon.new-topic {
 *   color: #f59e0b;
 * }
 * ```
 */

// Export types for use in other components
export type { SessionDecisionBadgeProps };
