'use client';

import React from 'react';
import {
    EyeIcon,
    HandRaisedIcon,
    UserIcon,
    MicrophoneIcon,
    CheckCircleIcon,
    ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

interface VisualMetrics {
    face_detected: boolean;
    gaze_direction: 'center' | 'left' | 'right' | 'unknown';
    gaze_score: number;
    is_looking_at_camera: boolean;
    hands_visible: boolean;
    num_hands: number;
    gesture_score: number;
    body_detected: boolean;
    posture_score: number;
    is_upright: boolean;
    shoulders_level: boolean;
}

interface FluencyMetrics {
    score: number;
    wpm: number;
    filler_count: number;
    fillers: string[];
}

interface RealTimeScoreCardProps {
    visualMetrics: VisualMetrics | null;
    fluencyMetrics: FluencyMetrics | null;
    isAnalyzing: boolean;
    framesProcessed: number;
}

const ScoreBar: React.FC<{
    score: number;
    label: string;
    icon: React.ReactNode;
    statusText?: string;
}> = ({ score, label, icon, statusText }) => {
    const getColorClass = (score: number) => {
        if (score >= 80) return 'bg-green-500';
        if (score >= 60) return 'bg-yellow-500';
        if (score >= 40) return 'bg-orange-500';
        return 'bg-red-500';
    };

    return (
        <div className="flex items-center gap-3 p-3 bg-white rounded-lg border border-gray-200">
            <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center text-gray-500">
                {icon}
            </div>
            <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">{label}</span>
                    <div className="flex items-center gap-2">
                        {statusText && (
                            <span className="text-xs text-gray-500">{statusText}</span>
                        )}
                        <span className={`text-sm font-bold ${score >= 60 ? 'text-green-600' : 'text-orange-600'}`}>
                            {Math.round(score)}%
                        </span>
                    </div>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-300 ${getColorClass(score)}`}
                        style={{ width: `${Math.min(100, score)}%` }}
                    />
                </div>
            </div>
        </div>
    );
};

const StatusIndicator: React.FC<{
    active: boolean;
    label: string;
    icon: React.ReactNode;
}> = ({ active, label, icon }) => (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${active ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-gray-50 text-gray-500 border border-gray-200'
        }`}>
        <div className="w-5 h-5">{icon}</div>
        <span className="text-sm font-medium">{label}</span>
        {active ? (
            <CheckCircleIcon className="w-4 h-4 text-green-500" />
        ) : (
            <ExclamationTriangleIcon className="w-4 h-4 text-gray-400" />
        )}
    </div>
);

export const RealTimeScoreCard: React.FC<RealTimeScoreCardProps> = ({
    visualMetrics,
    fluencyMetrics,
    isAnalyzing,
    framesProcessed
}) => {
    // Default values when not analyzing
    const eyeContactScore = visualMetrics?.gaze_score ?? 0;
    const gestureScore = visualMetrics?.gesture_score ?? 0;
    const postureScore = visualMetrics?.posture_score ?? 0;
    const fluencyScore = fluencyMetrics?.score ?? 0;

    // Calculate overall score
    const overallScore = (
        eyeContactScore * 0.25 +
        gestureScore * 0.20 +
        postureScore * 0.25 +
        fluencyScore * 0.30
    );

    // Gaze direction text
    const getGazeText = () => {
        if (!visualMetrics?.face_detected) return 'No face detected';
        if (visualMetrics.is_looking_at_camera) return 'Looking at camera ✓';
        return `Looking ${visualMetrics.gaze_direction}`;
    };

    // WPM status text
    const getWpmStatus = () => {
        if (!fluencyMetrics) return '';
        const wpm = fluencyMetrics.wpm;
        if (wpm < 100) return 'Too slow';
        if (wpm > 170) return 'Too fast';
        return 'Good pace';
    };

    return (
        <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-4 shadow-sm">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Real-Time Analysis</h3>
                <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${isAnalyzing ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                    <span className="text-xs text-gray-500">
                        {isAnalyzing ? `${framesProcessed} frames` : 'Paused'}
                    </span>
                </div>
            </div>

            {/* Overall Score */}
            <div className="mb-4 p-4 bg-white rounded-lg border border-gray-200 text-center">
                <div className="text-3xl font-bold text-primary-600">
                    {Math.round(overallScore)}%
                </div>
                <div className="text-sm text-gray-500">Overall Score</div>
            </div>

            {/* Status Indicators */}
            <div className="grid grid-cols-2 gap-2 mb-4">
                <StatusIndicator
                    active={visualMetrics?.face_detected ?? false}
                    label="Face"
                    icon={<UserIcon className="w-full h-full" />}
                />
                <StatusIndicator
                    active={visualMetrics?.hands_visible ?? false}
                    label={`Hands (${visualMetrics?.num_hands ?? 0})`}
                    icon={<HandRaisedIcon className="w-full h-full" />}
                />
                <StatusIndicator
                    active={visualMetrics?.is_looking_at_camera ?? false}
                    label="Eye Contact"
                    icon={<EyeIcon className="w-full h-full" />}
                />
                <StatusIndicator
                    active={visualMetrics?.is_upright ?? false}
                    label="Posture"
                    icon={<UserIcon className="w-full h-full" />}
                />
            </div>

            {/* Score Bars */}
            <div className="space-y-2">
                <ScoreBar
                    score={eyeContactScore}
                    label="Eye Contact"
                    icon={<EyeIcon className="w-5 h-5" />}
                    statusText={getGazeText()}
                />
                <ScoreBar
                    score={gestureScore}
                    label="Gestures"
                    icon={<HandRaisedIcon className="w-5 h-5" />}
                    statusText={visualMetrics?.hands_visible ? 'Visible' : 'Hidden'}
                />
                <ScoreBar
                    score={postureScore}
                    label="Posture"
                    icon={<UserIcon className="w-5 h-5" />}
                    statusText={visualMetrics?.shoulders_level ? 'Level' : 'Adjust'}
                />
                <ScoreBar
                    score={fluencyScore}
                    label="Fluency"
                    icon={<MicrophoneIcon className="w-5 h-5" />}
                    statusText={fluencyMetrics ? `${Math.round(fluencyMetrics.wpm)} WPM` : ''}
                />
            </div>

            {/* Filler Words Alert */}
            {fluencyMetrics && fluencyMetrics.filler_count > 0 && (
                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <div className="flex items-start gap-2">
                        <ExclamationTriangleIcon className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                        <div>
                            <p className="text-sm font-medium text-amber-800">
                                Filler words detected ({fluencyMetrics.filler_count})
                            </p>
                            <p className="text-xs text-amber-600 mt-1">
                                {fluencyMetrics.fillers.slice(0, 3).join(', ')}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Tips (when not analyzing) */}
            {!isAnalyzing && (
                <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm text-blue-700">
                        💡 Click "Start Session" to begin real-time analysis
                    </p>
                </div>
            )}
        </div>
    );
};

export default RealTimeScoreCard;
