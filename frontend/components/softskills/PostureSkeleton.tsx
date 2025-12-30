'use client';

import React from 'react';

interface PostureSkeletonProps {
    bodyDetected: boolean;
    shoulderTilt?: number;
    isUpright: boolean;
    shouldersLevel: boolean;
    postureScore: number;
    size?: 'sm' | 'md' | 'lg';
}

/**
 * Visual skeleton display showing posture status
 * Displays a simplified upper body with indicators for alignment
 */
export const PostureSkeleton: React.FC<PostureSkeletonProps> = ({
    bodyDetected,
    shoulderTilt = 0,
    isUpright,
    shouldersLevel,
    postureScore,
    size = 'md'
}) => {
    // Size configurations
    const sizes = {
        sm: { width: 80, height: 100, strokeWidth: 3, headSize: 16 },
        md: { width: 120, height: 150, strokeWidth: 4, headSize: 24 },
        lg: { width: 160, height: 200, strokeWidth: 5, headSize: 32 },
    };

    const s = sizes[size];
    const cx = s.width / 2;

    // Skeleton color based on score
    const getColor = () => {
        if (!bodyDetected) return '#9CA3AF'; // gray
        if (postureScore >= 80) return '#22C55E'; // green
        if (postureScore >= 60) return '#EAB308'; // yellow
        if (postureScore >= 40) return '#F97316'; // orange
        return '#EF4444'; // red
    };

    const color = getColor();

    // Calculate shoulder positions with tilt
    const shoulderY = s.height * 0.35;
    const shoulderWidth = s.width * 0.5;
    const tiltRad = (shoulderTilt * Math.PI) / 180;
    const leftShoulderY = shoulderY + Math.sin(tiltRad) * (shoulderWidth / 2);
    const rightShoulderY = shoulderY - Math.sin(tiltRad) * (shoulderWidth / 2);

    // Status indicators
    const StatusBadge: React.FC<{ good: boolean; label: string }> = ({ good, label }) => (
        <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${good ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
            }`}>
            <span>{good ? '✓' : '!'}</span>
            <span>{label}</span>
        </div>
    );

    if (!bodyDetected) {
        return (
            <div className="flex flex-col items-center gap-2">
                <div
                    className="flex items-center justify-center bg-gray-100 rounded-lg"
                    style={{ width: s.width, height: s.height }}
                >
                    <div className="text-gray-400 text-center text-sm">
                        <div className="text-2xl mb-1">👤</div>
                        <div>No body detected</div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col items-center gap-3">
            {/* Skeleton SVG */}
            <svg
                width={s.width}
                height={s.height}
                className="transition-all duration-300"
            >
                {/* Background */}
                <rect
                    x={0}
                    y={0}
                    width={s.width}
                    height={s.height}
                    rx={8}
                    fill="#F9FAFB"
                />

                {/* Guide lines */}
                <line
                    x1={cx}
                    y1={0}
                    x2={cx}
                    y2={s.height}
                    stroke="#E5E7EB"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                />
                <line
                    x1={0}
                    y1={shoulderY}
                    x2={s.width}
                    y2={shoulderY}
                    stroke="#E5E7EB"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                />

                {/* Head */}
                <circle
                    cx={cx}
                    cy={s.height * 0.15}
                    r={s.headSize / 2}
                    fill="none"
                    stroke={color}
                    strokeWidth={s.strokeWidth}
                />

                {/* Neck */}
                <line
                    x1={cx}
                    y1={s.height * 0.15 + s.headSize / 2}
                    x2={cx}
                    y2={shoulderY}
                    stroke={color}
                    strokeWidth={s.strokeWidth}
                    strokeLinecap="round"
                />

                {/* Shoulders */}
                <line
                    x1={cx - shoulderWidth / 2}
                    y1={leftShoulderY}
                    x2={cx + shoulderWidth / 2}
                    y2={rightShoulderY}
                    stroke={color}
                    strokeWidth={s.strokeWidth}
                    strokeLinecap="round"
                />

                {/* Spine */}
                <line
                    x1={cx}
                    y1={shoulderY}
                    x2={cx}
                    y2={s.height * 0.75}
                    stroke={color}
                    strokeWidth={s.strokeWidth}
                    strokeLinecap="round"
                />

                {/* Left arm */}
                <line
                    x1={cx - shoulderWidth / 2}
                    y1={leftShoulderY}
                    x2={cx - shoulderWidth / 2 - 10}
                    y2={s.height * 0.6}
                    stroke={color}
                    strokeWidth={s.strokeWidth}
                    strokeLinecap="round"
                />

                {/* Right arm */}
                <line
                    x1={cx + shoulderWidth / 2}
                    y1={rightShoulderY}
                    x2={cx + shoulderWidth / 2 + 10}
                    y2={s.height * 0.6}
                    stroke={color}
                    strokeWidth={s.strokeWidth}
                    strokeLinecap="round"
                />

                {/* Shoulder level indicators */}
                {!shouldersLevel && (
                    <>
                        <circle
                            cx={cx - shoulderWidth / 2}
                            cy={leftShoulderY}
                            r={4}
                            fill="#EF4444"
                        />
                        <circle
                            cx={cx + shoulderWidth / 2}
                            cy={rightShoulderY}
                            r={4}
                            fill="#EF4444"
                        />
                    </>
                )}

                {/* Score badge */}
                <g transform={`translate(${s.width - 25}, 10)`}>
                    <rect
                        x={0}
                        y={0}
                        width={24}
                        height={18}
                        rx={4}
                        fill={color}
                    />
                    <text
                        x={12}
                        y={13}
                        textAnchor="middle"
                        fill="white"
                        fontSize={10}
                        fontWeight="bold"
                    >
                        {Math.round(postureScore)}
                    </text>
                </g>
            </svg>

            {/* Status badges */}
            <div className="flex flex-wrap gap-2 justify-center">
                <StatusBadge good={isUpright} label={isUpright ? 'Upright' : 'Leaning'} />
                <StatusBadge good={shouldersLevel} label={shouldersLevel ? 'Level' : 'Tilted'} />
            </div>

            {/* Tilt info */}
            {shoulderTilt !== 0 && (
                <div className="text-xs text-gray-500">
                    Tilt: {shoulderTilt.toFixed(1)}°
                </div>
            )}
        </div>
    );
};

/**
 * Compact posture indicator for inline use
 */
export const PostureIndicatorCompact: React.FC<{
    isUpright: boolean;
    shouldersLevel: boolean;
    postureScore: number;
}> = ({ isUpright, shouldersLevel, postureScore }) => {
    const good = isUpright && shouldersLevel;

    return (
        <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${good
                ? 'bg-green-100 text-green-700'
                : 'bg-orange-100 text-orange-700'
            }`}>
            <span>{good ? '🧍' : '⚠️'}</span>
            <span>{Math.round(postureScore)}%</span>
            <span className="text-opacity-70">
                {good ? 'Good posture' : isUpright ? 'Level shoulders' : 'Sit upright'}
            </span>
        </div>
    );
};

export default PostureSkeleton;
