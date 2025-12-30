'use client';

import React from 'react';

interface GazeIndicatorProps {
    gazeDirection: 'center' | 'left' | 'right' | 'unknown';
    isLookingAtCamera: boolean;
    headYaw?: number;
    headPitch?: number;
    size?: 'sm' | 'md' | 'lg';
}

/**
 * Visual indicator showing where the user is looking
 * Displays an eye with a pupil that moves based on gaze direction
 */
export const GazeIndicator: React.FC<GazeIndicatorProps> = ({
    gazeDirection,
    isLookingAtCamera,
    headYaw = 0,
    headPitch = 0,
    size = 'md'
}) => {
    // Size configurations
    const sizes = {
        sm: { container: 60, eye: 40, pupil: 12, iris: 24 },
        md: { container: 100, eye: 70, pupil: 18, iris: 40 },
        lg: { container: 140, eye: 100, pupil: 24, iris: 56 },
    };

    const s = sizes[size];

    // Calculate pupil offset based on gaze direction and head pose
    const getPupilOffset = () => {
        let x = 0;
        let y = 0;

        // Base offset from gaze direction
        switch (gazeDirection) {
            case 'left':
                x = -s.iris * 0.25;
                break;
            case 'right':
                x = s.iris * 0.25;
                break;
            case 'center':
            default:
                x = 0;
        }

        // Add head yaw influence
        if (headYaw) {
            x += (headYaw / 30) * s.iris * 0.3;
        }

        // Add head pitch influence
        if (headPitch) {
            y -= (headPitch / 20) * s.iris * 0.2;
        }

        // Clamp to stay within bounds
        const maxOffset = (s.iris - s.pupil) / 2;
        x = Math.max(-maxOffset, Math.min(maxOffset, x));
        y = Math.max(-maxOffset, Math.min(maxOffset, y));

        return { x, y };
    };

    const offset = getPupilOffset();

    // Status color
    const statusColor = isLookingAtCamera
        ? 'bg-green-500'
        : gazeDirection === 'unknown'
            ? 'bg-gray-400'
            : 'bg-orange-500';

    const statusText = isLookingAtCamera
        ? '✓ Looking at camera'
        : gazeDirection === 'unknown'
            ? 'No face detected'
            : `Looking ${gazeDirection}`;

    return (
        <div className="flex flex-col items-center gap-2">
            {/* Eye visualization */}
            <div
                className="relative flex items-center justify-center"
                style={{ width: s.container, height: s.container }}
            >
                {/* Eye outline (sclera) */}
                <div
                    className="absolute bg-white border-4 border-gray-700 rounded-full shadow-inner flex items-center justify-center"
                    style={{
                        width: s.eye,
                        height: s.eye,
                        boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.15)'
                    }}
                >
                    {/* Iris */}
                    <div
                        className="absolute bg-gradient-to-br from-blue-400 to-blue-600 rounded-full flex items-center justify-center"
                        style={{
                            width: s.iris,
                            height: s.iris,
                            transform: `translate(${offset.x}px, ${offset.y}px)`,
                            transition: 'transform 0.15s ease-out',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                        }}
                    >
                        {/* Pupil */}
                        <div
                            className="bg-gray-900 rounded-full"
                            style={{
                                width: s.pupil,
                                height: s.pupil,
                            }}
                        />
                        {/* Light reflection */}
                        <div
                            className="absolute bg-white rounded-full opacity-80"
                            style={{
                                width: s.pupil * 0.4,
                                height: s.pupil * 0.4,
                                top: s.iris * 0.15,
                                right: s.iris * 0.15,
                            }}
                        />
                    </div>
                </div>

                {/* Direction indicator arrows */}
                {gazeDirection !== 'center' && gazeDirection !== 'unknown' && (
                    <div
                        className="absolute text-2xl text-orange-500 font-bold"
                        style={{
                            [gazeDirection === 'left' ? 'left' : 'right']: 0,
                            animation: 'pulse 1s infinite',
                        }}
                    >
                        {gazeDirection === 'left' ? '←' : '→'}
                    </div>
                )}
            </div>

            {/* Status text */}
            <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${statusColor}`} />
                <span className="text-sm font-medium text-gray-700">{statusText}</span>
            </div>

            {/* Head pose details (optional, for debugging) */}
            {(headYaw !== 0 || headPitch !== 0) && (
                <div className="text-xs text-gray-500">
                    Yaw: {headYaw.toFixed(1)}° | Pitch: {headPitch.toFixed(1)}°
                </div>
            )}
        </div>
    );
};

/**
 * Compact inline gaze indicator for use in headers/toolbars
 */
export const GazeIndicatorCompact: React.FC<{
    isLookingAtCamera: boolean;
    gazeDirection: 'center' | 'left' | 'right' | 'unknown';
}> = ({ isLookingAtCamera, gazeDirection }) => {
    const icon = gazeDirection === 'left'
        ? '👁️←'
        : gazeDirection === 'right'
            ? '→👁️'
            : gazeDirection === 'center'
                ? '👁️'
                : '❌';

    return (
        <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${isLookingAtCamera
                ? 'bg-green-100 text-green-700'
                : gazeDirection === 'unknown'
                    ? 'bg-gray-100 text-gray-500'
                    : 'bg-orange-100 text-orange-700'
            }`}>
            <span>{icon}</span>
            <span>
                {isLookingAtCamera
                    ? 'Eye Contact'
                    : gazeDirection === 'unknown'
                        ? 'No Face'
                        : `Look ${gazeDirection}`}
            </span>
        </div>
    );
};

export default GazeIndicator;
