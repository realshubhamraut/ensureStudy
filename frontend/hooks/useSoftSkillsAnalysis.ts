import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Hook for real-time soft skills analysis via WebSocket
 * 
 * Connects to the backend WebSocket endpoint and sends video frames
 * for analysis, receiving real-time scores back.
 */

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

interface SessionSummary {
    frames_analyzed: number;
    eye_contact_score: number;
    gaze_center_ratio: number;
    gesture_score: number;
    hands_visible_ratio: number;
    posture_score: number;
    is_upright_ratio: number;
}

interface SoftSkillsAnalysisState {
    isConnected: boolean;
    isAnalyzing: boolean;
    visualMetrics: VisualMetrics | null;
    fluencyMetrics: FluencyMetrics | null;
    sessionSummary: SessionSummary | null;
    error: string | null;
    framesProcessed: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8001';
const WS_URL = API_BASE_URL.replace('http', 'ws');

export function useSoftSkillsAnalysis(sessionId: string) {
    const wsRef = useRef<WebSocket | null>(null);
    const [state, setState] = useState<SoftSkillsAnalysisState>({
        isConnected: false,
        isAnalyzing: false,
        visualMetrics: null,
        fluencyMetrics: null,
        sessionSummary: null,
        error: null,
        framesProcessed: 0,
    });

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        const ws = new WebSocket(`${WS_URL}/softskills/ws/${sessionId}`);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('[SoftSkills] WebSocket connected');
            setState(prev => ({ ...prev, isConnected: true, error: null }));
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'analysis') {
                    setState(prev => ({
                        ...prev,
                        visualMetrics: {
                            face_detected: data.face_detected,
                            gaze_direction: data.gaze_direction,
                            gaze_score: data.gaze_score,
                            is_looking_at_camera: data.is_looking_at_camera,
                            hands_visible: data.hands_visible,
                            num_hands: data.num_hands,
                            gesture_score: data.gesture_score,
                            body_detected: data.body_detected,
                            posture_score: data.posture_score,
                            is_upright: data.is_upright,
                            shoulders_level: data.shoulders_level,
                        },
                        framesProcessed: prev.framesProcessed + 1,
                    }));
                } else if (data.type === 'fluency') {
                    setState(prev => ({
                        ...prev,
                        fluencyMetrics: {
                            score: data.score,
                            wpm: data.wpm,
                            filler_count: data.filler_count,
                            fillers: data.fillers,
                        },
                    }));
                } else if (data.type === 'summary') {
                    setState(prev => ({
                        ...prev,
                        sessionSummary: {
                            frames_analyzed: data.frames_analyzed,
                            eye_contact_score: data.eye_contact_score,
                            gaze_center_ratio: data.gaze_center_ratio,
                            gesture_score: data.gesture_score,
                            hands_visible_ratio: data.hands_visible_ratio,
                            posture_score: data.posture_score,
                            is_upright_ratio: data.is_upright_ratio,
                        },
                    }));
                } else if (data.type === 'session_complete') {
                    setState(prev => ({ ...prev, isAnalyzing: false }));
                }
            } catch (e) {
                console.error('[SoftSkills] Failed to parse message:', e);
            }
        };

        ws.onerror = (error) => {
            console.error('[SoftSkills] WebSocket error:', error);
            setState(prev => ({ ...prev, error: 'WebSocket connection error' }));
        };

        ws.onclose = () => {
            console.log('[SoftSkills] WebSocket closed');
            setState(prev => ({ ...prev, isConnected: false }));
        };
    }, [sessionId]);

    // Disconnect from WebSocket
    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setState(prev => ({ ...prev, isConnected: false, isAnalyzing: false }));
    }, []);

    // Send a frame for analysis
    const sendFrame = useCallback((frameBase64: string) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'frame',
                data: frameBase64,
            }));
        }
    }, []);

    // Send transcript for fluency analysis
    const sendTranscript = useCallback((text: string, durationSeconds: number) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'transcript',
                text,
                duration: durationSeconds,
            }));
        }
    }, []);

    // Request session summary
    const requestSummary = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'get_summary',
            }));
        }
    }, []);

    // End session
    const endSession = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'end_session',
            }));
        }
    }, []);

    // Start analyzing (connect and set flag)
    const startAnalysis = useCallback(() => {
        connect();
        setState(prev => ({ ...prev, isAnalyzing: true, framesProcessed: 0 }));
    }, [connect]);

    // Stop analyzing
    const stopAnalysis = useCallback(() => {
        endSession();
        setState(prev => ({ ...prev, isAnalyzing: false }));
    }, [endSession]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return {
        ...state,
        connect,
        disconnect,
        sendFrame,
        sendTranscript,
        requestSummary,
        endSession,
        startAnalysis,
        stopAnalysis,
    };
}

/**
 * Hook for capturing and sending video frames
 */
export function useFrameCapture(
    videoRef: React.RefObject<HTMLVideoElement>,
    sendFrame: (base64: string) => void,
    isAnalyzing: boolean,
    fps: number = 5 // Frames per second to capture
) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const intervalRef = useRef<number | null>(null);

    useEffect(() => {
        if (!isAnalyzing || !videoRef.current) {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            return;
        }

        // Create canvas if needed
        if (!canvasRef.current) {
            canvasRef.current = document.createElement('canvas');
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!ctx) return;

        // Set canvas size
        canvas.width = 640;
        canvas.height = 480;

        // Capture frames at specified FPS
        intervalRef.current = window.setInterval(() => {
            if (video.readyState >= 2) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const base64 = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
                sendFrame(base64);
            }
        }, 1000 / fps);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        };
    }, [isAnalyzing, videoRef, sendFrame, fps]);
}

export default useSoftSkillsAnalysis;
