// Type declarations for @met4citizen/talkinghead
declare module '@met4citizen/talkinghead' {
    export interface TalkingHeadOptions {
        ttsEndpoint?: string | null
        jwtGet?: (() => Promise<string>) | null
        ttsApikey?: string | null
        ttsLang?: string
        ttsVoice?: string
        ttsRate?: number
        ttsPitch?: number
        ttsVolume?: number
        ttsTrimStart?: number
        ttsTrimEnd?: number
        mixerGainSpeech?: number | null
        mixerGainBackground?: number | null
        lipsyncModules?: string[]
        lipsyncLang?: string
        pcmSampleRate?: number
        audioCtx?: AudioContext | null
        modelRoot?: string
        modelPixelRatio?: number
        modelFPS?: number
        modelMovementFactor?: number
        dracoEnabled?: boolean
        dracoDecoderPath?: string
        cameraView?: 'full' | 'mid' | 'upper' | 'head'
        cameraDistance?: number
        cameraX?: number
        cameraY?: number
        cameraRotateX?: number
        cameraRotateY?: number
        cameraRotateEnable?: boolean
        cameraPanEnable?: boolean
        cameraZoomEnable?: boolean
        lightAmbientColor?: number
        lightAmbientIntensity?: number
        lightDirectColor?: number
        lightDirectIntensity?: number
        lightDirectPhi?: number
        lightDirectTheta?: number
        lightSpotColor?: number
        lightSpotIntensity?: number
        lightSpotPhi?: number
        lightSpotTheta?: number
        lightSpotDispersion?: number
        avatarMood?: 'neutral' | 'happy' | 'angry' | 'sad' | 'fear' | 'disgust' | 'love' | 'sleep'
        avatarMute?: boolean
        avatarIdleEyeContact?: number
        avatarIdleHeadMove?: number
        avatarSpeakingEyeContact?: number
        avatarSpeakingHeadMove?: number
        avatarIgnoreCamera?: boolean
        statsNode?: HTMLElement | null
        statsStyle?: string | null
    }

    export interface AvatarConfig {
        url: string
        body?: 'M' | 'F'
        avatarMood?: string
        ttsLang?: string
        ttsVoice?: string
        ttsRate?: number
        ttsPitch?: number
        ttsVolume?: number
        lipsyncLang?: string
        baseline?: {
            headRotateX?: number
            eyeBlinkLeft?: number
            eyeBlinkRight?: number
            [key: string]: number | undefined
        }
        retarget?: {
            x?: number
            y?: number
            z?: number
            rx?: number
            ry?: number
            rz?: number
        }
        origin?: { x: number; y: number; z: number }
    }

    export interface ViewOptions {
        cameraDistance?: number
        cameraX?: number
        cameraY?: number
        cameraRotateX?: number
        cameraRotateY?: number
    }

    export interface LightingOptions {
        lightAmbientColor?: number
        lightAmbientIntensity?: number
        lightDirectColor?: number
        lightDirectIntensity?: number
        lightDirectPhi?: number
        lightDirectTheta?: number
        lightSpotColor?: number
        lightSpotIntensity?: number
        lightSpotPhi?: number
        lightSpotTheta?: number
        lightSpotDispersion?: number
    }

    export interface SpeakOptions {
        lipsyncLang?: string
        ttsLang?: string
        ttsVoice?: string
        ttsRate?: number
        ttsPitch?: number
        ttsVolume?: number
        avatarMood?: string
        avatarMute?: boolean
    }

    export class TalkingHead {
        constructor(container: HTMLElement, options?: TalkingHeadOptions)

        showAvatar(avatar: AvatarConfig, onprogress?: ((progress: number) => void) | null): Promise<void>
        setView(view: 'full' | 'mid' | 'upper' | 'head', options?: ViewOptions): void
        setLighting(options: LightingOptions): void

        speakText(text: string, options?: SpeakOptions, onsubtitles?: ((text: string) => void) | null, excludes?: string[]): Promise<void>
        speakAudio(audio: any, options?: SpeakOptions, onsubtitles?: ((text: string) => void) | null): Promise<void>
        speakEmoji(emoji: string): void
        speakBreak(duration: number): void
        speakMarker(onmarker: () => void): void

        stopSpeaking(): void

        lookAt(x: number, y: number, duration: number): void
        lookAhead(duration: number): void
        lookAtCamera(duration: number): void
        makeEyeContact(duration: number): void

        setMood(mood: 'neutral' | 'happy' | 'angry' | 'sad' | 'fear' | 'disgust' | 'love' | 'sleep'): void

        playBackgroundAudio(url: string): void
        stopBackgroundAudio(): void
        setMixerGain(speech: number, background?: number | null, fadeSecs?: number): void

        playAnimation(url: string, onprogress?: ((progress: number) => void) | null, duration?: number, index?: number, scale?: number): Promise<void>
        stopAnimation(): void

        playPose(url: string, onprogress?: ((progress: number) => void) | null, duration?: number, index?: number, scale?: number): Promise<void>
        stopPose(): void

        // Streaming methods
        streamStart(options?: SpeakOptions, onAudioStart?: () => void, onAudioEnd?: () => void, onSubtitles?: (text: string) => void, onMetrics?: () => void): void
        streamAudio(audio: ArrayBuffer): void
        streamNotifyEnd(): void
        streamInterrupt(): void
        streamStop(): void
    }
}
