'use client'

/**
 * LaTeX Renderer Component
 * 
 * Renders LaTeX blocks using KaTeX.
 * Supports both inline and block-level math rendering.
 */
import React, { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

export interface LatexBlock {
    id: string
    latex: string
    position: {
        within_text_index?: number
        step?: number
    }
    plaintext_fallback: string
}

interface LatexRendererProps {
    /** LaTeX string to render */
    latex: string
    /** Whether to render as block (display mode) or inline */
    block?: boolean
    /** Fallback text if rendering fails */
    fallback?: string
    /** Additional CSS classes */
    className?: string
}

interface LatexBlocksRendererProps {
    /** Array of LaTeX blocks from API */
    blocks: LatexBlock[]
    /** Render hint from API (katex or mathjax) */
    renderHint?: 'katex' | 'mathjax'
    /** Additional CSS classes */
    className?: string
}

/**
 * Render a single LaTeX expression
 */
export function LatexRenderer({
    latex,
    block = false,
    fallback = '',
    className = ''
}: LatexRendererProps) {
    const containerRef = useRef<HTMLSpanElement>(null)
    const [error, setError] = React.useState<string | null>(null)

    useEffect(() => {
        if (!containerRef.current || !latex) return

        try {
            katex.render(latex, containerRef.current, {
                displayMode: block,
                throwOnError: false,
                errorColor: '#ef4444',
                trust: false,
                strict: false
            })
            setError(null)
        } catch (e) {
            console.error('[KaTeX] Render error:', e)
            setError(fallback || latex)
        }
    }, [latex, block, fallback])

    if (error) {
        return (
            <span className={`font-mono text-sm text-gray-600 ${className}`}>
                {error}
            </span>
        )
    }

    return (
        <span
            ref={containerRef}
            className={`${block ? 'block my-4 text-center' : 'inline'} ${className}`}
        />
    )
}

/**
 * Render multiple LaTeX blocks from API response
 */
export function LatexBlocksRenderer({
    blocks,
    renderHint = 'katex',
    className = ''
}: LatexBlocksRendererProps) {
    if (!blocks || blocks.length === 0) {
        return null
    }

    // Only KaTeX is supported (as per requirements)
    if (renderHint !== 'katex') {
        console.warn(`[LaTeX] Unsupported render hint: ${renderHint}, using katex`)
    }

    return (
        <div className={`latex-blocks space-y-3 ${className}`}>
            {blocks.map((block) => (
                <div
                    key={block.id}
                    className="latex-block bg-gray-50 rounded-lg p-4 border border-gray-200"
                >
                    {block.position.step !== undefined && (
                        <div className="text-xs text-gray-500 mb-2">
                            Step {block.position.step}
                        </div>
                    )}
                    <LatexRenderer
                        latex={block.latex}
                        block={true}
                        fallback={block.plaintext_fallback}
                    />
                </div>
            ))}
        </div>
    )
}

/**
 * Inline LaTeX in text - replaces $...$ with rendered math
 */
export function InlineMathText({
    text,
    className = ''
}: {
    text: string
    className?: string
}) {
    // Split text by $...$ patterns
    const parts = text.split(/(\$[^$]+\$)/g)

    return (
        <span className={className}>
            {parts.map((part, index) => {
                // Check if this part is math (wrapped in $)
                if (part.startsWith('$') && part.endsWith('$')) {
                    const latex = part.slice(1, -1)
                    return (
                        <LatexRenderer
                            key={index}
                            latex={latex}
                            block={false}
                            fallback={latex}
                        />
                    )
                }
                return <span key={index}>{part}</span>
            })}
        </span>
    )
}

/**
 * Format answer text with LaTeX blocks highlighted
 */
export function MathAnswer({
    text,
    latexBlocks = [],
    className = ''
}: {
    text: string
    latexBlocks?: LatexBlock[]
    className?: string
}) {
    // If we have explicit LaTeX blocks, render them below the text
    const hasExplicitBlocks = latexBlocks && latexBlocks.length > 0

    return (
        <div className={className}>
            {/* Main text - try to render inline math */}
            <div className="prose prose-gray max-w-none">
                <InlineMathText text={text} />
            </div>

            {/* Explicit LaTeX blocks from API */}
            {hasExplicitBlocks && (
                <div className="mt-4">
                    <LatexBlocksRenderer blocks={latexBlocks} />
                </div>
            )}
        </div>
    )
}

export default LatexRenderer
