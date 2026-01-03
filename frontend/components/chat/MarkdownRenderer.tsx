'use client'

import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github-dark.css'

interface MarkdownRendererProps {
    content: string
    className?: string
}

/**
 * Enhanced Markdown Renderer for AI Tutor
 * 
 * Supports:
 * - LaTeX math equations: $inline$ and $$block$$
 * - Chemical formulas via LaTeX: $\ce{H2O}$, $\ce{CO2}$
 * - Code blocks with syntax highlighting
 * - GitHub Flavored Markdown (tables, strikethrough, etc.)
 * - Links, images, lists, etc.
 * 
 * Uses system sans (Inter) for clean, Perplexity-like aesthetic.
 */
export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
    return (
        <div className={`markdown-content ${className}`}>
            <ReactMarkdown
                remarkPlugins={[remarkMath, remarkGfm]}
                rehypePlugins={[rehypeKatex, rehypeHighlight]}
                components={{
                    // Headings - Clean, modern, not too heavy
                    h1: ({ children }) => (
                        <h1 className="text-xl font-semibold text-gray-900 mt-6 mb-3 first:mt-0 tracking-tight">{children}</h1>
                    ),
                    h2: ({ children }) => (
                        <h2 className="text-lg font-semibold text-gray-900 mt-5 mb-2 tracking-tight">{children}</h2>
                    ),
                    h3: ({ children }) => (
                        <h3 className="text-base font-semibold text-gray-900 mt-4 mb-2">{children}</h3>
                    ),

                    // Paragraphs - High legibility, relaxed leading
                    p: ({ children }) => (
                        <p className="text-gray-800 leading-7 mb-4 last:mb-0 text-[15px]">{children}</p>
                    ),

                    // Lists - consistent spacing
                    ul: ({ children }) => (
                        <ul className="list-disc list-outside space-y-1 mb-4 text-gray-800 ml-5 text-[15px] leading-7 marker:text-gray-400">{children}</ul>
                    ),
                    ol: ({ children }) => (
                        <ol className="list-decimal list-outside space-y-1 mb-4 text-gray-800 ml-5 text-[15px] leading-7 marker:text-gray-400">{children}</ol>
                    ),
                    li: ({ children }) => (
                        <li className="pl-1">{children}</li>
                    ),

                    // Code blocks
                    code: ({ className, children, ...props }) => {
                        const isInline = !className

                        if (isInline) {
                            return (
                                <code
                                    className="bg-gray-100 text-primary-700 px-1.5 py-0.5 rounded text-sm font-mono"
                                    {...props}
                                >
                                    {children}
                                </code>
                            )
                        }

                        return (
                            <code className={`${className} block`} {...props}>
                                {children}
                            </code>
                        )
                    },
                    pre: ({ children }) => (
                        <pre className="bg-gray-900 rounded-xl p-4 overflow-x-auto mb-4 text-sm">
                            {children}
                        </pre>
                    ),

                    // Blockquotes
                    blockquote: ({ children }) => (
                        <blockquote className="border-l-4 border-primary-500 bg-primary-50 px-4 py-3 mb-4 italic text-gray-700 rounded-r-lg">
                            {children}
                        </blockquote>
                    ),

                    // Tables
                    table: ({ children }) => (
                        <div className="overflow-x-auto mb-4">
                            <table className="min-w-full border-collapse border border-gray-200 rounded-lg overflow-hidden">
                                {children}
                            </table>
                        </div>
                    ),
                    thead: ({ children }) => (
                        <thead className="bg-gray-100">{children}</thead>
                    ),
                    th: ({ children }) => (
                        <th className="border border-gray-200 px-4 py-2 text-left font-semibold text-gray-900">
                            {children}
                        </th>
                    ),
                    td: ({ children }) => (
                        <td className="border border-gray-200 px-4 py-2 text-gray-800">{children}</td>
                    ),

                    // Links
                    a: ({ href, children }) => (
                        <a
                            href={href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-primary-600 hover:text-primary-700 underline underline-offset-2"
                        >
                            {children}
                        </a>
                    ),

                    // Strong and emphasis
                    strong: ({ children }) => (
                        <strong className="font-bold text-gray-900">{children}</strong>
                    ),
                    em: ({ children }) => (
                        <em className="italic text-gray-700">{children}</em>
                    ),

                    // Horizontal rule
                    hr: () => (
                        <hr className="border-t border-gray-200 my-6" />
                    ),
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    )
}
