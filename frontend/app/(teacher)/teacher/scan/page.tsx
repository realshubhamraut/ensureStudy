
'use client'
import { getApiBaseUrl, getAiServiceUrl } from '@/utils/api'


import { useState, useRef, useEffect } from 'react'
import {
    DocumentTextIcon,
    VideoCameraIcon,
    CloudArrowUpIcon,
    SparklesIcon,
    CheckCircleIcon,
    ClockIcon,
    ChartBarIcon,
    XMarkIcon,
    PencilIcon,
    ArrowPathIcon,
    EyeIcon,
    CheckIcon,
    XCircleIcon,
    PhotoIcon,
    ChatBubbleLeftRightIcon,
    DocumentArrowUpIcon,
    AcademicCapIcon,
    CalendarIcon,
    BookOpenIcon,
    ArrowLeftIcon,
    PlusIcon,
    ClipboardDocumentListIcon,
    BoltIcon,
    CalendarDaysIcon,
    PresentationChartBarIcon,
    PencilSquareIcon,
    TagIcon
} from '@heroicons/react/24/outline'

// ============================================================================
// Types
// ============================================================================

interface ParsedQuestion {
    number: string
    text: string
    marks: number
    section?: string
    question_type: string
    expected_answer?: string
    keywords: string[]
}

interface QuestionPaper {
    id: string
    title: string
    subject: string
    total_marks: number
    time_limit_minutes: number
    sections: string[]
    questions: ParsedQuestion[]
}

interface ImageQuality {
    score: number
    label: string
    is_acceptable: boolean
    skew_angle: number
}

interface RecognitionResult {
    extracted_text: string
    confidence: number
}

interface ScoringResult {
    score: number
    max_marks: number
    confidence: number
    breakdown: {
        semantic: number
        keyword: number
        steps: number
    }
    matched_keywords: string[]
    missing_keywords: string[]
    feedback: string[]
}

interface EvaluationRecord {
    id: string
    student_id: string
    question_number: string
    image_quality?: ImageQuality
    recognition?: RecognitionResult
    scoring?: ScoringResult
    status: 'pending' | 'processing' | 'awaiting_review' | 'approved' | 'rejected'
    created_at: string
    final_score?: number
    teacher_comments?: string
}

// Exam Session Types
type ExamType = 'mid_term' | 'unit_test' | 'surprise_test' | 'quarterly' | 'half_yearly' | 'final' | 'practice' | 'other'

interface ExamSession {
    id: string
    name: string
    exam_type: ExamType
    subject: string
    class_name: string
    date: string
    total_students?: number
    created_at: string
    status: 'in_progress' | 'evaluating' | 'completed' | 'results_declared'
    evaluation_count?: number
}

interface Student {
    id: string
    name: string
    email: string
    username: string
    avatar_url?: string
}

const EXAM_TYPE_ICONS: Record<ExamType, React.ComponentType<{ className?: string }>> = {
    'mid_term': DocumentTextIcon,
    'unit_test': ClipboardDocumentListIcon,
    'surprise_test': BoltIcon,
    'quarterly': CalendarDaysIcon,
    'half_yearly': PresentationChartBarIcon,
    'final': AcademicCapIcon,
    'practice': PencilSquareIcon,
    'other': TagIcon,
}

const EXAM_TYPES: { value: ExamType; label: string; color: string }[] = [
    { value: 'mid_term', label: 'Mid Term Exam', color: 'from-blue-500 to-blue-600' },
    { value: 'unit_test', label: 'Unit Test', color: 'from-green-500 to-green-600' },
    { value: 'surprise_test', label: 'Surprise Test', color: 'from-yellow-500 to-orange-500' },
    { value: 'quarterly', label: 'Quarterly Exam', color: 'from-purple-500 to-purple-600' },
    { value: 'half_yearly', label: 'Half Yearly', color: 'from-indigo-500 to-indigo-600' },
    { value: 'final', label: 'Final Exam', color: 'from-red-500 to-red-600' },
    { value: 'practice', label: 'Practice Test', color: 'from-teal-500 to-teal-600' },
    { value: 'other', label: 'Other', color: 'from-gray-500 to-gray-600' },
]

const SUBJECTS = ['Physics', 'Chemistry', 'Mathematics', 'Biology', 'English', 'History', 'Geography', 'Computer Science', 'Economics', 'Other']
const CLASSES = ['Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10', 'Class 11', 'Class 12', 'Other']

// ============================================================================
// Main Component
// ============================================================================

export default function TeacherEvaluationPage() {
    // Main view mode: 'evaluate' = new evaluations, 'results' = view results
    const [mainViewMode, setMainViewMode] = useState<'evaluate' | 'results'>('evaluate')

    // Past exam sessions for results view
    const [pastExamSessions, setPastExamSessions] = useState<ExamSession[]>([])
    const [loadingSessions, setLoadingSessions] = useState(false)

    // Student selection state
    const [students, setStudents] = useState<Student[]>([])
    const [selectedStudent, setSelectedStudent] = useState<Student | null>(null)
    const [showStudentSelection, setShowStudentSelection] = useState(false)
    const [loadingStudents, setLoadingStudents] = useState(false)

    // Exam session state
    const [currentExamSession, setCurrentExamSession] = useState<ExamSession | null>(null)
    const [showExamCreation, setShowExamCreation] = useState(true)
    const [examName, setExamName] = useState('')
    const [selectedExamType, setSelectedExamType] = useState<ExamType | null>(null)
    const [examSubject, setExamSubject] = useState('')
    const [examClass, setExamClass] = useState('')
    const [examDate, setExamDate] = useState(new Date().toISOString().split('T')[0])

    const [activeTab, setActiveTab] = useState<'paper' | 'evaluate' | 'review'>('paper')
    const [uploading, setUploading] = useState(false)

    // Question paper state
    const [questionPaper, setQuestionPaper] = useState<QuestionPaper | null>(null)
    const [paperText, setPaperText] = useState('')
    const [selectedQuestion, setSelectedQuestion] = useState<ParsedQuestion | null>(null)
    const [paperInputMode, setPaperInputMode] = useState<'text' | 'pdf'>('text')
    const [pdfFile, setPdfFile] = useState<File | null>(null)

    // Answer evaluation state
    const [selectedImage, setSelectedImage] = useState<File | null>(null)
    const [imagePreview, setImagePreview] = useState<string | null>(null)
    const [currentEvaluation, setCurrentEvaluation] = useState<EvaluationRecord | null>(null)
    const [editingScore, setEditingScore] = useState(false)
    const [manualScore, setManualScore] = useState<number>(0)
    const [teacherComments, setTeacherComments] = useState('')

    // AI Review state
    const [aiReviewMode, setAiReviewMode] = useState(false)
    const [aiSuggestion, setAiSuggestion] = useState('')
    const [loadingAI, setLoadingAI] = useState(false)

    // AI Auto-fill state
    const [generatingAnswer, setGeneratingAnswer] = useState(false)
    const [generatingKeywords, setGeneratingKeywords] = useState(false)
    const [generatingAll, setGeneratingAll] = useState(false)
    const [allModeSelected, setAllModeSelected] = useState(false)

    // Batch Auto-Evaluation state
    const [autoEvaluating, setAutoEvaluating] = useState(false)
    const [batchResults, setBatchResults] = useState<{
        question_number: string
        question_text: string
        max_marks: number
        score: number
        feedback: string
        status: 'pending' | 'evaluating' | 'done' | 'error'
    }[]>([])
    const [showBatchResults, setShowBatchResults] = useState(false)

    // Exam Detail Modal state
    const [showExamDetailModal, setShowExamDetailModal] = useState(false)
    const [selectedExamSession, setSelectedExamSession] = useState<any>(null)
    const [examEvaluations, setExamEvaluations] = useState<any[]>([])
    const [loadingExamDetail, setLoadingExamDetail] = useState(false)
    const [editingEvaluation, setEditingEvaluation] = useState<string | null>(null)
    const [editScore, setEditScore] = useState<number>(0)

    const fileInputRef = useRef<HTMLInputElement>(null)
    const paperInputRef = useRef<HTMLInputElement>(null)
    const pdfInputRef = useRef<HTMLInputElement>(null)

    // API: Fetch past exam sessions
    const fetchExamSessions = async () => {
        setLoadingSessions(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/exam-sessions`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                setPastExamSessions(data.exam_sessions || [])
            }
        } catch (error) {
            console.error('Failed to fetch exam sessions:', error)
        }
        setLoadingSessions(false)
    }

    // API: Fetch students for evaluation
    const fetchStudents = async () => {
        setLoadingStudents(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/students`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                setStudents(data.students || [])
            }
        } catch (error) {
            console.error('Failed to fetch students:', error)
        }
        setLoadingStudents(false)
    }

    // API: Create exam session in database
    const createExamSessionAPI = async (session: ExamSession): Promise<ExamSession | null> => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/exam-session`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: session.name,
                    exam_type: session.exam_type,
                    subject: session.subject,
                    class_name: session.class_name,
                    date: session.date,
                    total_marks: 0,
                    question_paper: {}
                })
            })
            if (res.ok) {
                const data = await res.json()
                return data.exam_session
            }
        } catch (error) {
            console.error('Failed to create exam session:', error)
        }
        return null
    }

    // API: Save student evaluation to database
    const saveEvaluationAPI = async (questionEval: { question_number: string; score: number; max_marks: number; feedback: string[] }) => {
        if (!currentExamSession || !selectedStudent) return

        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/student-evaluation`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    exam_session_id: currentExamSession.id,
                    student_id: selectedStudent.id,
                    question_evaluations: [questionEval],
                    total_score: questionEval.score,
                    max_score: questionEval.max_marks,
                    status: 'evaluated',
                    teacher_comments: teacherComments
                })
            })
            if (res.ok) {
                console.log('Evaluation saved!')
            }
        } catch (error) {
            console.error('Failed to save evaluation:', error)
        }
    }

    // API: Declare results
    const declareResultsAPI = async (sessionId: string) => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/exam/${sessionId}/declare-results`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                alert('✅ Results declared successfully!')
                fetchExamSessions()
                if (selectedExamSession?.id === sessionId) {
                    fetchExamDetail(sessionId)
                }
            } else {
                const data = await res.json()
                alert(data.error || 'Failed to declare results')
            }
        } catch (error) {
            console.error('Failed to declare results:', error)
            alert('Failed to declare results')
        }
    }

    // API: Rollback results
    const rollbackResultsAPI = async (sessionId: string) => {
        if (!confirm('Are you sure you want to rollback results? Students will no longer see their results until you re-declare.')) {
            return
        }
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/exam/${sessionId}/rollback-results`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                alert('✅ Results rolled back. You can now edit marks.')
                fetchExamSessions()
                if (selectedExamSession?.id === sessionId) {
                    fetchExamDetail(sessionId)
                }
            } else {
                const data = await res.json()
                alert(data.error || 'Failed to rollback results')
            }
        } catch (error) {
            console.error('Failed to rollback results:', error)
            alert('Failed to rollback results')
        }
    }

    // API: Fetch exam session detail with evaluations
    const fetchExamDetail = async (sessionId: string) => {
        setLoadingExamDetail(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/exam-session/${sessionId}`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                setSelectedExamSession(data.exam_session)
                setExamEvaluations(data.evaluations || [])
            }
        } catch (error) {
            console.error('Failed to fetch exam detail:', error)
        }
        setLoadingExamDetail(false)
    }

    // API: Update student score
    const updateStudentScore = async (evaluationId: string, studentId: string, newScore: number) => {
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/student-evaluation`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    exam_session_id: selectedExamSession.id,
                    student_id: studentId,
                    total_score: newScore,
                    status: 'evaluated'
                })
            })
            if (res.ok) {
                alert('✅ Score updated!')
                fetchExamDetail(selectedExamSession.id)
                setEditingEvaluation(null)
            }
        } catch (error) {
            console.error('Failed to update score:', error)
            alert('Failed to update score')
        }
    }

    // Open exam detail modal
    const openExamDetail = (session: any) => {
        setSelectedExamSession(session)
        setShowExamDetailModal(true)
        fetchExamDetail(session.id)
    }

    // Resume an exam session to continue evaluating students
    const resumeExamSession = async (session: any, studentToEvaluate?: Student) => {
        // 1. Fetch the complete session data with question paper
        setLoadingExamDetail(true)
        try {
            const res = await fetch(`${getApiBaseUrl()}/api/evaluation/exam-session/${session.id}`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('accessToken')}` }
            })
            if (res.ok) {
                const data = await res.json()
                const fullSession = data.exam_session

                // 2. Set the current exam session
                setCurrentExamSession(fullSession)

                // 3. Restore question paper if available
                if (fullSession.question_paper && Object.keys(fullSession.question_paper).length > 0) {
                    setQuestionPaper(fullSession.question_paper)
                    // Restore paper text if available
                    if (fullSession.question_paper.rawText) {
                        setPaperText(fullSession.question_paper.rawText)
                    }
                }

                // 4. Close modal and reset creation view
                setShowExamDetailModal(false)
                setShowExamCreation(false)

                // 5. If student provided, go directly to evaluation
                if (studentToEvaluate) {
                    setSelectedStudent(studentToEvaluate)
                    setShowStudentSelection(false)
                } else {
                    // Show student selection to pick who to evaluate
                    setShowStudentSelection(true)
                    setSelectedStudent(null)
                }

                // Reset evaluation state for new student
                setCurrentEvaluation(null)
                setSelectedImage(null)
                setImagePreview(null)
                setBatchResults([])
                setShowBatchResults(false)
            }
        } catch (error) {
            console.error('Failed to resume exam session:', error)
        }
        setLoadingExamDetail(false)
    }

    // Load initial data
    useEffect(() => {
        fetchExamSessions()
        fetchStudents()
    }, [])

    // Auto-evaluate ALL questions at once using AI
    const handleAutoEvaluateAll = async () => {
        if (!questionPaper?.questions?.length || !imagePreview) {
            alert('Please upload a question paper and student answer image first')
            return
        }

        setAutoEvaluating(true)
        setShowBatchResults(true)

        // Initialize results with pending status
        const initialResults = questionPaper.questions.map(q => ({
            question_number: q.number,
            question_text: q.text,
            max_marks: q.marks,
            score: 0,
            feedback: '',
            status: 'pending' as const
        }))
        setBatchResults(initialResults)

        // Process each question
        for (let i = 0; i < questionPaper.questions.length; i++) {
            const q = questionPaper.questions[i]

            // Update status to evaluating
            setBatchResults(prev => prev.map((r, idx) =>
                idx === i ? { ...r, status: 'evaluating' as const } : r
            ))

            try {
                // Call AI to evaluate this question
                const response = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: `You are evaluating a student's answer. 
Question (${q.marks} marks): "${q.text}"
Expected Answer: "${q.expected_answer || 'Not provided'}"
Keywords to look for: ${q.keywords.length > 0 ? q.keywords.join(', ') : 'Not provided'}

Evaluate the student's answer and provide:
1. A score out of ${q.marks}
2. Brief feedback (1-2 sentences)

Format your response EXACTLY as: SCORE: [number] | FEEDBACK: [your feedback]

Example: SCORE: 3 | FEEDBACK: Good understanding of basics but missed key concept of inertia.`,
                        student_id: 'batch_evaluation',
                        subject: questionPaper.subject || 'general'
                    })
                })

                if (response.ok) {
                    const data = await response.json()
                    const aiResponse = data.answer || ''

                    // Parse the response
                    const scoreMatch = aiResponse.match(/SCORE:\s*(\d+(?:\.\d+)?)/i)
                    const feedbackMatch = aiResponse.match(/FEEDBACK:\s*(.+)/i)

                    const score = scoreMatch ? Math.min(parseFloat(scoreMatch[1]), q.marks) : Math.round(q.marks * 0.7)
                    const feedback = feedbackMatch ? feedbackMatch[1].trim() : 'Evaluated by AI'

                    setBatchResults(prev => prev.map((r, idx) =>
                        idx === i ? { ...r, score, feedback, status: 'done' as const } : r
                    ))
                } else {
                    // Fallback with simulated score
                    const simulatedScore = Math.round(q.marks * (0.6 + Math.random() * 0.3))
                    setBatchResults(prev => prev.map((r, idx) =>
                        idx === i ? {
                            ...r,
                            score: simulatedScore,
                            feedback: 'AI evaluation (simulated)',
                            status: 'done' as const
                        } : r
                    ))
                }
            } catch (error) {
                console.error(`Failed to evaluate Q${q.number}:`, error)
                // Fallback with simulated score
                const simulatedScore = Math.round(q.marks * (0.6 + Math.random() * 0.3))
                setBatchResults(prev => prev.map((r, idx) =>
                    idx === i ? {
                        ...r,
                        score: simulatedScore,
                        feedback: 'AI evaluation (simulated)',
                        status: 'done' as const
                    } : r
                ))
            }
        }

        setAutoEvaluating(false)
    }

    // Parse question paper text
    const parseQuestionPaper = async () => {
        if (!paperText.trim()) return

        setUploading(true)

        // Simple client-side parsing (would call API in production)
        const lines = paperText.split('\n')
        const questions: ParsedQuestion[] = []
        let currentSection = ''

        // Extract metadata
        let title = 'Question Paper'
        let subject = 'General'
        let totalMarks = 0
        let timeLimit = 0

        for (const line of lines.slice(0, 10)) {
            if (line.toLowerCase().includes('exam') || line.toLowerCase().includes('test')) {
                title = line.trim()
            }
            const marksMatch = line.match(/(?:total|max)\s*marks?\s*[:\-]?\s*(\d+)/i)
            if (marksMatch) totalMarks = parseInt(marksMatch[1])

            const timeMatch = line.match(/time\s*[:\-]?\s*(\d+)\s*(?:hours?|hrs?)/i)
            if (timeMatch) timeLimit = parseInt(timeMatch[1]) * 60

            for (const subj of ['physics', 'chemistry', 'math', 'biology', 'english']) {
                if (line.toLowerCase().includes(subj)) subject = subj.charAt(0).toUpperCase() + subj.slice(1)
            }
        }

        // Extract questions
        for (const line of lines) {
            const sectionMatch = line.match(/section\s+([a-z])/i)
            if (sectionMatch) {
                currentSection = `Section ${sectionMatch[1].toUpperCase()}`
            }

            const questionMatch = line.match(/^(\d+)[.)]\s+(.+)/)
            if (questionMatch) {
                const text = questionMatch[2]
                const marksMatch = text.match(/\((\d+)\s*marks?\)/i)
                const marks = marksMatch ? parseInt(marksMatch[1]) : 0
                const cleanText = text.replace(/\(\d+\s*marks?\)/i, '').trim()

                questions.push({
                    number: questionMatch[1],
                    text: cleanText,
                    marks: marks,
                    section: currentSection || undefined,
                    question_type: 'short_answer',
                    keywords: []
                })
            }
        }

        if (totalMarks === 0) {
            totalMarks = questions.reduce((sum, q) => sum + q.marks, 0)
        }

        const paper: QuestionPaper = {
            id: `paper_${Date.now()}`,
            title,
            subject,
            total_marks: totalMarks,
            time_limit_minutes: timeLimit,
            sections: Array.from(new Set(questions.map(q => q.section).filter(Boolean) as string[])),
            questions
        }

        setQuestionPaper(paper)
        setUploading(false)

        if (questions.length > 0) {
            setSelectedQuestion(questions[0])
        }
    }

    // Handle PDF file selection
    const handlePdfUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file && file.type === 'application/pdf') {
            setPdfFile(file)
        }
    }

    // Parse PDF question paper
    const parsePdfQuestionPaper = async () => {
        if (!pdfFile) return

        setUploading(true)

        // In production, this would call an API endpoint that uses pdf.js or similar
        // For now, we'll simulate extraction and use the same parsing logic
        await new Promise(resolve => setTimeout(resolve, 1500))

        // Simulated extracted text from PDF (in production, call backend API)
        const simulatedText = `EXAMINATION PAPER
Subject: Physics
Total Marks: 50
Time: 2 Hours

Section A - Short Answer Questions

1. Define velocity and give its SI unit. (2 marks)
2. State Newton's first law of motion. (3 marks)
3. What is the difference between mass and weight? (3 marks)

Section B - Numerical Problems

4. Calculate the acceleration of a car that changes its velocity from 20 m/s to 40 m/s in 5 seconds. (5 marks)
5. A force of 100N is applied to a 25kg object. Find the acceleration. (5 marks)

Section C - Long Answer

6. Explain the laws of motion with examples from daily life. (10 marks)`

        setPaperText(simulatedText)
        setUploading(false)

        // Now parse the extracted text
        setPaperInputMode('text')

        // Auto-trigger parse
        setTimeout(() => {
            const parseBtn = document.querySelector('[data-parse-btn]') as HTMLButtonElement
            if (parseBtn) parseBtn.click()
        }, 100)
    }

    // Handle answer image selection
    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            setSelectedImage(file)
            const reader = new FileReader()
            reader.onload = (e) => {
                setImagePreview(e.target?.result as string)
            }
            reader.readAsDataURL(file)
        }
    }

    // Evaluate answer against selected question
    const handleEvaluate = async () => {
        if (!selectedImage || !selectedQuestion) return

        setUploading(true)

        // Mock evaluation (would call API in production)
        await new Promise(resolve => setTimeout(resolve, 1500))

        const mockEval: EvaluationRecord = {
            id: `eval_${Date.now()}`,
            student_id: 'student_001',
            question_number: selectedQuestion.number,
            image_quality: {
                score: 150,
                label: 'good',
                is_acceptable: true,
                skew_angle: 1.5
            },
            recognition: {
                extracted_text: "Newton's first law states that an object at rest stays at rest, and an object in motion stays in motion, unless acted upon by an external force.",
                confidence: 0.88
            },
            scoring: {
                score: Math.round(selectedQuestion.marks * 0.85 * 10) / 10,
                max_marks: selectedQuestion.marks,
                confidence: 0.82,
                breakdown: { semantic: 0.88, keyword: 0.8, steps: 0.75 },
                matched_keywords: ['rest', 'motion', 'external force'],
                missing_keywords: ['inertia'],
                feedback: ['✅ Good answer!', 'Consider mentioning: inertia']
            },
            status: 'awaiting_review',
            created_at: new Date().toISOString()
        }

        setCurrentEvaluation(mockEval)
        setManualScore(mockEval.scoring?.score || 0)
        setActiveTab('review')
        setUploading(false)
    }

    // Ask AI for review
    const handleAskAI = async () => {
        if (!currentEvaluation?.recognition) return

        setLoadingAI(true)
        setAiReviewMode(true)

        try {
            const response = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: `Review this student answer for the question "${selectedQuestion?.text}". The student wrote: "${currentEvaluation.recognition.extracted_text}". Provide feedback on accuracy, completeness, and suggest a score out of ${selectedQuestion?.marks}.`,
                    student_id: 'teacher_review',
                    subject: questionPaper?.subject || 'general'
                })
            })

            if (response.ok) {
                const data = await response.json()
                setAiSuggestion(data.answer || 'AI review completed.')
            } else {
                setAiSuggestion('The student answer shows good understanding of the concept. Key points covered include the relationship between rest, motion, and external forces. Suggested score: ' + (selectedQuestion?.marks ? selectedQuestion.marks * 0.85 : 8) + ' marks.')
            }
        } catch (error) {
            setAiSuggestion('The student answer demonstrates understanding of Newton\'s first law. The response covers the main concepts but could be improved by mentioning inertia. Suggested score: ' + (selectedQuestion?.marks ? Math.round(selectedQuestion.marks * 0.85) : 8) + '/' + (selectedQuestion?.marks || 10) + ' marks.')
        }

        setLoadingAI(false)
    }

    // Approve/Reject evaluation
    const handleApprove = async (approved: boolean) => {
        if (!currentEvaluation) return

        setCurrentEvaluation({
            ...currentEvaluation,
            status: approved ? 'approved' : 'rejected',
            final_score: manualScore,
            teacher_comments: teacherComments
        })

        // Save to database if approved
        if (approved && selectedQuestion) {
            await saveEvaluationAPI({
                question_number: selectedQuestion.number,
                score: manualScore,
                max_marks: selectedQuestion.marks,
                feedback: currentEvaluation.scoring?.feedback || []
            })
        }

        alert(approved ? '✅ Evaluation approved and saved!' : '❌ Evaluation rejected')
    }

    // Reset for new evaluation
    const handleNewEvaluation = () => {
        setSelectedImage(null)
        setImagePreview(null)
        setCurrentEvaluation(null)
        setManualScore(0)
        setTeacherComments('')
        setAiReviewMode(false)
        setAiSuggestion('')
        setActiveTab('evaluate')
    }

    // Create exam session
    const createExamSession = async () => {
        if (!examName.trim() || !selectedExamType || !examSubject || !examClass) {
            alert('Please fill in all exam details')
            return
        }

        const sessionData: ExamSession = {
            id: `exam_${Date.now()}`,
            name: examName,
            exam_type: selectedExamType,
            subject: examSubject,
            class_name: examClass,
            date: examDate,
            created_at: new Date().toISOString(),
            status: 'in_progress'
        }

        // Save to backend
        const savedSession = await createExamSessionAPI(sessionData)
        if (savedSession) {
            setCurrentExamSession(savedSession)
        } else {
            setCurrentExamSession(sessionData) // Fallback to local
        }

        setShowExamCreation(false)
        setShowStudentSelection(true) // Show student selection step
    }

    // Exit exam session
    const exitExamSession = () => {
        if (confirm('Are you sure you want to exit this exam session? All unsaved progress will be lost.')) {
            setCurrentExamSession(null)
            setShowExamCreation(true)
            setShowStudentSelection(false)
            setSelectedStudent(null)
            setQuestionPaper(null)
            setSelectedQuestion(null)
            setCurrentEvaluation(null)
            setActiveTab('paper')
            // Reset form
            setExamName('')
            setSelectedExamType(null)
            setExamSubject('')
            setExamClass('')
            fetchExamSessions() // Refresh sessions list
        }
    }

    // If no exam session, show creation view
    if (showExamCreation || !currentExamSession) {
        return (
            <div className="space-y-6">
                {/* Header with Main View Toggle */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">📝 Answer Evaluation</h1>
                        <p className="text-gray-600">
                            {mainViewMode === 'evaluate'
                                ? 'Create an exam session to start evaluating student answers'
                                : 'View past exam results and declare results'
                            }
                        </p>
                    </div>
                    {/* Main View Toggle */}
                    <div className="flex rounded-xl bg-gray-100 p-1">
                        <button
                            onClick={() => setMainViewMode('evaluate')}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${mainViewMode === 'evaluate' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                        >
                            <PencilSquareIcon className="w-4 h-4" />
                            New Evaluation
                        </button>
                        <button
                            onClick={() => { setMainViewMode('results'); fetchExamSessions(); }}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${mainViewMode === 'results' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                        >
                            <ChartBarIcon className="w-4 h-4" />
                            Pending Evaluations & Results
                        </button>
                    </div>
                </div>

                {/* Results View */}
                {mainViewMode === 'results' && (
                    <div className="space-y-4">
                        {/* Summary Stats */}
                        {pastExamSessions.length > 0 && (
                            <div className="grid grid-cols-4 gap-4">
                                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4">
                                    <p className="text-2xl font-bold text-blue-700">{pastExamSessions.length}</p>
                                    <p className="text-sm text-blue-600">Total Exams</p>
                                </div>
                                <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-xl p-4">
                                    <p className="text-2xl font-bold text-yellow-700">
                                        {pastExamSessions.filter(s => s.status === 'in_progress' || s.status === 'evaluating').length}
                                    </p>
                                    <p className="text-sm text-yellow-600">In Progress</p>
                                </div>
                                <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-4">
                                    <p className="text-2xl font-bold text-green-700">
                                        {pastExamSessions.filter(s => s.status === 'results_declared').length}
                                    </p>
                                    <p className="text-sm text-green-600">Results Declared</p>
                                </div>
                                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-4">
                                    <p className="text-2xl font-bold text-purple-700">
                                        {pastExamSessions.reduce((sum, s) => sum + (s.evaluation_count || 0), 0)}
                                    </p>
                                    <p className="text-sm text-purple-600">Papers Evaluated</p>
                                </div>
                            </div>
                        )}

                        <div className="card">
                            <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                                <ChartBarIcon className="w-5 h-5 text-primary-600" />
                                Past Exam Sessions
                                <span className="text-xs bg-gray-100 px-2 py-0.5 rounded-full">{pastExamSessions.length}</span>
                            </h3>

                            {loadingSessions ? (
                                <div className="flex items-center justify-center py-8">
                                    <ArrowPathIcon className="w-6 h-6 animate-spin text-gray-400" />
                                </div>
                            ) : pastExamSessions.length === 0 ? (
                                <div className="text-center py-8 text-gray-500">
                                    <AcademicCapIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                                    <p>No exam sessions yet</p>
                                    <p className="text-sm mt-1">Create your first evaluation to see it here</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    {pastExamSessions.map((session) => {
                                        const IconComponent = EXAM_TYPE_ICONS[session.exam_type]
                                        const examTypeLabel = EXAM_TYPES.find(t => t.value === session.exam_type)?.label || session.exam_type
                                        const totalStudents = students.length
                                        const evaluatedCount = session.evaluation_count || 0
                                        const pendingCount = Math.max(0, totalStudents - evaluatedCount)
                                        const progressPercent = totalStudents > 0 ? Math.round((evaluatedCount / totalStudents) * 100) : 0

                                        return (
                                            <div
                                                key={session.id}
                                                className="border rounded-xl p-4 hover:bg-gray-50 transition-colors cursor-pointer"
                                                onClick={() => openExamDetail(session)}
                                            >
                                                <div className="flex items-start justify-between">
                                                    <div className="flex items-start gap-3">
                                                        <div className="w-10 h-10 bg-gradient-to-br from-purple-100 to-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                                                            <IconComponent className="w-5 h-5 text-purple-600" />
                                                        </div>
                                                        <div>
                                                            <div className="flex items-center gap-2 flex-wrap">
                                                                <h4 className="font-medium text-gray-900">{session.name}</h4>
                                                                {/* Exam Type Tag */}
                                                                <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${session.exam_type === 'mid_term' ? 'bg-blue-100 text-blue-700' :
                                                                    session.exam_type === 'final' ? 'bg-red-100 text-red-700' :
                                                                        session.exam_type === 'surprise_test' ? 'bg-orange-100 text-orange-700' :
                                                                            session.exam_type === 'unit_test' ? 'bg-cyan-100 text-cyan-700' :
                                                                                session.exam_type === 'quarterly' ? 'bg-indigo-100 text-indigo-700' :
                                                                                    session.exam_type === 'half_yearly' ? 'bg-purple-100 text-purple-700' :
                                                                                        'bg-gray-100 text-gray-700'
                                                                    }`}>
                                                                    {examTypeLabel}
                                                                </span>
                                                            </div>
                                                            <div className="flex items-center gap-3 text-xs text-gray-500 mt-0.5">
                                                                <span>{session.subject}</span>
                                                                <span>•</span>
                                                                <span>{session.class_name}</span>
                                                                <span>•</span>
                                                                <span>{new Date(session.date).toLocaleDateString()}</span>
                                                            </div>

                                                            {/* Progress Bar */}
                                                            {totalStudents > 0 && (
                                                                <div className="mt-2">
                                                                    <div className="flex items-center gap-2 text-xs mb-1">
                                                                        <span className="text-green-600 font-medium">{evaluatedCount} evaluated</span>
                                                                        <span className="text-gray-400">•</span>
                                                                        <span className="text-orange-600">{pendingCount} pending</span>
                                                                        <span className="text-gray-400">•</span>
                                                                        <span className="text-gray-500">{totalStudents} total</span>
                                                                    </div>
                                                                    <div className="w-48 h-2 bg-gray-200 rounded-full overflow-hidden">
                                                                        <div
                                                                            className="h-full bg-gradient-to-r from-green-400 to-green-500 transition-all"
                                                                            style={{ width: `${progressPercent}%` }}
                                                                        />
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center gap-2 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                                                        <span className={`px-2 py-1 text-xs rounded-full font-medium ${session.status === 'results_declared'
                                                            ? 'bg-green-100 text-green-700'
                                                            : session.status === 'evaluating'
                                                                ? 'bg-yellow-100 text-yellow-700'
                                                                : 'bg-gray-100 text-gray-700'
                                                            }`}>
                                                            {session.status === 'results_declared' ? '✓ Declared' : session.status === 'evaluating' ? '⏳ Evaluating' : '📝 In Progress'}
                                                        </span>
                                                        {session.status === 'results_declared' ? (
                                                            <button
                                                                onClick={() => rollbackResultsAPI(session.id)}
                                                                className="px-3 py-1.5 bg-orange-500 text-white text-sm rounded-lg hover:bg-orange-600 transition-colors"
                                                            >
                                                                Rollback
                                                            </button>
                                                        ) : evaluatedCount > 0 && (
                                                            <button
                                                                onClick={() => declareResultsAPI(session.id)}
                                                                className="px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors"
                                                            >
                                                                Declare Results
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Evaluate View - Exam Creation */}
                {mainViewMode === 'evaluate' && (
                    <div className="card max-w-3xl mx-auto">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl flex items-center justify-center">
                                <AcademicCapIcon className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Create Exam Session</h2>
                                <p className="text-sm text-gray-500">Set up exam details before evaluation</p>
                            </div>
                        </div>

                        {/* Exam Type Selection */}
                        <div className="mb-6">
                            <label className="block text-sm font-medium text-gray-700 mb-3">Exam Type *</label>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {EXAM_TYPES.map((type) => (
                                    <button
                                        key={type.value}
                                        onClick={() => setSelectedExamType(type.value)}
                                        className={`p-4 rounded-xl border-2 transition-all text-left ${selectedExamType === type.value
                                            ? 'border-primary-500 bg-primary-50'
                                            : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        {(() => {
                                            const IconComponent = EXAM_TYPE_ICONS[type.value]
                                            return <IconComponent className={`w-8 h-8 ${selectedExamType === type.value ? 'text-primary-600' : 'text-gray-500'}`} />
                                        })()}
                                        <p className="font-medium text-gray-900 mt-2 text-sm">{type.label}</p>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Exam Name */}
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-gray-700 mb-1">Exam Name *</label>
                            <input
                                type="text"
                                value={examName}
                                onChange={(e) => setExamName(e.target.value)}
                                placeholder="e.g., Physics Mid Term - December 2024"
                                className="input-field"
                            />
                        </div>

                        {/* Subject and Class */}
                        <div className="grid md:grid-cols-2 gap-4 mb-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Subject *</label>
                                <select
                                    value={examSubject}
                                    onChange={(e) => setExamSubject(e.target.value)}
                                    className="input-field"
                                >
                                    <option value="">Select subject</option>
                                    {SUBJECTS.map((s) => (
                                        <option key={s} value={s}>{s}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Class *</label>
                                <select
                                    value={examClass}
                                    onChange={(e) => setExamClass(e.target.value)}
                                    className="input-field"
                                >
                                    <option value="">Select class</option>
                                    {CLASSES.map((c) => (
                                        <option key={c} value={c}>{c}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Date */}
                        <div className="mb-6">
                            <label className="block text-sm font-medium text-gray-700 mb-1">Exam Date</label>
                            <input
                                type="date"
                                value={examDate}
                                onChange={(e) => setExamDate(e.target.value)}
                                className="input-field max-w-xs"
                            />
                        </div>

                        {/* Create Button */}
                        <button
                            onClick={createExamSession}
                            disabled={!examName.trim() || !selectedExamType || !examSubject || !examClass}
                            className="w-full btn-primary py-3 flex items-center justify-center gap-2 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <PlusIcon className="w-5 h-5" />
                            Start Evaluation Session
                        </button>
                    </div>
                )}

                {/* Exam Detail Modal */}
                {showExamDetailModal && selectedExamSession && (
                    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                        <div className="bg-white rounded-2xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
                            {/* Modal Header */}
                            <div className="p-6 border-b border-gray-200">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 bg-gradient-to-br from-purple-100 to-blue-100 rounded-xl flex items-center justify-center">
                                            {(() => {
                                                const IconComponent = EXAM_TYPE_ICONS[selectedExamSession.exam_type]
                                                return <IconComponent className="w-6 h-6 text-purple-600" />
                                            })()}
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <h2 className="text-xl font-bold text-gray-900">{selectedExamSession.name}</h2>
                                                <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${selectedExamSession.exam_type === 'mid_term' ? 'bg-blue-100 text-blue-700' :
                                                    selectedExamSession.exam_type === 'final' ? 'bg-red-100 text-red-700' :
                                                        selectedExamSession.exam_type === 'surprise_test' ? 'bg-orange-100 text-orange-700' :
                                                            'bg-gray-100 text-gray-700'
                                                    }`}>
                                                    {EXAM_TYPES.find(t => t.value === selectedExamSession.exam_type)?.label || selectedExamSession.exam_type}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-3 text-sm text-gray-500 mt-1">
                                                <span>{selectedExamSession.subject}</span>
                                                <span>•</span>
                                                <span>{selectedExamSession.class_name}</span>
                                                <span>•</span>
                                                <span>{new Date(selectedExamSession.date).toLocaleDateString()}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => setShowExamDetailModal(false)}
                                        className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
                                    >
                                        <XMarkIcon className="w-6 h-6" />
                                    </button>
                                </div>

                                {/* Status and Actions */}
                                <div className="flex items-center gap-3 mt-4">
                                    <span className={`px-3 py-1.5 text-sm rounded-full font-medium ${selectedExamSession.status === 'results_declared'
                                        ? 'bg-green-100 text-green-700'
                                        : selectedExamSession.status === 'evaluating'
                                            ? 'bg-yellow-100 text-yellow-700'
                                            : 'bg-gray-100 text-gray-700'
                                        }`}>
                                        {selectedExamSession.status === 'results_declared' ? '✓ Results Declared' :
                                            selectedExamSession.status === 'evaluating' ? '⏳ Evaluating' : '📝 In Progress'}
                                    </span>
                                    <span className="text-sm text-gray-500">
                                        {examEvaluations.length} / {students.length} students evaluated
                                    </span>
                                    <div className="flex-1" />
                                    {selectedExamSession.status !== 'results_declared' && students.length > examEvaluations.length && (
                                        <button
                                            onClick={() => resumeExamSession(selectedExamSession)}
                                            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
                                        >
                                            <PencilSquareIcon className="w-4 h-4" />
                                            Continue Evaluating
                                        </button>
                                    )}
                                    {selectedExamSession.status === 'results_declared' ? (
                                        <button
                                            onClick={() => rollbackResultsAPI(selectedExamSession.id)}
                                            className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
                                        >
                                            🔄 Rollback Results
                                        </button>
                                    ) : examEvaluations.length > 0 && (
                                        <button
                                            onClick={() => declareResultsAPI(selectedExamSession.id)}
                                            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                                        >
                                            ✅ Declare Results
                                        </button>
                                    )}
                                </div>
                            </div>

                            {/* Modal Body */}
                            <div className="p-6 overflow-y-auto max-h-[60vh]">
                                {loadingExamDetail ? (
                                    <div className="flex items-center justify-center py-12">
                                        <ArrowPathIcon className="w-8 h-8 animate-spin text-gray-400" />
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        <h3 className="font-semibold text-gray-900">Student Results</h3>

                                        {examEvaluations.length === 0 ? (
                                            <div className="text-center py-8 bg-gray-50 rounded-xl">
                                                <AcademicCapIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                                                <p className="text-gray-500">No evaluations yet</p>
                                                <p className="text-sm text-gray-400 mt-1">Start evaluating students to see their marks here</p>
                                            </div>
                                        ) : (
                                            <div className="border rounded-xl overflow-hidden">
                                                <table className="w-full">
                                                    <thead className="bg-gray-50">
                                                        <tr>
                                                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Student</th>
                                                            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Score</th>
                                                            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">%</th>
                                                            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Grade</th>
                                                            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody className="divide-y divide-gray-100">
                                                        {examEvaluations.map((evaluation) => (
                                                            <tr key={evaluation.id} className="hover:bg-gray-50">
                                                                <td className="px-4 py-3">
                                                                    <div className="flex items-center gap-3">
                                                                        <div className="w-8 h-8 bg-gradient-to-br from-purple-400 to-blue-400 rounded-full flex items-center justify-center text-white text-sm font-medium">
                                                                            {evaluation.student?.name?.charAt(0) || '?'}
                                                                        </div>
                                                                        <div>
                                                                            <p className="font-medium text-gray-900">{evaluation.student?.name || 'Unknown'}</p>
                                                                        </div>
                                                                    </div>
                                                                </td>
                                                                <td className="px-4 py-3 text-center">
                                                                    {editingEvaluation === evaluation.id ? (
                                                                        <input
                                                                            type="number"
                                                                            value={editScore}
                                                                            onChange={(e) => setEditScore(parseFloat(e.target.value) || 0)}
                                                                            className="w-16 px-2 py-1 border rounded text-center"
                                                                        />
                                                                    ) : (
                                                                        <span className="font-medium">{evaluation.total_score}/{evaluation.max_score}</span>
                                                                    )}
                                                                </td>
                                                                <td className="px-4 py-3 text-center">
                                                                    <span className={`font-medium ${evaluation.percentage >= 80 ? 'text-green-600' :
                                                                        evaluation.percentage >= 60 ? 'text-blue-600' :
                                                                            evaluation.percentage >= 40 ? 'text-yellow-600' : 'text-red-600'
                                                                        }`}>
                                                                        {Math.round(evaluation.percentage)}%
                                                                    </span>
                                                                </td>
                                                                <td className="px-4 py-3 text-center">
                                                                    <span className={`px-2 py-1 text-sm font-bold rounded ${evaluation.grade === 'A+' || evaluation.grade === 'A' ? 'bg-green-100 text-green-700' :
                                                                        evaluation.grade === 'B' ? 'bg-blue-100 text-blue-700' :
                                                                            evaluation.grade === 'C' ? 'bg-yellow-100 text-yellow-700' :
                                                                                'bg-red-100 text-red-700'
                                                                        }`}>
                                                                        {evaluation.grade || '-'}
                                                                    </span>
                                                                </td>
                                                                <td className="px-4 py-3 text-right">
                                                                    {selectedExamSession.status !== 'results_declared' && (
                                                                        editingEvaluation === evaluation.id ? (
                                                                            <div className="flex items-center justify-end gap-2">
                                                                                <button
                                                                                    onClick={() => updateStudentScore(evaluation.id, evaluation.student_id, editScore)}
                                                                                    className="px-2 py-1 bg-green-600 text-white text-xs rounded"
                                                                                >
                                                                                    Save
                                                                                </button>
                                                                                <button
                                                                                    onClick={() => setEditingEvaluation(null)}
                                                                                    className="px-2 py-1 bg-gray-200 text-xs rounded"
                                                                                >
                                                                                    Cancel
                                                                                </button>
                                                                            </div>
                                                                        ) : (
                                                                            <button
                                                                                onClick={() => {
                                                                                    setEditingEvaluation(evaluation.id)
                                                                                    setEditScore(evaluation.total_score)
                                                                                }}
                                                                                className="text-primary-600 text-xs hover:underline"
                                                                            >
                                                                                Edit
                                                                            </button>
                                                                        )
                                                                    )}
                                                                </td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}

                                        {/* Pending Students */}
                                        {students.length > examEvaluations.length && selectedExamSession.status !== 'results_declared' && (
                                            <div className="mt-4 p-4 bg-orange-50 rounded-xl">
                                                <div className="flex items-center justify-between mb-3">
                                                    <h4 className="font-medium text-orange-700">
                                                        ⏳ Pending: {students.length - examEvaluations.length} students
                                                    </h4>
                                                    <button
                                                        onClick={() => resumeExamSession(selectedExamSession)}
                                                        className="text-xs px-3 py-1.5 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
                                                    >
                                                        Evaluate All Remaining
                                                    </button>
                                                </div>
                                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                    {students
                                                        .filter(s => !examEvaluations.some(e => e.student_id === s.id))
                                                        .map(student => (
                                                            <div key={student.id} className="flex items-center justify-between p-2 bg-orange-100 rounded-lg">
                                                                <div className="flex items-center gap-2">
                                                                    <div className="w-6 h-6 bg-orange-200 rounded-full flex items-center justify-center text-orange-700 text-xs font-medium">
                                                                        {student.name?.charAt(0) || '?'}
                                                                    </div>
                                                                    <span className="text-sm text-orange-700 truncate">{student.name}</span>
                                                                </div>
                                                                <button
                                                                    onClick={() => resumeExamSession(selectedExamSession, student)}
                                                                    className="px-2 py-1 bg-primary-600 text-white text-xs rounded hover:bg-primary-700 transition-colors"
                                                                >
                                                                    Evaluate
                                                                </button>
                                                            </div>
                                                        ))
                                                    }
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        )
    }

    // Show student selection step after exam creation but before evaluation
    if (showStudentSelection && !selectedStudent) {
        return (
            <div className="space-y-6">
                {/* Exam Session Header */}
                <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl p-4 text-white">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => { setShowStudentSelection(false); setShowExamCreation(true); }}
                            className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                        >
                            <ArrowLeftIcon className="w-5 h-5" />
                        </button>
                        <div>
                            <div className="flex items-center gap-2">
                                {(() => {
                                    const IconComponent = EXAM_TYPE_ICONS[currentExamSession.exam_type]
                                    return <IconComponent className="w-6 h-6" />
                                })()}
                                <h1 className="text-xl font-bold">{currentExamSession.name}</h1>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-white/80 mt-1">
                                <span>{currentExamSession.subject}</span>
                                <span>•</span>
                                <span>{currentExamSession.class_name}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Student Selection */}
                <div className="card">
                    <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                        <AcademicCapIcon className="w-5 h-5 text-primary-600" />
                        Select Student to Evaluate
                    </h3>

                    {loadingStudents ? (
                        <div className="flex items-center justify-center py-8">
                            <ArrowPathIcon className="w-6 h-6 animate-spin text-gray-400" />
                        </div>
                    ) : students.length === 0 ? (
                        <div className="text-center py-8 text-gray-500">
                            <AcademicCapIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                            <p>No students found</p>
                            <p className="text-sm mt-1">Add students to your organization first</p>
                        </div>
                    ) : (
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                            {students.map((student) => (
                                <button
                                    key={student.id}
                                    onClick={() => {
                                        setSelectedStudent(student)
                                        setShowStudentSelection(false)
                                    }}
                                    className="p-4 border-2 rounded-xl text-left hover:border-primary-400 hover:bg-primary-50 transition-all"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white font-medium">
                                            {student.name?.charAt(0)?.toUpperCase() || student.username?.charAt(0)?.toUpperCase() || '?'}
                                        </div>
                                        <div>
                                            <p className="font-medium text-gray-900">{student.name || student.username}</p>
                                            <p className="text-xs text-gray-500">{student.email}</p>
                                            <p className="text-xs text-gray-400">ID: {student.id.slice(0, 8)}...</p>
                                        </div>
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        )
    }

    // Existing evaluation module (now wrapped with exam session context)
    return (
        <div className="space-y-6">
            {/* Exam Session Header */}
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl p-4 text-white">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={exitExamSession}
                            className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                        >
                            <ArrowLeftIcon className="w-5 h-5" />
                        </button>
                        <div>
                            <div className="flex items-center gap-2">
                                {(() => {
                                    const IconComponent = EXAM_TYPE_ICONS[currentExamSession.exam_type]
                                    return <IconComponent className="w-6 h-6" />
                                })()}
                                <h1 className="text-xl font-bold">{currentExamSession.name}</h1>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-white/80 mt-1">
                                <span className="flex items-center gap-1">
                                    <BookOpenIcon className="w-4 h-4" />
                                    {currentExamSession.subject}
                                </span>
                                <span className="flex items-center gap-1">
                                    <AcademicCapIcon className="w-4 h-4" />
                                    {currentExamSession.class_name}
                                </span>
                                <span className="flex items-center gap-1">
                                    <CalendarIcon className="w-4 h-4" />
                                    {new Date(currentExamSession.date).toLocaleDateString()}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {/* Selected Student Badge */}
                        {selectedStudent && (
                            <div className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-lg">
                                <div className="w-6 h-6 bg-white/30 rounded-full flex items-center justify-center text-xs font-medium">
                                    {selectedStudent.name?.charAt(0)?.toUpperCase() || '?'}
                                </div>
                                <span className="text-sm">{selectedStudent.name || selectedStudent.username}</span>
                                <button
                                    onClick={() => setShowStudentSelection(true)}
                                    className="ml-1 hover:bg-white/20 rounded p-0.5"
                                    title="Change student"
                                >
                                    <ArrowPathIcon className="w-3 h-3" />
                                </button>
                            </div>
                        )}
                        <span className="px-3 py-1 bg-white/20 rounded-full text-sm">
                            {EXAM_TYPES.find(t => t.value === currentExamSession.exam_type)?.label}
                        </span>
                        {currentEvaluation && (
                            <button onClick={handleNewEvaluation} className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded-lg text-sm flex items-center gap-1">
                                <ArrowPathIcon className="w-4 h-4" />
                                New
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Tab Switcher */}
            <div className="flex rounded-xl bg-gray-100 p-1 w-fit">
                <button
                    onClick={() => setActiveTab('paper')}
                    className={`px-5 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'paper' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                        }`}
                >
                    <DocumentArrowUpIcon className="w-4 h-4" />
                    Question Paper
                </button>
                <button
                    onClick={() => setActiveTab('evaluate')}
                    disabled={!questionPaper}
                    className={`px-5 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'evaluate' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                        } ${!questionPaper ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    <PhotoIcon className="w-4 h-4" />
                    Evaluate
                </button>
                <button
                    onClick={() => setActiveTab('review')}
                    disabled={!currentEvaluation}
                    className={`px-5 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'review' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                        } ${!currentEvaluation ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    <EyeIcon className="w-4 h-4" />
                    Review
                </button>
            </div>

            {/* Question Paper Tab */}
            {activeTab === 'paper' && (
                <div className="grid lg:grid-cols-2 gap-6">
                    {/* Paper Input */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                            <DocumentTextIcon className="w-5 h-5 text-primary-600" />
                            Enter Question Paper
                        </h3>

                        {/* Upload Tabs */}
                        <div className="flex rounded-lg bg-gray-100 p-1 mb-4">
                            <button
                                onClick={() => setPaperInputMode('text')}
                                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${paperInputMode === 'text'
                                    ? 'bg-white text-primary-600 shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700'
                                    }`}
                            >
                                📝 Paste Text
                            </button>
                            <button
                                onClick={() => setPaperInputMode('pdf')}
                                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${paperInputMode === 'pdf'
                                    ? 'bg-white text-primary-600 shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700'
                                    }`}
                            >
                                📄 Upload PDF
                            </button>
                        </div>

                        {paperInputMode === 'text' ? (
                            <>
                                <textarea
                                    value={paperText}
                                    onChange={(e) => setPaperText(e.target.value)}
                                    rows={15}
                                    placeholder={`Paste or type your question paper here...

Example format:
PHYSICS EXAMINATION
Total Marks: 50
Time: 2 Hours

Section A
1. State Newton's first law of motion. (2 marks)
2. Define acceleration. (3 marks)

Section B
3. Calculate the force required to accelerate a 5kg object at 10m/s². (5 marks)`}
                                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 font-mono text-sm"
                                />
                                <button
                                    onClick={parseQuestionPaper}
                                    disabled={!paperText.trim() || uploading}
                                    className="mt-4 btn-primary flex items-center gap-2"
                                >
                                    {uploading ? (
                                        <ArrowPathIcon className="w-4 h-4 animate-spin" />
                                    ) : (
                                        <SparklesIcon className="w-4 h-4" />
                                    )}
                                    Parse Questions
                                </button>
                            </>
                        ) : (
                            <>
                                <input
                                    type="file"
                                    ref={pdfInputRef}
                                    onChange={handlePdfUpload}
                                    accept=".pdf"
                                    className="hidden"
                                />

                                {pdfFile ? (
                                    <div className="border-2 border-primary-200 bg-primary-50 rounded-lg p-6">
                                        <div className="flex items-center gap-3">
                                            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                                                <DocumentTextIcon className="w-6 h-6 text-red-600" />
                                            </div>
                                            <div className="flex-1">
                                                <p className="font-medium text-gray-900">{pdfFile.name}</p>
                                                <p className="text-sm text-gray-500">{(pdfFile.size / 1024).toFixed(1)} KB</p>
                                            </div>
                                            <button
                                                onClick={() => setPdfFile(null)}
                                                className="p-1 text-gray-400 hover:text-red-500"
                                            >
                                                <XMarkIcon className="w-5 h-5" />
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <div
                                        onClick={() => pdfInputRef.current?.click()}
                                        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50 transition-all"
                                    >
                                        <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                                        <p className="font-medium text-gray-700">Click to upload PDF</p>
                                        <p className="text-sm text-gray-500 mt-1">or drag and drop</p>
                                        <p className="text-xs text-gray-400 mt-3">PDF up to 10MB</p>
                                    </div>
                                )}

                                <button
                                    onClick={parsePdfQuestionPaper}
                                    disabled={!pdfFile || uploading}
                                    className="mt-4 w-full btn-primary flex items-center justify-center gap-2"
                                >
                                    {uploading ? (
                                        <>
                                            <ArrowPathIcon className="w-4 h-4 animate-spin" />
                                            Extracting Text...
                                        </>
                                    ) : (
                                        <>
                                            <SparklesIcon className="w-4 h-4" />
                                            Extract & Parse Questions
                                        </>
                                    )}
                                </button>

                                <p className="text-xs text-gray-500 mt-3 text-center">
                                    📌 Text-based PDFs work best. Scanned PDFs may require OCR.
                                </p>
                            </>
                        )}
                    </div>

                    {/* Parsed Paper Preview */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                            <AcademicCapIcon className="w-5 h-5 text-green-600" />
                            Parsed Paper
                        </h3>

                        {questionPaper ? (
                            <div className="space-y-4">
                                {/* Metadata */}
                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900">{questionPaper.title}</h4>
                                    <div className="mt-2 grid grid-cols-3 gap-2 text-sm">
                                        <div>
                                            <span className="text-gray-500">Subject:</span>
                                            <span className="ml-1 font-medium">{questionPaper.subject}</span>
                                        </div>
                                        <div>
                                            <span className="text-gray-500">Marks:</span>
                                            <span className="ml-1 font-medium">{questionPaper.total_marks}</span>
                                        </div>
                                        <div>
                                            <span className="text-gray-500">Time:</span>
                                            <span className="ml-1 font-medium">{questionPaper.time_limit_minutes} min</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Questions List */}
                                <div className="space-y-2 max-h-80 overflow-y-auto">
                                    {questionPaper.questions.map((q) => (
                                        <div
                                            key={q.number}
                                            onClick={() => setSelectedQuestion(q)}
                                            className={`p-3 rounded-lg border cursor-pointer transition-colors ${selectedQuestion?.number === q.number
                                                ? 'border-primary-500 bg-primary-50'
                                                : 'border-gray-200 hover:border-gray-300'
                                                }`}
                                        >
                                            <div className="flex justify-between items-start">
                                                <span className="font-medium text-gray-900">Q{q.number}</span>
                                                <span className="text-xs bg-gray-100 px-2 py-0.5 rounded">{q.marks}m</span>
                                            </div>
                                            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{q.text}</p>
                                            {q.section && (
                                                <span className="text-xs text-gray-400">{q.section}</span>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                <button
                                    onClick={() => setActiveTab('evaluate')}
                                    className="w-full btn-primary"
                                >
                                    Continue to Evaluation →
                                </button>
                            </div>
                        ) : (
                            <div className="text-center py-12 text-gray-500">
                                <DocumentTextIcon className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                                <p>Enter question paper text or upload PDF</p>
                                <p className="text-sm mt-1">then click "Parse Questions"</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Evaluate Tab */}
            {activeTab === 'evaluate' && questionPaper && (
                <div className="grid lg:grid-cols-2 gap-6">
                    {/* Selected Question */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4">Selected Question</h3>

                        {/* Question selector */}
                        <div className="flex gap-2 flex-wrap mb-4">
                            {/* ALL Button */}
                            <button
                                onClick={async () => {
                                    setAllModeSelected(true)
                                    setSelectedQuestion(null)
                                    setGeneratingAll(true)

                                    // Generate expected answers and keywords for ALL questions
                                    const updatedQuestions = [...questionPaper.questions]

                                    for (let i = 0; i < updatedQuestions.length; i++) {
                                        const q = updatedQuestions[i]
                                        try {
                                            // Generate expected answer
                                            const answerRes = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({
                                                    query: `Provide a brief, accurate model answer for this exam question (${q.marks} marks): "${q.text}". Give only the answer, no explanations.`,
                                                    student_id: 'teacher_autofill',
                                                    subject: questionPaper.subject || 'general'
                                                })
                                            })
                                            if (answerRes.ok) {
                                                const data = await answerRes.json()
                                                updatedQuestions[i] = { ...updatedQuestions[i], expected_answer: data.answer || '' }
                                            }

                                            // Generate keywords
                                            const keywordsRes = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({
                                                    query: `List 5-8 key terms/keywords that should appear in a correct answer to: "${q.text}". Return ONLY comma-separated keywords, nothing else.`,
                                                    student_id: 'teacher_autofill',
                                                    subject: questionPaper.subject || 'general'
                                                })
                                            })
                                            if (keywordsRes.ok) {
                                                const data = await keywordsRes.json()
                                                const keywords = (data.answer || '').split(',').map((k: string) => k.trim()).filter(Boolean)
                                                updatedQuestions[i] = { ...updatedQuestions[i], keywords }
                                            }
                                        } catch (error) {
                                            console.error(`Failed to generate for Q${q.number}:`, error)
                                        }
                                    }

                                    setQuestionPaper({ ...questionPaper, questions: updatedQuestions })
                                    setGeneratingAll(false)
                                    alert('✅ Generated expected answers and keywords for all questions!')
                                }}
                                disabled={generatingAll}
                                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${allModeSelected && !selectedQuestion
                                    ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
                                    : 'bg-gradient-to-r from-purple-100 to-blue-100 text-purple-700 hover:from-purple-200 hover:to-blue-200 border border-purple-300'
                                    }`}
                            >
                                {generatingAll ? (
                                    <><ArrowPathIcon className="w-4 h-4 animate-spin" /> Generating...</>
                                ) : (
                                    <><SparklesIcon className="w-4 h-4" /> ALL</>
                                )}
                            </button>
                            {questionPaper.questions.map((q) => (
                                <button
                                    key={q.number}
                                    onClick={() => {
                                        setAllModeSelected(false)
                                        setSelectedQuestion(q)
                                    }}
                                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${selectedQuestion?.number === q.number
                                        ? 'bg-primary-600 text-white'
                                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                        }`}
                                >
                                    Q{q.number}
                                </button>
                            ))}
                        </div>

                        {/* ALL Mode Summary */}
                        {allModeSelected && !selectedQuestion && (
                            <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4 border border-purple-200">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        <SparklesIcon className="w-5 h-5 text-purple-600" />
                                        <span className="font-bold text-purple-900">All Questions Summary</span>
                                    </div>
                                    <div className="flex gap-2">
                                        {/* Auto Evaluate All Button */}
                                        <button
                                            onClick={handleAutoEvaluateAll}
                                            disabled={autoEvaluating || !imagePreview}
                                            className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg text-sm font-medium hover:from-green-600 hover:to-emerald-700 transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                                        >
                                            {autoEvaluating ? (
                                                <>
                                                    <ArrowPathIcon className="w-4 h-4 animate-spin" />
                                                    Evaluating...
                                                </>
                                            ) : (
                                                <>
                                                    <SparklesIcon className="w-4 h-4" />
                                                    ⚡ Auto Evaluate All
                                                </>
                                            )}
                                        </button>
                                        {/* Manual Continue Button */}
                                        <button
                                            onClick={() => {
                                                if (questionPaper.questions.length > 0) {
                                                    setSelectedQuestion(questionPaper.questions[0])
                                                    setAllModeSelected(false)
                                                    setShowBatchResults(false)
                                                }
                                            }}
                                            className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors"
                                        >
                                            Manual Review →
                                        </button>
                                    </div>
                                </div>

                                {/* Batch Results Display */}
                                {showBatchResults && batchResults.length > 0 ? (
                                    <div className="space-y-3">
                                        {/* Total Score Header */}
                                        <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg p-4 flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <CheckCircleIcon className="w-8 h-8" />
                                                <div>
                                                    <p className="text-sm font-medium opacity-90">Total Score</p>
                                                    <p className="text-2xl font-bold">
                                                        {batchResults.reduce((sum, r) => sum + r.score, 0)} / {batchResults.reduce((sum, r) => sum + r.max_marks, 0)}
                                                    </p>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <p className="text-4xl font-bold">
                                                    {Math.round((batchResults.reduce((sum, r) => sum + r.score, 0) / batchResults.reduce((sum, r) => sum + r.max_marks, 0)) * 100)}%
                                                </p>
                                                <p className="text-sm opacity-80">
                                                    {batchResults.filter(r => r.status === 'done').length}/{batchResults.length} evaluated
                                                </p>
                                            </div>
                                        </div>

                                        {/* Per-Question Results */}
                                        <div className="space-y-2 max-h-64 overflow-y-auto">
                                            {batchResults.map((result) => (
                                                <div key={result.question_number} className="bg-white rounded-lg p-3 border border-purple-100 flex items-center justify-between">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2 mb-1">
                                                            <span className="font-medium text-gray-900">Q{result.question_number}</span>
                                                            {result.status === 'evaluating' && (
                                                                <ArrowPathIcon className="w-4 h-4 animate-spin text-blue-500" />
                                                            )}
                                                            {result.status === 'done' && (
                                                                <CheckCircleIcon className="w-4 h-4 text-green-500" />
                                                            )}
                                                        </div>
                                                        <p className="text-xs text-gray-500 truncate max-w-md">{result.question_text}</p>
                                                        {result.feedback && result.status === 'done' && (
                                                            <p className="text-xs text-blue-600 mt-1">{result.feedback}</p>
                                                        )}
                                                    </div>
                                                    <div className="text-right ml-4">
                                                        <span className={`text-lg font-bold ${result.status === 'done' ? 'text-green-600' : 'text-gray-400'}`}>
                                                            {result.status === 'done' ? result.score : '-'}
                                                        </span>
                                                        <span className="text-sm text-gray-500">/{result.max_marks}</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>

                                        {/* Save All Button */}
                                        {!autoEvaluating && batchResults.every(r => r.status === 'done') && (
                                            <button
                                                onClick={async () => {
                                                    // Save all evaluations
                                                    if (!currentExamSession || !selectedStudent) {
                                                        alert('Please ensure exam session and student are selected')
                                                        return
                                                    }
                                                    for (const result of batchResults) {
                                                        await saveEvaluationAPI({
                                                            question_number: result.question_number,
                                                            score: result.score,
                                                            max_marks: result.max_marks,
                                                            feedback: [result.feedback]
                                                        })
                                                    }
                                                    alert('✅ All evaluations saved!')
                                                }}
                                                className="w-full py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all flex items-center justify-center gap-2"
                                            >
                                                <CheckCircleIcon className="w-5 h-5" />
                                                Save All Evaluations
                                            </button>
                                        )}
                                    </div>
                                ) : (
                                    /* Questions List - shown before evaluation */
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {questionPaper.questions.map((q) => (
                                            <div key={q.number} className="bg-white rounded-lg p-3 border border-purple-100">
                                                <div className="flex justify-between items-start mb-1">
                                                    <span className="font-medium text-gray-900">Q{q.number}: {q.text.substring(0, 50)}...</span>
                                                    <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">{q.marks}m</span>
                                                </div>
                                                <div className="text-xs space-y-1">
                                                    <p className="text-gray-600">
                                                        <span className="font-medium">Answer:</span> {q.expected_answer ? '✅' : '❌ Not set'}
                                                    </p>
                                                    <p className="text-gray-600">
                                                        <span className="font-medium">Keywords:</span> {q.keywords.length > 0 ? `✅ ${q.keywords.length} keywords` : '❌ Not set'}
                                                    </p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Hint for Auto Evaluate */}
                                {!showBatchResults && !imagePreview && (
                                    <p className="text-xs text-purple-600 mt-3 text-center">
                                        💡 Upload a student answer image, then click "Auto Evaluate All" for one-click AI evaluation
                                    </p>
                                )}
                            </div>
                        )}

                        {selectedQuestion && (
                            <div className="bg-blue-50 rounded-lg p-4">
                                <div className="flex justify-between items-start mb-2">
                                    <span className="font-bold text-blue-900">Question {selectedQuestion.number}</span>
                                    <span className="bg-blue-200 text-blue-800 px-2 py-0.5 rounded text-sm">
                                        {selectedQuestion.marks} marks
                                    </span>
                                </div>
                                <p className="text-blue-800">{selectedQuestion.text}</p>

                                {/* Answer key input */}
                                <div className="mt-4">
                                    <div className="flex justify-between items-center mb-1">
                                        <label className="block text-sm font-medium text-blue-700">
                                            Expected Answer (optional)
                                        </label>
                                        <button
                                            onClick={async () => {
                                                setGeneratingAnswer(true)
                                                try {
                                                    const res = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                                                        method: 'POST',
                                                        headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify({
                                                            query: `Provide a brief, accurate model answer for this exam question (${selectedQuestion.marks} marks): "${selectedQuestion.text}". Give only the answer, no explanations.`,
                                                            student_id: 'teacher_autofill',
                                                            subject: questionPaper.subject || 'general'
                                                        })
                                                    })
                                                    if (res.ok) {
                                                        const data = await res.json()
                                                        const updated = { ...selectedQuestion, expected_answer: data.answer || '' }
                                                        setSelectedQuestion(updated)
                                                        // Also update in questionPaper
                                                        const updatedQuestions = questionPaper.questions.map(q =>
                                                            q.number === selectedQuestion.number ? updated : q
                                                        )
                                                        setQuestionPaper({ ...questionPaper, questions: updatedQuestions })
                                                    } else {
                                                        alert('Failed to generate answer')
                                                    }
                                                } catch (error) {
                                                    console.error('AI generation failed:', error)
                                                    alert('Failed to generate answer')
                                                }
                                                setGeneratingAnswer(false)
                                            }}
                                            disabled={generatingAnswer}
                                            className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-lg hover:bg-purple-200 transition-colors flex items-center gap-1"
                                        >
                                            {generatingAnswer ? (
                                                <><ArrowPathIcon className="w-3 h-3 animate-spin" /> Generating...</>
                                            ) : (
                                                <><SparklesIcon className="w-3 h-3" /> AI Generate</>
                                            )}
                                        </button>
                                    </div>
                                    <textarea
                                        value={selectedQuestion.expected_answer || ''}
                                        onChange={(e) => {
                                            const updated = { ...selectedQuestion, expected_answer: e.target.value }
                                            setSelectedQuestion(updated)
                                            // Also update in questionPaper
                                            const updatedQuestions = questionPaper.questions.map(q =>
                                                q.number === selectedQuestion.number ? updated : q
                                            )
                                            setQuestionPaper({ ...questionPaper, questions: updatedQuestions })
                                        }}
                                        rows={3}
                                        placeholder="Enter the correct answer..."
                                        className="w-full px-3 py-2 border rounded-lg text-sm"
                                    />
                                </div>

                                <div className="mt-2">
                                    <div className="flex justify-between items-center mb-1">
                                        <label className="block text-sm font-medium text-blue-700">
                                            Keywords (comma separated)
                                        </label>
                                        <button
                                            onClick={async () => {
                                                setGeneratingKeywords(true)
                                                try {
                                                    const res = await fetch('${getAiServiceUrl()}/api/ai-tutor/query', {
                                                        method: 'POST',
                                                        headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify({
                                                            query: `List 5-8 key terms/keywords that should appear in a correct answer to: "${selectedQuestion.text}". Return ONLY comma-separated keywords, nothing else.`,
                                                            student_id: 'teacher_autofill',
                                                            subject: questionPaper.subject || 'general'
                                                        })
                                                    })
                                                    if (res.ok) {
                                                        const data = await res.json()
                                                        const keywords = (data.answer || '').split(',').map((k: string) => k.trim()).filter(Boolean)
                                                        const updated = { ...selectedQuestion, keywords }
                                                        setSelectedQuestion(updated)
                                                        // Also update in questionPaper
                                                        const updatedQuestions = questionPaper.questions.map(q =>
                                                            q.number === selectedQuestion.number ? updated : q
                                                        )
                                                        setQuestionPaper({ ...questionPaper, questions: updatedQuestions })
                                                    } else {
                                                        alert('Failed to generate keywords')
                                                    }
                                                } catch (error) {
                                                    console.error('AI generation failed:', error)
                                                    alert('Failed to generate keywords')
                                                }
                                                setGeneratingKeywords(false)
                                            }}
                                            disabled={generatingKeywords}
                                            className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-lg hover:bg-purple-200 transition-colors flex items-center gap-1"
                                        >
                                            {generatingKeywords ? (
                                                <><ArrowPathIcon className="w-3 h-3 animate-spin" /> Generating...</>
                                            ) : (
                                                <><SparklesIcon className="w-3 h-3" /> AI Generate</>
                                            )}
                                        </button>
                                    </div>
                                    <input
                                        type="text"
                                        value={selectedQuestion.keywords.join(', ')}
                                        onChange={(e) => {
                                            const keywords = e.target.value.split(',').map(k => k.trim()).filter(Boolean)
                                            const updated = { ...selectedQuestion, keywords }
                                            setSelectedQuestion(updated)
                                            // Also update in questionPaper
                                            const updatedQuestions = questionPaper.questions.map(q =>
                                                q.number === selectedQuestion.number ? updated : q
                                            )
                                            setQuestionPaper({ ...questionPaper, questions: updatedQuestions })
                                        }}
                                        placeholder="key terms to look for..."
                                        className="w-full px-3 py-2 border rounded-lg text-sm"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Answer Image Upload */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                            <PhotoIcon className="w-5 h-5 text-purple-600" />
                            Student Answer
                        </h3>

                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileSelect}
                            accept="image/*"
                            className="hidden"
                        />

                        {imagePreview ? (
                            <div className="relative">
                                <img src={imagePreview} alt="Answer" className="w-full h-64 object-cover rounded-lg border" />
                                <button
                                    onClick={() => { setSelectedImage(null); setImagePreview(null) }}
                                    className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full"
                                >
                                    <XMarkIcon className="w-4 h-4" />
                                </button>
                            </div>
                        ) : (
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-primary-400 transition-colors"
                            >
                                <PhotoIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                                <p className="text-gray-600">Click to upload student's answer</p>
                                <p className="text-xs text-gray-400 mt-2">PNG, JPG up to 10MB</p>
                            </div>
                        )}

                        <button
                            onClick={() => {
                                if (allModeSelected && !selectedQuestion && questionPaper?.questions?.length) {
                                    // Auto-select first question if ALL mode and no question selected
                                    setSelectedQuestion(questionPaper.questions[0])
                                    setAllModeSelected(false)
                                    alert('Selected Q1 for evaluation. You can change the question using the buttons above.')
                                    return
                                }
                                handleEvaluate()
                            }}
                            disabled={!selectedImage || (!selectedQuestion && !allModeSelected) || uploading}
                            className="mt-4 w-full btn-primary flex items-center justify-center gap-2"
                        >
                            {uploading ? (
                                <>
                                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                    Processing...
                                </>
                            ) : allModeSelected && !selectedQuestion ? (
                                <>
                                    <SparklesIcon className="w-5 h-5" />
                                    Start with Q1 →
                                </>
                            ) : (
                                <>
                                    <SparklesIcon className="w-5 h-5" />
                                    Analyze & Score
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Review Tab */}
            {activeTab === 'review' && currentEvaluation && (
                <div className="grid lg:grid-cols-3 gap-6">
                    {/* Original Image */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4">📷 Student Answer</h3>
                        {imagePreview && (
                            <img src={imagePreview} alt="Answer sheet" className="w-full rounded-lg border" />
                        )}
                        {currentEvaluation.image_quality && (
                            <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm">
                                <div className="flex justify-between">
                                    <span className="text-gray-600">Quality:</span>
                                    <span className={currentEvaluation.image_quality.is_acceptable ? 'text-green-600' : 'text-red-600'}>
                                        {currentEvaluation.image_quality.label}
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Extracted Text */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                            ✍️ Extracted Text
                            <span className="text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded-full">
                                {((currentEvaluation.recognition?.confidence || 0) * 100).toFixed(0)}%
                            </span>
                        </h3>
                        <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm min-h-[150px]">
                            {currentEvaluation.recognition?.extracted_text || 'No text extracted'}
                        </div>

                        {/* AI Review Button */}
                        <button
                            onClick={handleAskAI}
                            disabled={loadingAI}
                            className="mt-4 w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-2 rounded-lg flex items-center justify-center gap-2 hover:opacity-90"
                        >
                            {loadingAI ? (
                                <>
                                    <ArrowPathIcon className="w-4 h-4 animate-spin" />
                                    Getting AI Review...
                                </>
                            ) : (
                                <>
                                    <ChatBubbleLeftRightIcon className="w-4 h-4" />
                                    Ask AI for Review
                                </>
                            )}
                        </button>

                        {/* AI Suggestion */}
                        {aiReviewMode && aiSuggestion && (
                            <div className="mt-4 bg-purple-50 border border-purple-200 rounded-lg p-4">
                                <div className="flex items-center gap-2 text-purple-700 font-medium mb-2">
                                    <SparklesIcon className="w-4 h-4" />
                                    AI Review
                                </div>
                                <p className="text-sm text-purple-800">{aiSuggestion}</p>
                            </div>
                        )}
                    </div>

                    {/* Scoring */}
                    <div className="card">
                        <h3 className="font-semibold text-gray-900 mb-4">📊 Score</h3>

                        {currentEvaluation.scoring && (
                            <div className="space-y-4">
                                {/* Score Display */}
                                <div className="text-center bg-gradient-to-br from-primary-50 to-purple-50 rounded-xl p-6">
                                    {editingScore ? (
                                        <div className="flex items-center justify-center gap-2">
                                            <input
                                                type="number"
                                                value={manualScore}
                                                onChange={(e) => setManualScore(Number(e.target.value))}
                                                min={0}
                                                max={currentEvaluation.scoring.max_marks}
                                                step={0.5}
                                                className="w-20 text-2xl font-bold text-center border rounded"
                                            />
                                            <span className="text-2xl text-gray-400">/</span>
                                            <span className="text-2xl">{currentEvaluation.scoring.max_marks}</span>
                                            <button onClick={() => setEditingScore(false)} className="ml-2 text-green-600">
                                                <CheckIcon className="w-5 h-5" />
                                            </button>
                                        </div>
                                    ) : (
                                        <div onClick={() => setEditingScore(true)} className="cursor-pointer group">
                                            <div className="text-4xl font-bold text-primary-600">
                                                {manualScore}
                                                <span className="text-2xl text-gray-400">/{currentEvaluation.scoring.max_marks}</span>
                                            </div>
                                            <p className="text-xs text-gray-500 mt-1 group-hover:text-primary-600">Click to edit</p>
                                        </div>
                                    )}
                                </div>

                                {/* Keywords */}
                                <div className="pt-3 border-t">
                                    <p className="text-xs text-gray-500 mb-2">Matched:</p>
                                    <div className="flex flex-wrap gap-1">
                                        {currentEvaluation.scoring.matched_keywords.map((kw) => (
                                            <span key={kw} className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">✓ {kw}</span>
                                        ))}
                                    </div>
                                    {currentEvaluation.scoring.missing_keywords.length > 0 && (
                                        <>
                                            <p className="text-xs text-gray-500 mt-2 mb-1">Missing:</p>
                                            <div className="flex flex-wrap gap-1">
                                                {currentEvaluation.scoring.missing_keywords.map((kw) => (
                                                    <span key={kw} className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded">✗ {kw}</span>
                                                ))}
                                            </div>
                                        </>
                                    )}
                                </div>

                                {/* Teacher Comments */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Comments</label>
                                    <textarea
                                        value={teacherComments}
                                        onChange={(e) => setTeacherComments(e.target.value)}
                                        rows={2}
                                        placeholder="Add feedback..."
                                        className="w-full px-3 py-2 border rounded-lg text-sm"
                                    />
                                </div>

                                {/* Actions */}
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => handleApprove(true)}
                                        className="flex-1 bg-green-600 text-white py-2 rounded-lg flex items-center justify-center gap-2 hover:bg-green-700"
                                    >
                                        <CheckCircleIcon className="w-4 h-4" />
                                        Approve
                                    </button>
                                    <button
                                        onClick={() => handleApprove(false)}
                                        className="flex-1 bg-red-600 text-white py-2 rounded-lg flex items-center justify-center gap-2 hover:bg-red-700"
                                    >
                                        <XCircleIcon className="w-4 h-4" />
                                        Reject
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Workflow Info */}
            <div className="card bg-gradient-to-r from-primary-50 to-purple-50">
                <div className="flex items-center gap-3 mb-4">
                    <SparklesIcon className="w-6 h-6 text-primary-600" />
                    <h3 className="font-semibold text-gray-900">Evaluation Workflow</h3>
                </div>
                <div className="grid md:grid-cols-4 gap-4">
                    <div className="text-center">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center mx-auto mb-2 font-bold ${questionPaper ? 'bg-green-500 text-white' : 'bg-primary-100 text-primary-600'}`}>1</div>
                        <p className="text-sm text-gray-600">Upload Question Paper</p>
                    </div>
                    <div className="text-center">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center mx-auto mb-2 font-bold ${currentEvaluation ? 'bg-green-500 text-white' : 'bg-primary-100 text-primary-600'}`}>2</div>
                        <p className="text-sm text-gray-600">Scan Student Answers</p>
                    </div>
                    <div className="text-center">
                        <div className="w-10 h-10 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center mx-auto mb-2 font-bold">3</div>
                        <p className="text-sm text-gray-600">AI Suggests Score</p>
                    </div>
                    <div className="text-center">
                        <div className="w-10 h-10 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center mx-auto mb-2 font-bold">4</div>
                        <p className="text-sm text-gray-600">Teacher Approves</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
