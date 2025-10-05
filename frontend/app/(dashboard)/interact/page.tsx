'use client'

import { getApiBaseUrl } from '@/utils/api'

import { useSession } from 'next-auth/react'
import { useState, useEffect, useRef } from 'react'
import {
    ChatBubbleLeftRightIcon,
    MagnifyingGlassIcon,
    PaperAirplaneIcon,
    UserCircleIcon,
    UserGroupIcon,
    AcademicCapIcon,
    PlusIcon,
    ArrowPathIcon,
    XMarkIcon,
    CheckIcon
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface User {
    id: string
    username: string
    first_name: string
    last_name: string
    avatar_url?: string
    role: string
}

interface Message {
    id: string
    conversation_id: string
    sender_id: string
    content: string
    message_type: string
    is_deleted: boolean
    is_edited: boolean
    is_read: boolean
    created_at: string
    sender?: User
}

interface Conversation {
    id: string
    type: string
    title?: string
    participants: {
        id: string
        user_id: string
        role: string
        user: User
    }[]
    last_message?: Message
    unread_count?: number
    last_message_at?: string
}

export default function InteractPage() {
    const { data: session } = useSession()
    const [conversations, setConversations] = useState<Conversation[]>([])
    const [contacts, setContacts] = useState<User[]>([])
    const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null)
    const [messages, setMessages] = useState<Message[]>([])
    const [newMessage, setNewMessage] = useState('')
    const [searchQuery, setSearchQuery] = useState('')
    const [showNewChat, setShowNewChat] = useState(false)
    const [loading, setLoading] = useState(true)
    const [sendingMessage, setSendingMessage] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || getApiBaseUrl()

    // Fetch conversations
    useEffect(() => {
        if (!session?.accessToken) return

        async function fetchConversations() {
            try {
                const res = await fetch(`${apiUrl}/api/interact/conversations`, {
                    headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                })
                if (res.ok) {
                    const data = await res.json()
                    setConversations(data)
                }
            } catch (err) {
                console.error('Error fetching conversations:', err)
            } finally {
                setLoading(false)
            }
        }

        fetchConversations()
        // Poll for new messages every 5 seconds
        const interval = setInterval(fetchConversations, 5000)
        return () => clearInterval(interval)
    }, [session?.accessToken, apiUrl])

    // Fetch contacts for new chat
    useEffect(() => {
        if (!session?.accessToken || !showNewChat) return

        async function fetchContacts() {
            try {
                const res = await fetch(`${apiUrl}/api/interact/contacts?search=${searchQuery}`, {
                    headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                })
                if (res.ok) {
                    const data = await res.json()
                    setContacts(data)
                }
            } catch (err) {
                console.error('Error fetching contacts:', err)
            }
        }

        fetchContacts()
    }, [session?.accessToken, showNewChat, searchQuery, apiUrl])

    // Fetch messages when conversation is selected
    useEffect(() => {
        if (!session?.accessToken || !selectedConversation) return

        async function fetchMessages() {
            try {
                const res = await fetch(`${apiUrl}/api/interact/conversations/${selectedConversation?.id}/messages`, {
                    headers: { 'Authorization': `Bearer ${session?.accessToken}` }
                })
                if (res.ok) {
                    const data = await res.json()
                    setMessages(data)
                }
            } catch (err) {
                console.error('Error fetching messages:', err)
            }
        }

        fetchMessages()
        const interval = setInterval(fetchMessages, 3000)
        return () => clearInterval(interval)
    }, [session?.accessToken, selectedConversation?.id, apiUrl])

    // Scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Send message
    async function handleSendMessage(e: React.FormEvent) {
        e.preventDefault()
        if (!newMessage.trim() || !selectedConversation || sendingMessage) return

        setSendingMessage(true)
        try {
            const res = await fetch(`${apiUrl}/api/interact/messages`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    conversation_id: selectedConversation.id,
                    content: newMessage.trim()
                })
            })

            if (res.ok) {
                const msg = await res.json()
                setMessages(prev => [...prev, msg])
                setNewMessage('')
            }
        } catch (err) {
            console.error('Error sending message:', err)
        } finally {
            setSendingMessage(false)
        }
    }

    // Start new conversation
    async function startConversation(contactId: string) {
        try {
            const res = await fetch(`${apiUrl}/api/interact/conversations`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.accessToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: 'direct',
                    participant_ids: [contactId]
                })
            })

            if (res.ok) {
                const conv = await res.json()
                setConversations(prev => [conv, ...prev.filter(c => c.id !== conv.id)])
                setSelectedConversation(conv)
                setShowNewChat(false)
            }
        } catch (err) {
            console.error('Error starting conversation:', err)
        }
    }

    // Get conversation display name
    function getConversationName(conv: Conversation) {
        if (conv.title) return conv.title
        const otherParticipants = conv.participants.filter(p => p.user_id !== session?.user?.id)
        if (otherParticipants.length === 0) return 'Unknown'
        const user = otherParticipants[0].user
        return user?.first_name && user?.last_name
            ? `${user.first_name} ${user.last_name}`
            : user?.username || 'Unknown'
    }

    // Get role icon
    function getRoleIcon(role: string) {
        switch (role) {
            case 'teacher':
                return <AcademicCapIcon className="w-4 h-4 text-purple-500" />
            case 'parent':
                return <UserGroupIcon className="w-4 h-4 text-orange-500" />
            default:
                return <UserCircleIcon className="w-4 h-4 text-blue-500" />
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[calc(100vh-120px)]">
                <ArrowPathIcon className="w-8 h-8 animate-spin text-primary-600" />
            </div>
        )
    }

    return (
        <div className="flex h-[calc(100vh-64px)] -m-6 bg-gray-50">
            {/* Sidebar - Conversations List */}
            <div className="w-80 border-r border-gray-200 bg-white flex flex-col">
                {/* Header */}
                <div className="p-4 border-b border-gray-100">
                    <div className="flex items-center justify-between mb-3">
                        <h2 className="text-lg font-bold text-gray-900">Messages</h2>
                        <button
                            onClick={() => setShowNewChat(true)}
                            className="p-2 rounded-lg bg-primary-50 text-primary-600 hover:bg-primary-100"
                        >
                            <PlusIcon className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="relative">
                        <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search conversations..."
                            className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                    </div>
                </div>

                {/* Conversations List */}
                <div className="flex-1 overflow-y-auto">
                    {conversations.length === 0 ? (
                        <div className="p-4 text-center text-gray-500">
                            <ChatBubbleLeftRightIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                            <p>No conversations yet</p>
                            <button
                                onClick={() => setShowNewChat(true)}
                                className="mt-2 text-primary-600 hover:underline"
                            >
                                Start a new chat
                            </button>
                        </div>
                    ) : (
                        conversations.map(conv => (
                            <button
                                key={conv.id}
                                onClick={() => setSelectedConversation(conv)}
                                className={clsx(
                                    'w-full p-4 flex items-start gap-3 hover:bg-gray-50 text-left border-b border-gray-100',
                                    selectedConversation?.id === conv.id && 'bg-primary-50'
                                )}
                            >
                                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white font-semibold">
                                    {getConversationName(conv).charAt(0).toUpperCase()}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <span className="font-medium text-gray-900 truncate">
                                            {getConversationName(conv)}
                                        </span>
                                        {conv.unread_count && conv.unread_count > 0 && (
                                            <span className="bg-primary-600 text-white text-xs px-2 py-0.5 rounded-full">
                                                {conv.unread_count}
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-sm text-gray-500 truncate">
                                        {conv.last_message?.content || 'No messages yet'}
                                    </p>
                                </div>
                            </button>
                        ))
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-white">
                {selectedConversation ? (
                    <>
                        {/* Chat Header */}
                        <div className="p-4 border-b border-gray-200 flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white font-semibold">
                                {getConversationName(selectedConversation).charAt(0).toUpperCase()}
                            </div>
                            <div className="flex-1">
                                <h3 className="font-semibold text-gray-900">
                                    {getConversationName(selectedConversation)}
                                </h3>
                                <div className="flex items-center gap-1 text-sm text-gray-500">
                                    {getRoleIcon(selectedConversation.participants.find(p => p.user_id !== session?.user?.id)?.user?.role || 'student')}
                                    <span className="capitalize">
                                        {selectedConversation.participants.find(p => p.user_id !== session?.user?.id)?.user?.role || 'Student'}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
                            {messages.map(msg => (
                                <div
                                    key={msg.id}
                                    className={clsx(
                                        'flex',
                                        msg.sender_id === session?.user?.id ? 'justify-end' : 'justify-start'
                                    )}
                                >
                                    <div className={clsx(
                                        'max-w-[70%] rounded-2xl px-4 py-2',
                                        msg.sender_id === session?.user?.id
                                            ? 'bg-primary-600 text-white rounded-br-sm'
                                            : 'bg-white text-gray-900 rounded-bl-sm shadow-sm border border-gray-100'
                                    )}>
                                        {msg.sender_id !== session?.user?.id && (
                                            <p className="text-xs font-medium mb-1 text-primary-600">
                                                {msg.sender?.first_name || msg.sender?.username}
                                            </p>
                                        )}
                                        <p>{msg.content}</p>
                                        <div className={clsx(
                                            'text-xs mt-1 flex items-center gap-1 justify-end',
                                            msg.sender_id === session?.user?.id ? 'text-white/70' : 'text-gray-500'
                                        )}>
                                            {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            {msg.is_edited && ' (edited)'}
                                            {/* Tick indicators for sent messages */}
                                            {msg.sender_id === session?.user?.id && (
                                                <span className="ml-1 flex">
                                                    {msg.is_read ? (
                                                        // Double green ticks for read
                                                        <span className="flex text-green-400">
                                                            <CheckIcon className="w-3.5 h-3.5 -mr-1.5" />
                                                            <CheckIcon className="w-3.5 h-3.5" />
                                                        </span>
                                                    ) : (
                                                        // Single tick for sent
                                                        <CheckIcon className="w-3.5 h-3.5" />
                                                    )}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Message Input */}
                        <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-200 bg-white">
                            <div className="flex items-center gap-3">
                                <input
                                    type="text"
                                    value={newMessage}
                                    onChange={(e) => setNewMessage(e.target.value)}
                                    placeholder="Type a message..."
                                    className="flex-1 px-4 py-3 rounded-full border border-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
                                />
                                <button
                                    type="submit"
                                    disabled={!newMessage.trim() || sendingMessage}
                                    className="p-3 rounded-full bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {sendingMessage ? (
                                        <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                    ) : (
                                        <PaperAirplaneIcon className="w-5 h-5" />
                                    )}
                                </button>
                            </div>
                        </form>
                    </>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-gray-500 bg-gray-50">
                        <div className="text-center">
                            <ChatBubbleLeftRightIcon className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                            <p className="text-lg">Select a conversation to start chatting</p>
                        </div>
                    </div>
                )}
            </div>

            {/* New Chat Modal */}
            {showNewChat && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4">
                        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
                            <h3 className="text-lg font-bold">New Conversation</h3>
                            <button onClick={() => setShowNewChat(false)}>
                                <XMarkIcon className="w-6 h-6 text-gray-500" />
                            </button>
                        </div>
                        <div className="p-4">
                            <input
                                type="text"
                                placeholder="Search contacts..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full px-4 py-2 rounded-lg border border-gray-200 mb-4 focus:outline-none focus:ring-2 focus:ring-primary-500"
                            />
                            <div className="max-h-64 overflow-y-auto space-y-2">
                                {contacts.map(contact => (
                                    <button
                                        key={contact.id}
                                        onClick={() => startConversation(contact.id)}
                                        className="w-full p-3 flex items-center gap-3 rounded-lg hover:bg-gray-50"
                                    >
                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-400 to-gray-600 flex items-center justify-center text-white font-semibold">
                                            {(contact.first_name || contact.username).charAt(0).toUpperCase()}
                                        </div>
                                        <div className="flex-1 text-left">
                                            <p className="font-medium text-gray-900">
                                                {contact.first_name && contact.last_name
                                                    ? `${contact.first_name} ${contact.last_name}`
                                                    : contact.username}
                                            </p>
                                            <div className="flex items-center gap-1 text-sm text-gray-500">
                                                {getRoleIcon(contact.role)}
                                                <span className="capitalize">{contact.role}</span>
                                            </div>
                                        </div>
                                    </button>
                                ))}
                                {contacts.length === 0 && (
                                    <p className="text-center text-gray-500 py-4">No contacts found</p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
