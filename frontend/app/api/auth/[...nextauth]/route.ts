import NextAuth, { NextAuthOptions } from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

declare module 'next-auth' {
    interface Session {
        user: {
            id: string
            name?: string
            email: string
            username: string
            role: 'student' | 'teacher' | 'parent' | 'admin'
        }
        accessToken: string
        refreshToken: string
    }

    interface User {
        id: string
        email: string
        username: string
        role: string
        accessToken: string
        refreshToken: string
    }
}

declare module 'next-auth/jwt' {
    interface JWT {
        userId: string
        role: string
        accessToken: string
        refreshToken: string
    }
}

const authOptions: NextAuthOptions = {
    providers: [
        CredentialsProvider({
            name: 'Credentials',
            credentials: {
                email: { label: 'Email', type: 'email' },
                password: { label: 'Password', type: 'password' },
            },
            async authorize(credentials) {
                if (!credentials?.email || !credentials?.password) {
                    throw new Error('Email and password required')
                }

                try {
                    const res = await fetch(
                        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/auth/login`,
                        {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                email: credentials.email,
                                password: credentials.password,
                            }),
                        }
                    )

                    if (!res.ok) {
                        const error = await res.json()
                        throw new Error(error.error || 'Invalid credentials')
                    }

                    const data = await res.json()

                    return {
                        id: data.user.id,
                        email: data.user.email,
                        username: data.user.username,
                        role: data.user.role,
                        accessToken: data.access_token,
                        refreshToken: data.refresh_token,
                    }
                } catch (error: any) {
                    throw new Error(error.message || 'Authentication failed')
                }
            },
        }),
    ],

    callbacks: {
        async jwt({ token, user }) {
            if (user) {
                token.userId = user.id
                token.role = user.role
                token.accessToken = user.accessToken
                token.refreshToken = user.refreshToken
            }
            return token
        },

        async session({ session, token }) {
            if (session.user) {
                session.user.id = token.userId
                session.user.role = token.role as any
            }
            session.accessToken = token.accessToken
            session.refreshToken = token.refreshToken
            return session
        },
    },

    pages: {
        signIn: '/auth/signin',
        error: '/auth/error',
    },

    session: {
        strategy: 'jwt',
        maxAge: 24 * 60 * 60, // 24 hours
    },

    secret: process.env.NEXTAUTH_SECRET,
}

const handler = NextAuth(authOptions)

export { handler as GET, handler as POST }
