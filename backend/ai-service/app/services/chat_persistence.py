"""
Chat Persistence Service

Saves chat messages and sources to the core-service database.
Used by AI service to persist conversations after generating responses.
"""
import os
import logging
import httpx
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Core service URL for internal communication
CORE_SERVICE_URL = os.getenv("CORE_SERVICE_URL", "http://localhost:9000")


class ChatPersistenceError(Exception):
    """Error saving chat to database"""
    pass


async def save_user_message(
    conversation_id: str,
    user_id: str,
    content: str,
    auth_token: str,
    subject: Optional[str] = None,
    classroom_id: Optional[str] = None,
    image_url: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Save a user message to the database.
    
    Args:
        conversation_id: Conversation ID (frontend-generated is OK)
        user_id: User ID
        content: Message content
        auth_token: JWT token for authentication
        subject: Optional subject tag
        classroom_id: Optional classroom ID
        image_url: Optional image URL if user sent an image
        
    Returns:
        Message dict from API or None on error
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{CORE_SERVICE_URL}/api/chat/conversations/{conversation_id}/messages",
                json={
                    "type": "user",
                    "content": content,
                    "subject": subject,
                    "classroom_id": classroom_id,
                    "image_url": image_url
                },
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"[CHAT-PERSIST] Saved user message to conversation {conversation_id}")
                return result.get("message")
            else:
                logger.warning(f"[CHAT-PERSIST] Failed to save user message: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"[CHAT-PERSIST] Error saving user message: {e}")
        return None


async def save_assistant_message(
    conversation_id: str,
    user_id: str,
    content: str,
    auth_token: str,
    response_data: Optional[Dict[str, Any]] = None,
    subject: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Save an assistant (AI) message to the database.
    
    Args:
        conversation_id: Conversation ID
        user_id: User ID
        content: AI response text
        auth_token: JWT token for authentication
        response_data: Full response JSON for detailed view
        subject: Optional subject tag
        
    Returns:
        Message dict from API or None on error
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{CORE_SERVICE_URL}/api/chat/conversations/{conversation_id}/messages",
                json={
                    "type": "assistant",
                    "content": content,
                    "response": response_data,
                    "subject": subject
                },
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"[CHAT-PERSIST] Saved assistant message to conversation {conversation_id}")
                return result.get("message")
            else:
                logger.warning(f"[CHAT-PERSIST] Failed to save assistant message: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"[CHAT-PERSIST] Error saving assistant message: {e}")
        return None


async def save_sources(
    conversation_id: str,
    sources: List[Dict[str, Any]],
    auth_token: str
) -> bool:
    """
    Save sources/resources for a conversation.
    
    Args:
        conversation_id: Conversation ID
        sources: List of source dicts with type, title, url, etc.
        auth_token: JWT token for authentication
        
    Returns:
        True if saved successfully, False otherwise
    """
    if not sources:
        return True  # Nothing to save
        
    try:
        # Transform sources to expected format
        formatted_sources = []
        for src in sources:
            formatted_sources.append({
                "type": src.get("type", "article"),
                "title": src.get("title", "Untitled"),
                "url": src.get("url") or src.get("source"),
                "thumbnailUrl": src.get("thumbnailUrl") or src.get("thumbnail"),
                "relevance": int(src.get("relevance", 0) * 100) if isinstance(src.get("relevance"), float) else src.get("relevance"),
                "snippet": src.get("snippet") or src.get("preview"),
                "source": src.get("source_name") or src.get("source"),
                "embedUrl": src.get("embedUrl"),
                "duration": src.get("duration"),
            })
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{CORE_SERVICE_URL}/api/chat/conversations/{conversation_id}/sources",
                json={"sources": formatted_sources},
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"[CHAT-PERSIST] Saved {len(sources)} sources to conversation {conversation_id}")
                return True
            else:
                logger.warning(f"[CHAT-PERSIST] Failed to save sources: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"[CHAT-PERSIST] Error saving sources: {e}")
        return False


async def save_chat_exchange(
    conversation_id: str,
    user_id: str,
    user_message: str,
    ai_response: str,
    auth_token: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    response_data: Optional[Dict[str, Any]] = None,
    subject: Optional[str] = None,
    classroom_id: Optional[str] = None
) -> bool:
    """
    Save a complete chat exchange (user message + AI response + sources).
    
    This is the main function to call after generating an AI response.
    
    Args:
        conversation_id: Conversation ID
        user_id: User ID
        user_message: User's question
        ai_response: AI's answer text
        auth_token: JWT token for authentication
        sources: Optional list of sources
        response_data: Full response data for detailed view
        subject: Optional subject tag
        classroom_id: Optional classroom ID
        
    Returns:
        True if all saves succeeded, False otherwise
    """
    success = True
    
    # Save user message
    user_result = await save_user_message(
        conversation_id=conversation_id,
        user_id=user_id,
        content=user_message,
        auth_token=auth_token,
        subject=subject,
        classroom_id=classroom_id
    )
    if not user_result:
        success = False
        logger.warning("[CHAT-PERSIST] Failed to save user message")
    
    # Save assistant message
    assistant_result = await save_assistant_message(
        conversation_id=conversation_id,
        user_id=user_id,
        content=ai_response,
        auth_token=auth_token,
        response_data=response_data,
        subject=subject
    )
    if not assistant_result:
        success = False
        logger.warning("[CHAT-PERSIST] Failed to save assistant message")
    
    # Save sources if provided
    if sources:
        sources_result = await save_sources(
            conversation_id=conversation_id,
            sources=sources,
            auth_token=auth_token
        )
        if not sources_result:
            success = False
            logger.warning("[CHAT-PERSIST] Failed to save sources")
    
    return success


# Singleton instance
_persistence_service = None


def get_chat_persistence():
    """Get singleton chat persistence service"""
    global _persistence_service
    if _persistence_service is None:
        _persistence_service = ChatPersistenceService()
    return _persistence_service


class ChatPersistenceService:
    """
    Chat persistence service class for dependency injection.
    Wraps the async functions for easier use.
    """
    
    def __init__(self):
        self.core_url = CORE_SERVICE_URL
        
    async def save_exchange(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        ai_response: str,
        auth_token: str,
        **kwargs
    ) -> bool:
        """Save a complete chat exchange"""
        return await save_chat_exchange(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=user_message,
            ai_response=ai_response,
            auth_token=auth_token,
            **kwargs
        )
