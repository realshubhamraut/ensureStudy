"""
Meeting Event Producer - Publish meeting events to Kafka
Topics: meeting-events, meeting-recordings, meeting-analytics
"""
import json
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import os

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

# Initialize producer (lazy initialization)
_producer = None


def get_producer():
    """Get or create Kafka producer with lazy initialization"""
    global _producer
    if _producer is None:
        try:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            print(f"✅ Kafka producer connected to {KAFKA_BOOTSTRAP_SERVERS}")
        except NoBrokersAvailable:
            print(f"⚠️ Kafka not available at {KAFKA_BOOTSTRAP_SERVERS}, events will not be published")
            return None
    return _producer


def publish_meeting_event(event_type: str, meeting_id: str, data: dict):
    """
    Publish a meeting event to Kafka
    
    Event types:
    - meeting.created
    - meeting.started
    - meeting.ended
    - meeting.participant.joined
    - meeting.participant.left
    - meeting.cancelled
    """
    producer = get_producer()
    if not producer:
        return False
    
    event = {
        "event_type": event_type,
        "meeting_id": meeting_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    
    try:
        future = producer.send(
            topic='meeting-events',
            key=meeting_id,
            value=event
        )
        future.get(timeout=10)  # Wait for confirmation
        print(f"📤 Published: {event_type} for meeting {meeting_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to publish event: {e}")
        return False


def publish_recording_event(meeting_id: str, recording_id: str, data: dict):
    """
    Publish a recording event to trigger processing pipeline
    
    Data should include:
    - storage_url: URL of the uploaded recording
    - duration_seconds: Length of recording
    - format: video format (webm, mp4)
    """
    producer = get_producer()
    if not producer:
        return False
    
    event = {
        "event_type": "recording.uploaded",
        "meeting_id": meeting_id,
        "recording_id": recording_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    
    try:
        future = producer.send(
            topic='meeting-recordings',
            key=meeting_id,
            value=event
        )
        future.get(timeout=10)
        print(f"📤 Published recording event for meeting {meeting_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to publish recording event: {e}")
        return False


def publish_analytics_event(meeting_id: str, classroom_id: str, event_type: str, user_id: str, metadata: dict = None):
    """
    Publish analytics events for real-time tracking
    
    Event types:
    - participant.joined
    - participant.left
    - screen.shared
    - reaction.sent
    - chat.message
    """
    producer = get_producer()
    if not producer:
        return False
    
    event = {
        "meeting_id": meeting_id,
        "classroom_id": classroom_id,
        "event_type": event_type,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }
    
    try:
        future = producer.send(
            topic='meeting-analytics',
            key=f"{classroom_id}:{meeting_id}",
            value=event
        )
        future.get(timeout=10)
        return True
    except Exception as e:
        print(f"❌ Failed to publish analytics event: {e}")
        return False


# Convenience functions for common events
def on_meeting_created(meeting_id: str, classroom_id: str, host_id: str, title: str):
    return publish_meeting_event('meeting.created', meeting_id, {
        'classroom_id': classroom_id,
        'host_id': host_id,
        'title': title
    })


def on_meeting_started(meeting_id: str, classroom_id: str, host_id: str):
    return publish_meeting_event('meeting.started', meeting_id, {
        'classroom_id': classroom_id,
        'host_id': host_id,
        'started_at': datetime.utcnow().isoformat()
    })


def on_meeting_ended(meeting_id: str, classroom_id: str, duration_minutes: int, participant_count: int):
    return publish_meeting_event('meeting.ended', meeting_id, {
        'classroom_id': classroom_id,
        'duration_minutes': duration_minutes,
        'participant_count': participant_count,
        'ended_at': datetime.utcnow().isoformat()
    })


def on_participant_joined(meeting_id: str, classroom_id: str, user_id: str, role: str):
    publish_meeting_event('meeting.participant.joined', meeting_id, {
        'user_id': user_id,
        'role': role
    })
    return publish_analytics_event(meeting_id, classroom_id, 'participant.joined', user_id)


def on_participant_left(meeting_id: str, classroom_id: str, user_id: str, duration_seconds: int):
    publish_meeting_event('meeting.participant.left', meeting_id, {
        'user_id': user_id,
        'duration_seconds': duration_seconds
    })
    return publish_analytics_event(meeting_id, classroom_id, 'participant.left', user_id, {
        'duration_seconds': duration_seconds
    })


def on_recording_uploaded(meeting_id: str, recording_id: str, storage_url: str, duration_seconds: int):
    return publish_recording_event(meeting_id, recording_id, {
        'storage_url': storage_url,
        'duration_seconds': duration_seconds,
        'format': 'webm'
    })
