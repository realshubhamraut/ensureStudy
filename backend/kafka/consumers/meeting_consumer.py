"""
Meeting Event Consumer - Process meeting events from Kafka
Handles: recording processing triggers, analytics aggregation
"""
import json
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import os
import threading

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
CONSUMER_GROUP = 'meeting-processor-group'


def create_consumer(topics: list):
    """Create a Kafka consumer for specified topics"""
    try:
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        print(f"✅ Consumer connected, subscribed to: {topics}")
        return consumer
    except NoBrokersAvailable:
        print(f"⚠️ Kafka not available at {KAFKA_BOOTSTRAP_SERVERS}")
        return None


def process_meeting_event(event: dict):
    """Process a meeting event"""
    event_type = event.get('event_type')
    meeting_id = event.get('meeting_id')
    data = event.get('data', {})
    
    print(f"📥 Processing: {event_type} for meeting {meeting_id}")
    
    if event_type == 'meeting.created':
        # Could trigger notifications, calendar sync, etc.
        pass
    
    elif event_type == 'meeting.started':
        # Could update analytics, notify absent students
        pass
    
    elif event_type == 'meeting.ended':
        # Trigger post-meeting processing
        duration = data.get('duration_minutes', 0)
        participants = data.get('participant_count', 0)
        print(f"  Meeting ended: {duration} mins, {participants} participants")
    
    elif event_type == 'meeting.participant.joined':
        # Update live participant count
        pass
    
    elif event_type == 'meeting.participant.left':
        # Update attendance records
        pass


def process_recording_event(event: dict):
    """Process a recording upload event - trigger transcription pipeline"""
    meeting_id = event.get('meeting_id')
    recording_id = event.get('recording_id')
    data = event.get('data', {})
    
    storage_url = data.get('storage_url')
    duration = data.get('duration_seconds', 0)
    
    print(f"📼 Processing recording for meeting {meeting_id}")
    print(f"  URL: {storage_url}")
    print(f"  Duration: {duration // 60}m {duration % 60}s")
    
    # TODO: Trigger Whisper transcription
    # TODO: Trigger AI summarization
    # TODO: Store in MongoDB
    # TODO: Generate embeddings for Qdrant
    
    # For now, just log
    print(f"  [TODO] Would trigger transcription pipeline")


def process_analytics_event(event: dict):
    """Process analytics events for real-time metrics"""
    meeting_id = event.get('meeting_id')
    classroom_id = event.get('classroom_id')
    event_type = event.get('event_type')
    user_id = event.get('user_id')
    
    # TODO: Write to Cassandra for time-series analytics
    # TODO: Update real-time dashboard metrics
    
    print(f"📊 Analytics: {event_type} - user {user_id[:8]}... in meeting {meeting_id[:8]}...")


def run_meeting_events_consumer():
    """Run the meeting events consumer loop"""
    consumer = create_consumer(['meeting-events'])
    if not consumer:
        return
    
    print("🎧 Listening for meeting events...")
    try:
        for message in consumer:
            process_meeting_event(message.value)
    except KeyboardInterrupt:
        print("Consumer stopped")
    finally:
        consumer.close()


def run_recording_events_consumer():
    """Run the recording events consumer loop"""
    consumer = create_consumer(['meeting-recordings'])
    if not consumer:
        return
    
    print("🎧 Listening for recording events...")
    try:
        for message in consumer:
            process_recording_event(message.value)
    except KeyboardInterrupt:
        print("Consumer stopped")
    finally:
        consumer.close()


def run_analytics_events_consumer():
    """Run the analytics events consumer loop"""
    consumer = create_consumer(['meeting-analytics'])
    if not consumer:
        return
    
    print("🎧 Listening for analytics events...")
    try:
        for message in consumer:
            process_analytics_event(message.value)
    except KeyboardInterrupt:
        print("Consumer stopped")
    finally:
        consumer.close()


def run_all_consumers():
    """Run all consumers in separate threads"""
    threads = [
        threading.Thread(target=run_meeting_events_consumer, daemon=True),
        threading.Thread(target=run_recording_events_consumer, daemon=True),
        threading.Thread(target=run_analytics_events_consumer, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    print("🚀 All meeting consumers started")
    
    # Keep main thread alive
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nShutting down consumers...")


if __name__ == '__main__':
    run_all_consumers()
