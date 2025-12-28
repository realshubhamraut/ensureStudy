"""
Kafka Configuration Utilities
"""
import os
from typing import Optional
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import json


def get_kafka_config() -> dict:
    """Get Kafka configuration from environment"""
    return {
        "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(","),
        "client_id": os.getenv("KAFKA_CLIENT_ID", "ensurestudy-client"),
        "group_id": os.getenv("KAFKA_GROUP_ID", "ensurestudy-consumers"),
    }


def create_producer() -> KafkaProducer:
    """Create a Kafka producer"""
    config = get_kafka_config()
    
    return KafkaProducer(
        bootstrap_servers=config["bootstrap_servers"],
        client_id=config["client_id"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
        max_in_flight_requests_per_connection=1
    )


def create_consumer(
    topics: list,
    group_id: Optional[str] = None,
    auto_offset_reset: str = "earliest"
) -> KafkaConsumer:
    """Create a Kafka consumer"""
    config = get_kafka_config()
    
    return KafkaConsumer(
        *topics,
        bootstrap_servers=config["bootstrap_servers"],
        group_id=group_id or config["group_id"],
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None
    )


def create_topics(topics_config: list) -> None:
    """Create Kafka topics from configuration"""
    config = get_kafka_config()
    
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=config["bootstrap_servers"],
            client_id=config["client_id"]
        )
        
        new_topics = []
        for topic in topics_config:
            new_topics.append(NewTopic(
                name=topic["name"],
                num_partitions=topic.get("partitions", 3),
                replication_factor=topic.get("replication_factor", 1)
            ))
        
        admin_client.create_topics(new_topics=new_topics, validate_only=False)
        print(f"✓ Created {len(new_topics)} Kafka topics")
        
    except Exception as e:
        print(f"Error creating topics: {e}")


# Topic names as constants
class Topics:
    STUDENT_EVENTS = "student-events"
    CHAT_MESSAGES = "chat-messages"
    ASSESSMENT_SUBMISSIONS = "assessment-submissions"
    MODERATION_EVENTS = "moderation-events"
    LEADERBOARD_UPDATES = "leaderboard-updates"
    PROGRESS_UPDATES = "progress-updates"
    ANALYTICS_EVENTS = "analytics-events"
    # INTEGRATION NOTE: Added for document processing pipeline
    DOCUMENT_PROCESSING = "document-processing"

