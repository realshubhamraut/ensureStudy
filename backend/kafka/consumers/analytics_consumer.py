"""
Analytics Consumer - Aggregate events for dashboards
"""
import os
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict
from kafka import KafkaConsumer

from backend.kafka.config.kafka_config import create_consumer, Topics


class AnalyticsConsumer:
    """
    Consumes events and aggregates metrics for analytics dashboards.
    
    Aggregates:
    - Active users per hour
    - Questions asked per topic
    - Assessment performance trends
    - Study session durations
    """
    
    def __init__(self, group_id: str = "analytics-consumers"):
        self.topics = [
            Topics.STUDENT_EVENTS,
            Topics.ASSESSMENT_SUBMISSIONS,
            Topics.PROGRESS_UPDATES
        ]
        self.consumer = create_consumer(
            topics=self.topics,
            group_id=group_id
        )
        self.running = False
        
        # In-memory aggregations (would use Redis in production)
        self.hourly_active_users = defaultdict(set)
        self.topic_questions = defaultdict(int)
        self.assessment_scores = defaultdict(list)
        self.session_durations = []
    
    def start(self) -> None:
        """Start consuming and aggregating"""
        self.running = True
        print(f"Starting analytics consumer for topics: {self.topics}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    event = message.value
                    self._process_event(event)
                except Exception as e:
                    print(f"Error processing event: {e}")
        
        finally:
            self.consumer.close()
    
    def _process_event(self, event: Dict[str, Any]) -> None:
        """Process and aggregate event"""
        event_type = event.get("event_type", "")
        user_id = event.get("user_id", "")
        timestamp = event.get("timestamp", "")
        
        # Track active user
        if user_id:
            hour_key = timestamp[:13] if timestamp else datetime.utcnow().strftime("%Y-%m-%dT%H")
            self.hourly_active_users[hour_key].add(user_id)
        
        # Aggregate by event type
        if event_type == "question_asked":
            data = event.get("data", {})
            subject = data.get("subject", "unknown")
            self.topic_questions[subject] += 1
        
        elif event_type == "submission":
            subject = event.get("subject", "unknown")
            score = event.get("score", 0)
            self.assessment_scores[subject].append(score)
        
        elif event_type == "study_session_end":
            data = event.get("data", {})
            duration = data.get("duration_seconds", 0)
            self.session_durations.append(duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics"""
        current_hour = datetime.utcnow().strftime("%Y-%m-%dT%H")
        
        return {
            "active_users_current_hour": len(self.hourly_active_users.get(current_hour, set())),
            "questions_by_subject": dict(self.topic_questions),
            "avg_session_duration_seconds": sum(self.session_durations) / len(self.session_durations) if self.session_durations else 0,
            "avg_scores_by_subject": {
                subject: sum(scores) / len(scores) if scores else 0
                for subject, scores in self.assessment_scores.items()
            }
        }
    
    def stop(self) -> None:
        """Stop the consumer"""
        self.running = False


if __name__ == "__main__":
    consumer = AnalyticsConsumer()
    consumer.start()
