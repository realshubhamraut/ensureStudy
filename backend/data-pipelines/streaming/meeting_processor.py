"""
PySpark Meeting Processor - Big Data Pipeline for Meeting Recordings

This job consumes recording events from Kafka and:
1. Downloads the recording from storage
2. Extracts audio and runs speech-to-text (Whisper)
3. Generates summary using LLM
4. Creates embeddings for vector search
5. Stores results in MongoDB and Qdrant
6. Publishes analytics to Cassandra

Run: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 meeting_processor.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
import json
import os

# Initialize Spark with Kafka and MongoDB connectors
spark = SparkSession.builder \
    .appName("MeetingRecordingProcessor") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
    .config("spark.mongodb.write.connection.uri", 
            os.getenv("MONGODB_URI", "mongodb://ensure_study:mongodb_password_123@localhost:27017/ensure_study_meetings")) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "meeting-recordings"

# Define schema for recording events
recording_schema = StructType([
    StructField("event_type", StringType(), True),
    StructField("meeting_id", StringType(), True),
    StructField("recording_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("data", StructType([
        StructField("storage_url", StringType(), True),
        StructField("duration_seconds", IntegerType(), True),
        StructField("format", StringType(), True)
    ]), True)
])


def process_recording_batch(df, epoch_id):
    """Process a batch of recording events"""
    if df.count() == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"Processing batch {epoch_id} with {df.count()} recordings")
    print(f"{'='*60}")
    
    # Collect rows for processing
    rows = df.collect()
    
    for row in rows:
        meeting_id = row.meeting_id
        recording_id = row.recording_id
        storage_url = row.data.storage_url if row.data else None
        duration = row.data.duration_seconds if row.data else 0
        
        print(f"\n📼 Processing: Meeting {meeting_id[:8]}...")
        print(f"   Recording: {recording_id[:8]}...")
        print(f"   Duration: {duration // 60}m {duration % 60}s")
        print(f"   URL: {storage_url}")
        
        # Step 1: Download recording (placeholder)
        print("   [1/5] Downloading recording...")
        # In production: download from S3/GCS
        
        # Step 2: Extract audio and transcribe
        print("   [2/5] Running speech-to-text...")
        # In production: Use Whisper API or local model
        transcript = generate_mock_transcript(meeting_id, duration)
        
        # Step 3: Generate summary
        print("   [3/5] Generating summary...")
        # In production: Call LLM API
        summary = generate_mock_summary(transcript)
        
        # Step 4: Create embeddings
        print("   [4/5] Creating embeddings...")
        # In production: Use embedding model
        
        # Step 5: Store in MongoDB
        print("   [5/5] Storing in MongoDB...")
        store_transcript(meeting_id, recording_id, transcript, summary)
        
        print(f"   ✅ Processing complete for {meeting_id[:8]}...")


def generate_mock_transcript(meeting_id: str, duration_seconds: int) -> str:
    """Generate a mock transcript for testing"""
    return f"""[Meeting Transcript]
Meeting ID: {meeting_id}
Duration: {duration_seconds // 60} minutes

00:00 - Teacher: Good morning everyone, let's begin today's session.
00:15 - Teacher: Today we'll be covering the laws of motion.
01:30 - Teacher: Newton's first law states that an object at rest stays at rest.
03:00 - Student: Could you explain inertia more?
03:30 - Teacher: Of course, inertia is the resistance to change in motion.
05:00 - Teacher: Let's move on to Newton's second law, F=ma.
08:00 - Teacher: For homework, please solve problems 1-10 in chapter 5.
{duration_seconds // 60}:00 - Teacher: That concludes today's session. Any questions?
"""


def generate_mock_summary(transcript: str) -> dict:
    """Generate a mock summary for testing"""
    return {
        "brief": "Lecture on Newton's Laws of Motion covering inertia and F=ma.",
        "detailed": "The session covered Newton's laws of motion. The teacher explained the first law about inertia and objects at rest. Students asked questions about inertia which were addressed. The second law (F=ma) was also covered. Homework was assigned from chapter 5.",
        "key_points": [
            "Newton's first law - objects at rest stay at rest",
            "Inertia is resistance to change in motion",
            "Newton's second law - Force equals mass times acceleration",
            "Homework: Chapter 5, problems 1-10"
        ],
        "topics_discussed": [
            "Newton's Laws of Motion",
            "Inertia",
            "Force and Acceleration"
        ],
        "action_items": [
            "Complete homework problems 1-10 from chapter 5"
        ]
    }


def store_transcript(meeting_id: str, recording_id: str, transcript: str, summary: dict):
    """Store transcript in MongoDB"""
    # Create a DataFrame with the transcript data
    transcript_data = [{
        "meeting_id": meeting_id,
        "recording_id": recording_id,
        "full_transcript": transcript,
        "summary": json.dumps(summary),
        "status": "processed"
    }]
    
    transcript_df = spark.createDataFrame(transcript_data)
    
    # Write to MongoDB
    try:
        transcript_df.write \
            .format("mongodb") \
            .mode("append") \
            .option("database", "ensure_study_meetings") \
            .option("collection", "meeting_transcripts") \
            .save()
        print("   ✅ Transcript stored in MongoDB")
    except Exception as e:
        print(f"   ⚠️ MongoDB write failed: {e}")
        # Fallback: log the data
        print(f"   📝 Transcript logged: {len(transcript)} chars")


def main():
    print("\n" + "="*60)
    print("🚀 Starting Meeting Recording Processor")
    print("="*60)
    print(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {KAFKA_TOPIC}")
    print("="*60 + "\n")
    
    # Read from Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    # Parse JSON from Kafka value
    parsed_df = kafka_df \
        .select(from_json(col("value").cast("string"), recording_schema).alias("data")) \
        .select("data.*")
    
    # Process each batch
    query = parsed_df.writeStream \
        .foreachBatch(process_recording_batch) \
        .option("checkpointLocation", "/tmp/meeting_processor_checkpoint") \
        .trigger(processingTime="30 seconds") \
        .start()
    
    print("✅ Streaming query started. Waiting for recordings...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
