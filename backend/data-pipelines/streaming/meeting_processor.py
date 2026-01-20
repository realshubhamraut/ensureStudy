"""
PySpark Meeting Processor - Big Data Pipeline for Meeting Recordings

This job consumes recording events from Kafka and:
1. Calls AI service for transcription (Whisper)
2. Calls AI service for summary (Gemini)
3. Creates embeddings and stores in Qdrant
4. Stores results in MongoDB
5. Stores analytics in Cassandra

Run: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 meeting_processor.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import json
import os
import requests
from datetime import datetime

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

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "meeting-recordings"
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Define schema for recording events
recording_schema = StructType([
    StructField("event_type", StringType(), True),
    StructField("meeting_id", StringType(), True),
    StructField("recording_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("classroom_id", StringType(), True),
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
    
    rows = df.collect()
    
    for row in rows:
        meeting_id = row.meeting_id
        recording_id = row.recording_id
        classroom_id = getattr(row, 'classroom_id', None) or 'unknown'
        storage_url = row.data.storage_url if row.data else None
        duration = row.data.duration_seconds if row.data else 0
        
        print(f"\n📼 Processing: Meeting {meeting_id[:8]}...")
        print(f"   Recording: {recording_id[:8]}...")
        print(f"   Classroom: {classroom_id[:8] if classroom_id else 'N/A'}...")
        print(f"   Duration: {duration // 60}m {duration % 60}s")
        print(f"   URL: {storage_url}")
        
        if not storage_url:
            print("   ❌ No storage URL provided, skipping")
            continue
        
        try:
            # Step 1: Call AI service for transcription
            print("   [1/4] Calling AI service for transcription...")
            transcript_response = requests.post(
                f"{AI_SERVICE_URL}/api/meetings/transcribe",
                json={
                    "meeting_id": meeting_id,
                    "recording_url": storage_url,
                    "duration_seconds": duration
                },
                timeout=300
            )
            
            if transcript_response.status_code != 200:
                print(f"   ❌ Transcription failed: {transcript_response.text}")
                continue
                
            transcript_data = transcript_response.json()
            transcript = transcript_data.get("transcript", "")
            segments = transcript_data.get("segments", [])
            
            # Step 2: Call AI service for summary
            print("   [2/4] Calling AI service for summarization...")
            summary_response = requests.post(
                f"{AI_SERVICE_URL}/api/meetings/summarize",
                json={
                    "meeting_id": meeting_id,
                    "transcript": transcript
                },
                timeout=60
            )
            
            if summary_response.status_code != 200:
                print(f"   ⚠️ Summarization failed: {summary_response.text}")
                summary = {}
            else:
                summary = summary_response.json()
            
            # Step 3: Create embeddings and store in Qdrant
            print("   [3/4] Creating embeddings and storing in Qdrant...")
            store_embeddings_in_qdrant(meeting_id, classroom_id, transcript, segments)
            
            # Step 4: Store analytics in Cassandra
            print("   [4/4] Storing analytics in Cassandra...")
            store_analytics_cassandra(meeting_id, classroom_id, duration, len(transcript.split()))
            
            print(f"   ✅ Processing complete for {meeting_id[:8]}...")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")


def store_embeddings_in_qdrant(meeting_id: str, classroom_id: str, transcript: str, segments: list):
    """Create embeddings and store in Qdrant"""
    if not OPENAI_API_KEY:
        print("      ⚠️ OPENAI_API_KEY not set, skipping embeddings")
        return
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, VectorParams, Distance
        
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Ensure collection exists
        try:
            qdrant.get_collection("meeting_chunks")
        except:
            qdrant.create_collection(
                collection_name="meeting_chunks",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        
        # Chunk transcript
        chunks = chunk_transcript(transcript, segments)
        
        # Get embeddings and store
        points = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk["text"])
            if embedding:
                points.append(PointStruct(
                    id=abs(hash(f"{meeting_id}_{i}")) % (10**10),
                    vector=embedding,
                    payload={
                        "meeting_id": meeting_id,
                        "classroom_id": classroom_id,
                        "text": chunk["text"],
                        "timestamp": chunk.get("timestamp", ""),
                        "speaker": chunk.get("speaker", ""),
                        "chunk_index": i
                    }
                ))
        
        if points:
            qdrant.upsert(collection_name="meeting_chunks", points=points)
            print(f"      ✅ Stored {len(points)} chunks in Qdrant")
        
    except ImportError:
        print("      ⚠️ qdrant-client not installed")
    except Exception as e:
        print(f"      ⚠️ Qdrant storage failed: {e}")


def chunk_transcript(transcript: str, segments: list) -> list:
    """Chunk transcript into smaller pieces for embedding"""
    chunks = []
    
    if segments:
        current_chunk = {"text": "", "speaker": "", "timestamp": ""}
        for seg in segments:
            if len(current_chunk["text"]) > 500:
                chunks.append(current_chunk)
                current_chunk = {"text": "", "speaker": "", "timestamp": ""}
            
            current_chunk["text"] += " " + seg.get("text", "")
            current_chunk["speaker"] = seg.get("speaker", "")
            current_chunk["timestamp"] = f"{seg.get('start', 0)}-{seg.get('end', 0)}"
        
        if current_chunk["text"]:
            chunks.append(current_chunk)
    else:
        # Simple chunking by sentences
        sentences = transcript.split(". ")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 500:
                chunks.append({"text": current_chunk, "timestamp": "", "speaker": ""})
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append({"text": current_chunk, "timestamp": "", "speaker": ""})
    
    return chunks


def get_embedding(text: str) -> list:
    """Get embedding from OpenAI API"""
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            print(f"      Embedding API error: {response.text}")
    except Exception as e:
        print(f"      Embedding error: {e}")
    
    return None


def store_analytics_cassandra(meeting_id: str, classroom_id: str, duration: int, word_count: int):
    """Store analytics in Cassandra"""
    try:
        from cassandra.cluster import Cluster
        
        cluster = Cluster([os.getenv("CASSANDRA_HOST", "localhost")])
        session = cluster.connect()
        
        session.execute("""
            CREATE KEYSPACE IF NOT EXISTS ensure_study
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
        """)
        session.execute("USE ensure_study")
        session.execute("""
            CREATE TABLE IF NOT EXISTS meeting_analytics (
                classroom_id text,
                meeting_id text,
                processed_at timestamp,
                duration_seconds int,
                word_count int,
                PRIMARY KEY ((classroom_id), processed_at, meeting_id)
            )
        """)
        
        session.execute("""
            INSERT INTO meeting_analytics (classroom_id, meeting_id, processed_at, duration_seconds, word_count)
            VALUES (%s, %s, toTimestamp(now()), %s, %s)
        """, (classroom_id, meeting_id, duration, word_count))
        
        print("      ✅ Analytics stored in Cassandra")
        cluster.shutdown()
    except ImportError:
        print("      ⚠️ cassandra-driver not installed")
    except Exception as e:
        print(f"      ⚠️ Cassandra write failed: {e}")


def main():
    print("\n" + "="*60)
    print("🚀 Starting Meeting Recording Processor")
    print("="*60)
    print(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {KAFKA_TOPIC}")
    print(f"AI Service: {AI_SERVICE_URL}")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"OpenAI Key: {'configured' if OPENAI_API_KEY else 'NOT SET'}")
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
