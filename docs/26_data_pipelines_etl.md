# Page 26: Data Pipelines — ETL, Spark & Analytics

---

## 26.1 Overview

ensureStudy uses **Apache PySpark** for batch ETL and real-time streaming, pulling student data from PostgreSQL, processing meeting recordings from Kafka, engineering ML features, and storing analytics in Cassandra.

### Source: `backend/data-pipelines/` (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `etl/extract/extract_student_data.py` | 156 | PySpark extractors for student data |
| `etl/transform/feature_engineering.py` | 140 | Feature engineering for ML models |
| `streaming/kafka_spark_streaming.py` | — | Real-time Kafka consumer |
| `streaming/meeting_processor.py` | 315 | Meeting recording pipeline |

---

## 26.2 Batch ETL Pipeline

### Extract: `StudentDataExtractor`

Reads directly from PostgreSQL using JDBC:

```python
spark = SparkSession.builder \
    .appName("EnsureStudy-ETL") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.5.0") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

### Data Sources Extracted

| Method | Tables Joined | Output Schema |
|--------|--------------|---------------|
| `extract_progress_data()` | `progress` + `users` | user_id, topic, subject, confidence_score, times_studied, is_weak, class_id, school_id |
| `extract_assessment_results()` | `assessment_results` + `assessments` | user_id, assessment_id, score, max_score, time_taken, confidence, topic, difficulty |
| `extract_leaderboard()` | `leaderboard` + `users` | user_id, global_points, class_points, study_streak, level, xp |
| `extract_chat_sessions()` | `chat_sessions` + `users` | user_id, title, message_count, class_id, school_id |

All extractions support **date-range filtering** for incremental processing.

---

## 26.3 Feature Engineering

### Source: `FeatureEngineer` class (140 lines)

| Method | Input | Output Features | Purpose |
|--------|-------|----------------|---------|
| `engineer_student_features()` | progress + assessment DataFrames | avg_confidence, total_study_sessions, weak_topic_count, topics_covered, avg_score, engagement_score, is_at_risk | Per-student ML features |
| `identify_weak_topics()` | assessment DataFrame | topic, subject, struggle_count, avg_score, avg_time | Topics where students score < 60 |
| `calculate_student_rankings()` | features DataFrame | global_rank, subject_rank, percentile | Dense rank, percent rank |
| `create_time_series_features()` | progress DataFrame | prev_confidence, confidence_change, update_sequence | Confidence trend analysis |

### Key Derived Features

```python
# Engagement score = weighted combination
engagement_score = total_study_sessions * 0.3 + total_assessments * 0.7

# At-risk flag
is_at_risk = (avg_confidence < 40) AND (avg_score < 50)
```

### Windowing Functions

```python
# Global ranking by engagement
global_window = Window.orderBy(col("engagement_score").desc())
dense_rank().over(global_window)

# Subject-specific ranking
subject_window = Window.partitionBy("subject").orderBy(col("avg_score").desc())

# Time-series lag features
time_window = Window.partitionBy("user_id", "topic").orderBy("updated_at")
lag("confidence_score", 1).over(time_window).alias("prev_confidence")
```

---

## 26.4 Meeting Processor (Spark Streaming)

### Source: `meeting_processor.py` (315 lines)

A **4-step streaming pipeline** that consumes Kafka recording events:

```mermaid\nflowchart TB\n    K[\"📨 Kafka<br/>meeting-recordings\"] --> S1\n\n    subgraph PIPELINE[\"PySpark Streaming — foreachBatch\"]\n        direction TB\n        S1[\"① Transcription<br/>POST /api/meetings/transcribe<br/>OpenAI Whisper API<br/>→ transcript + segments\"]\n        S2[\"② Summarization<br/>POST /api/meetings/summarize<br/>Google Gemini 1.5 Flash<br/>→ brief, detailed, actions\"]\n        S3[\"③ Embedding + Qdrant<br/>Chunk transcript (500-char max)<br/>Embed: text-embedding-3-small<br/>Upsert into meeting_chunks\"]\n        S4[\"④ Cassandra Analytics<br/>meeting_analytics table<br/>Partitioned by classroom_id<br/>Sorted by processed_at\"]\n        S1 --> S2 --> S3 --> S4\n    end\n\n    S3 --> QD[\"🔍 Qdrant<br/>meeting_chunks\"]\n    S4 --> CA[\"📊 Cassandra<br/>ensure_study.meeting_analytics\"]\n\n    style S1 fill:#3b82f6,color:#fff\n    style S2 fill:#8b5cf6,color:#fff\n    style S3 fill:#f59e0b,color:#000\n    style S4 fill:#10b981,color:#fff\n```

### Kafka Event Schema

```python
recording_schema = StructType([
    StructField("event_type", StringType()),
    StructField("meeting_id", StringType()),
    StructField("recording_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("classroom_id", StringType()),
    StructField("data", StructType([
        StructField("storage_url", StringType()),
        StructField("duration_seconds", IntegerType()),
        StructField("format", StringType())
    ]))
])
```

### Streaming Configuration

```python
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "meeting-recordings") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

query = parsed_df.writeStream \
    .foreachBatch(process_recording_batch) \
    .option("checkpointLocation", "/tmp/meeting_processor_checkpoint") \
    .trigger(processingTime="30 seconds") \
    .start()
```

### Cassandra Schema

```sql
CREATE TABLE meeting_analytics (
    classroom_id text,
    meeting_id text,
    processed_at timestamp,
    duration_seconds int,
    word_count int,
    PRIMARY KEY ((classroom_id), processed_at, meeting_id)
);
```

---

## 26.5 Execution

```bash
# Batch ETL
make run-etl
# → cd backend/data-pipelines && python -m pyspark etl/extract/extract_student_data.py

# Streaming
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,\
               org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
    meeting_processor.py
```
