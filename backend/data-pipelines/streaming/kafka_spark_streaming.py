"""
PySpark Kafka Streaming - Real-time ETL from Kafka
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, count, avg, current_timestamp,
    expr, to_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, TimestampType, ArrayType, MapType
)


def create_spark_session() -> SparkSession:
    """Create Spark session with Kafka support"""
    return SparkSession.builder \
        .appName("EnsureStudy-Streaming") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoints") \
        .getOrCreate()


def get_student_event_schema() -> StructType:
    """Schema for student events"""
    return StructType([
        StructField("event_type", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("data", MapType(StringType(), StringType()), True),
        StructField("metadata", MapType(StringType(), StringType()), True),
        StructField("source", StringType(), True)
    ])


def get_assessment_schema() -> StructType:
    """Schema for assessment submissions"""
    return StructType([
        StructField("event_type", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("assessment_id", StringType(), True),
        StructField("score", FloatType(), True),
        StructField("time_taken_seconds", IntegerType(), True),
        StructField("topic", StringType(), True),
        StructField("subject", StringType(), True),
        StructField("timestamp", StringType(), True)
    ])


class KafkaSparkStreaming:
    """
    Real-time streaming processor using Spark Structured Streaming + Kafka.
    
    Features:
    - Consume from Kafka topics
    - Apply transformations
    - Write to PostgreSQL, S3, or other sinks
    """
    
    def __init__(self):
        self.spark = create_spark_session()
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    def read_from_kafka(self, topic: str, schema: StructType):
        """Read streaming data from Kafka topic"""
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .load() \
            .select(
                from_json(col("value").cast("string"), schema).alias("data"),
                col("timestamp").alias("kafka_timestamp")
            ) \
            .select("data.*", "kafka_timestamp")
    
    def stream_student_events(self):
        """Stream and aggregate student events"""
        schema = get_student_event_schema()
        
        df = self.read_from_kafka("student-events", schema)
        
        # Aggregate by event type per 5-minute window
        aggregated = df \
            .withColumn("event_time", to_timestamp(col("timestamp"))) \
            .withWatermark("event_time", "10 minutes") \
            .groupBy(
                window(col("event_time"), "5 minutes"),
                col("event_type")
            ) \
            .agg(
                count("*").alias("event_count"),
                expr("count(distinct user_id)").alias("unique_users")
            )
        
        # Write to console for debugging
        query = aggregated \
            .writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", "false") \
            .trigger(processingTime="1 minute") \
            .start()
        
        return query
    
    def stream_assessment_scores(self):
        """Stream and aggregate assessment scores"""
        schema = get_assessment_schema()
        
        df = self.read_from_kafka("assessment-submissions", schema)
        
        # Filter to submissions only
        submissions = df.filter(col("event_type") == "submission")
        
        # Aggregate scores by subject per 1-hour window
        aggregated = submissions \
            .withColumn("event_time", to_timestamp(col("timestamp"))) \
            .withWatermark("event_time", "1 hour") \
            .groupBy(
                window(col("event_time"), "1 hour"),
                col("subject")
            ) \
            .agg(
                avg("score").alias("avg_score"),
                count("*").alias("submission_count"),
                avg("time_taken_seconds").alias("avg_time_seconds")
            )
        
        query = aggregated \
            .writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", "false") \
            .trigger(processingTime="5 minutes") \
            .start()
        
        return query
    
    def write_to_postgres(self, df, table_name: str, mode: str = "append"):
        """Write streaming dataframe to PostgreSQL"""
        jdbc_url = f"jdbc:postgresql://{os.getenv('DB_HOST', 'localhost')}:5432/ensure_study"
        
        def write_batch(batch_df, batch_id):
            batch_df.write \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table_name) \
                .option("user", os.getenv("DB_USER")) \
                .option("password", os.getenv("DB_PASSWORD")) \
                .option("driver", "org.postgresql.Driver") \
                .mode(mode) \
                .save()
        
        return df.writeStream \
            .foreachBatch(write_batch) \
            .outputMode("update") \
            .start()
    
    def stop(self):
        """Stop all streaming queries"""
        for query in self.spark.streams.active:
            query.stop()
        self.spark.stop()


def run_streaming():
    """Run the streaming pipeline"""
    streaming = KafkaSparkStreaming()
    
    try:
        print("Starting Kafka-Spark streaming...")
        
        # Start student events stream
        student_query = streaming.stream_student_events()
        
        # Start assessment stream
        assessment_query = streaming.stream_assessment_scores()
        
        # Wait for termination
        student_query.awaitTermination()
        
    except KeyboardInterrupt:
        print("Stopping streaming...")
        streaming.stop()


if __name__ == "__main__":
    run_streaming()
