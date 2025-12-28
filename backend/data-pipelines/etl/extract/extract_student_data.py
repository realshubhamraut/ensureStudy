"""
PySpark ETL - Extract student data from PostgreSQL
"""
import os
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_timestamp, lit


def create_spark_session() -> SparkSession:
    """Create Spark session with JDBC support"""
    return SparkSession.builder \
        .appName("EnsureStudy-ETL") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.5.0") \
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "4g")) \
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g")) \
        .getOrCreate()


class StudentDataExtractor:
    """Extract student data from PostgreSQL using PySpark"""
    
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or create_spark_session()
        self.jdbc_url = f"jdbc:postgresql://{os.getenv('DB_HOST', 'localhost')}:5432/{os.getenv('DB_NAME', 'ensure_study')}"
        self.db_properties = {
            "user": os.getenv("DB_USER", "ensure_study_user"),
            "password": os.getenv("DB_PASSWORD", "secure_password_123"),
            "driver": "org.postgresql.Driver"
        }
    
    def extract_progress_data(
        self,
        start_date: str,
        end_date: str
    ) -> DataFrame:
        """Extract progress records with user info"""
        query = f"""
        (SELECT 
            p.id, p.user_id, p.topic, p.subject,
            p.confidence_score, p.times_studied,
            p.last_studied, p.is_weak,
            p.created_at, p.updated_at,
            u.username, u.role, u.class_id, u.school_id
        FROM progress p
        JOIN users u ON p.user_id = u.id
        WHERE p.updated_at BETWEEN '{start_date}' AND '{end_date}'
        ) AS progress_data
        """
        
        return self.spark.read.jdbc(
            url=self.jdbc_url,
            table=query,
            properties=self.db_properties
        )
    
    def extract_assessment_results(
        self,
        start_date: str,
        end_date: str
    ) -> DataFrame:
        """Extract assessment results"""
        query = f"""
        (SELECT 
            ar.id, ar.user_id, ar.assessment_id,
            ar.score, ar.max_score, ar.time_taken_seconds,
            ar.confidence_score, ar.completed_at,
            a.topic, a.subject, a.difficulty
        FROM assessment_results ar
        JOIN assessments a ON ar.assessment_id = a.id
        WHERE ar.completed_at BETWEEN '{start_date}' AND '{end_date}'
        ) AS assessment_data
        """
        
        return self.spark.read.jdbc(
            url=self.jdbc_url,
            table=query,
            properties=self.db_properties
        )
    
    def extract_leaderboard(self) -> DataFrame:
        """Extract current leaderboard data"""
        query = """
        (SELECT 
            l.user_id, l.global_points, l.class_points,
            l.study_streak, l.level, l.xp,
            u.username, u.class_id, u.school_id
        FROM leaderboard l
        JOIN users u ON l.user_id = u.id
        WHERE u.is_active = true
        ) AS leaderboard_data
        """
        
        return self.spark.read.jdbc(
            url=self.jdbc_url,
            table=query,
            properties=self.db_properties
        )
    
    def extract_chat_sessions(
        self,
        start_date: str,
        end_date: str
    ) -> DataFrame:
        """Extract chat session data for analytics"""
        query = f"""
        (SELECT 
            cs.id, cs.user_id, cs.title,
            json_array_length(cs.messages::json) as message_count,
            cs.created_at, cs.updated_at,
            u.class_id, u.school_id
        FROM chat_sessions cs
        JOIN users u ON cs.user_id = u.id
        WHERE cs.updated_at BETWEEN '{start_date}' AND '{end_date}'
        ) AS chat_data
        """
        
        return self.spark.read.jdbc(
            url=self.jdbc_url,
            table=query,
            properties=self.db_properties
        )


def run_extraction(days_back: int = 7):
    """Run extraction for the past N days"""
    extractor = StudentDataExtractor()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Extracting data from {start_str} to {end_str}")
    
    # Extract each dataset
    progress_df = extractor.extract_progress_data(start_str, end_str)
    print(f"✓ Extracted {progress_df.count()} progress records")
    
    assessment_df = extractor.extract_assessment_results(start_str, end_str)
    print(f"✓ Extracted {assessment_df.count()} assessment results")
    
    leaderboard_df = extractor.extract_leaderboard()
    print(f"✓ Extracted {leaderboard_df.count()} leaderboard entries")
    
    return {
        "progress": progress_df,
        "assessments": assessment_df,
        "leaderboard": leaderboard_df
    }


if __name__ == "__main__":
    run_extraction()
