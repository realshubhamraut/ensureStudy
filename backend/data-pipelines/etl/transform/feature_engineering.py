"""
Feature Engineering - Transform raw data into ML features
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, when, avg, count, sum as spark_sum,
    datediff, current_date, lag, lead,
    explode, from_json, array_max, array_min,
    window, dense_rank, percent_rank, row_number,
    lit, concat, coalesce
)
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, FloatType


class FeatureEngineer:
    """Transform raw data into ML-ready features"""
    
    @staticmethod
    def engineer_student_features(
        progress_df: DataFrame,
        assessment_df: DataFrame
    ) -> DataFrame:
        """
        Create student-level features for ML models.
        
        Features:
        - avg_score: Average assessment score
        - total_assessments: Number completed
        - avg_confidence: Average confidence score
        - weak_topic_count: Number of weak topics
        - study_frequency: Times studied / days active
        """
        # Aggregate assessment performance per student
        agg_assessments = assessment_df.groupBy("user_id", "subject").agg(
            avg("score").alias("avg_score"),
            count("id").alias("total_assessments"),
            avg("time_taken_seconds").alias("avg_time_seconds"),
            avg("confidence_score").alias("assessment_confidence")
        )
        
        # Aggregate progress data
        agg_progress = progress_df.groupBy("user_id", "subject").agg(
            avg("confidence_score").alias("avg_confidence"),
            spark_sum("times_studied").alias("total_study_sessions"),
            spark_sum(when(col("is_weak"), 1).otherwise(0)).alias("weak_topic_count"),
            count("topic").alias("topics_covered")
        )
        
        # Join datasets
        combined = agg_progress.join(
            agg_assessments,
            on=["user_id", "subject"],
            how="left"
        )
        
        # Add derived features
        features = combined.select(
            col("user_id"),
            col("subject"),
            coalesce(col("avg_confidence"), lit(50.0)).alias("avg_confidence"),
            coalesce(col("total_study_sessions"), lit(0)).alias("total_study_sessions"),
            coalesce(col("weak_topic_count"), lit(0)).alias("weak_topic_count"),
            coalesce(col("topics_covered"), lit(0)).alias("topics_covered"),
            coalesce(col("avg_score"), lit(0.0)).alias("avg_score"),
            coalesce(col("total_assessments"), lit(0)).alias("total_assessments"),
            coalesce(col("avg_time_seconds"), lit(0.0)).alias("avg_time_seconds"),
            # Engagement score: combination of study frequency and assessment activity
            (
                coalesce(col("total_study_sessions"), lit(0)) * 0.3 +
                coalesce(col("total_assessments"), lit(0)) * 0.7
            ).alias("engagement_score"),
            # At-risk flag: low confidence and low assessment scores
            when(
                (col("avg_confidence") < 40) & (col("avg_score") < 50),
                True
            ).otherwise(False).alias("is_at_risk")
        )
        
        return features
    
    @staticmethod
    def identify_weak_topics(assessment_df: DataFrame) -> DataFrame:
        """Identify topics where students struggle"""
        # Topics with average score < 60
        weak = assessment_df.filter(col("score") < 60).groupBy(
            "topic", "subject"
        ).agg(
            count("id").alias("struggle_count"),
            avg("score").alias("avg_score"),
            avg("time_taken_seconds").alias("avg_time")
        ).orderBy(col("struggle_count").desc())
        
        return weak
    
    @staticmethod
    def calculate_student_rankings(features_df: DataFrame) -> DataFrame:
        """Add ranking columns for leaderboard"""
        # Window for global ranking by engagement
        global_window = Window.orderBy(col("engagement_score").desc())
        
        # Window for subject-specific ranking
        subject_window = Window.partitionBy("subject").orderBy(
            col("avg_score").desc()
        )
        
        ranked = features_df.select(
            "*",
            dense_rank().over(global_window).alias("global_rank"),
            dense_rank().over(subject_window).alias("subject_rank"),
            percent_rank().over(global_window).alias("percentile")
        )
        
        return ranked
    
    @staticmethod
    def create_time_series_features(progress_df: DataFrame) -> DataFrame:
        """Create time-series features for trend analysis"""
        # Window for time-based analysis
        time_window = Window.partitionBy("user_id", "topic").orderBy("updated_at")
        
        ts_features = progress_df.select(
            "*",
            lag("confidence_score", 1).over(time_window).alias("prev_confidence"),
            (col("confidence_score") - lag("confidence_score", 1).over(time_window)).alias("confidence_change"),
            row_number().over(time_window).alias("update_sequence")
        )
        
        return ts_features


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    
    # Example usage
    engineer = FeatureEngineer()
    print("Feature engineering module loaded")
