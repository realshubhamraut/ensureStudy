"""
ensureStudy Dashboards - Main App
"""
import streamlit as st

st.set_page_config(
    page_title="ensureStudy Dashboards",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 ensureStudy Evaluation Dashboards")
st.markdown("AI-first learning platform monitoring and evaluation.")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["Overview", "RAG Evaluator", "Classifier Inspector", "Pipeline Monitor"]
)

if page == "Overview":
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Students", "1,234", "+12%")
    with col2:
        st.metric("Questions Today", "456", "+8%")
    with col3:
        st.metric("Assessments Completed", "89", "+15%")
    with col4:
        st.metric("System Health", "99.9%", "+0.1%")
    
    st.markdown("---")
    
    st.subheader("Quick Links")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 RAG Evaluator
        Monitor retrieval quality, MRR scores, and citation accuracy.
        
        **Key Metrics:**
        - Mean Reciprocal Rank
        - Citation Accuracy
        - Retrieval Latency
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Classifier Inspector
        Analyze moderation classifier performance.
        
        **Key Metrics:**
        - Precision / Recall
        - Confusion Matrix
        - F1-Score Trends
        """)
    
    with col3:
        st.markdown("""
        ### ⚙️ Pipeline Monitor
        Track ETL jobs and Kafka consumer health.
        
        **Key Metrics:**
        - Job Success Rate
        - Processing Throughput
        - Consumer Lag
        """)

elif page == "RAG Evaluator":
    st.header("📊 RAG Evaluator")
    st.info("Retrieval quality and RAG pipeline monitoring")
    
    import plotly.graph_objects as go
    import numpy as np
    
    # Simulated metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MRR Score", "0.85", "+0.02")
    with col2:
        st.metric("Citation Accuracy", "92.5%", "+1.5%")
    with col3:
        st.metric("Avg Retrieval Time", "145ms", "-12ms")
    with col4:
        st.metric("Documents Retrieved", "12,456", "+234")
    
    st.subheader("MRR Score Trend (Last 7 Days)")
    dates = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    mrr_scores = [0.82, 0.83, 0.81, 0.84, 0.85, 0.86, 0.85]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=mrr_scores, mode='lines+markers', name='MRR'))
    fig.update_layout(yaxis_range=[0.7, 1.0])
    st.plotly_chart(fig, use_container_width=False)

elif page == "Classifier Inspector":
    st.header("🔍 Classifier Inspector")
    st.info("Moderation classifier performance dashboard")
    
    import numpy as np
    import plotly.figure_factory as ff
    
    # Synthetic confusion matrix
    cm = np.array([[850, 50], [30, 70]])
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['Academic', 'Off-Topic'],
        y=['Academic', 'Off-Topic'],
        colorscale='Blues'
    )
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "92.0%")
    with col2:
        st.metric("Precision", "94.4%")
    with col3:
        st.metric("Recall", "70.0%")
    with col4:
        st.metric("F1-Score", "80.5%")

elif page == "Pipeline Monitor":
    st.header("⚙️ Pipeline Monitor")
    st.info("ETL and streaming pipeline health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ETL Jobs (24h)", "48/48", "100%")
    with col2:
        st.metric("Kafka Consumer Lag", "234", "-12")
    with col3:
        st.metric("Records Processed", "1.2M", "+100K")
    
    st.subheader("Recent Jobs")
    import pandas as pd
    
    jobs = pd.DataFrame({
        "Job": ["daily_etl_pipeline", "weekly_analytics", "ml_feature_refresh"],
        "Last Run": ["2025-12-21 02:00", "2025-12-21 00:00", "2025-12-21 06:00"],
        "Duration": ["4m 32s", "12m 15s", "8m 45s"],
        "Status": ["✅ Success", "✅ Success", "✅ Success"],
        "Records": [125034, 892456, 45678]
    })
    st.dataframe(jobs, use_container_width=True)
