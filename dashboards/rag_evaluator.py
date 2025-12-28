"""
Streamlit RAG Evaluator Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="RAG Evaluator - ensureStudy",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Pipeline Evaluation Dashboard")
st.markdown("Monitor retrieval quality, citation accuracy, and latency metrics.")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.selectbox(
    "Date Range",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days"]
)
subject_filter = st.sidebar.multiselect(
    "Subjects",
    ["Biology", "Chemistry", "Physics", "Math", "History", "All"],
    default=["All"]
)

# Generate synthetic data (replace with real data in production)
np.random.seed(42)
days = 30 if "30" in date_range else (90 if "90" in date_range else 7)
dates = [datetime.now() - timedelta(days=i) for i in range(days)]
dates.reverse()

mrr_scores = np.random.uniform(0.70, 0.85, len(dates))
mrr_scores = np.convolve(mrr_scores, np.ones(3)/3, mode='same')  # Smooth

citation_accuracy = np.random.uniform(0.88, 0.98, len(dates))
retrieval_latency = np.random.uniform(0.8, 2.5, len(dates))

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_mrr = mrr_scores[-1]
    prev_mrr = mrr_scores[-2]
    delta = (current_mrr - prev_mrr) * 100
    st.metric(
        "Mean Reciprocal Rank (MRR)",
        f"{current_mrr:.3f}",
        f"{delta:+.2f}%",
        delta_color="normal"
    )

with col2:
    current_citation = citation_accuracy[-1]
    st.metric(
        "Citation Accuracy",
        f"{current_citation:.1%}",
        f"{(citation_accuracy[-1] - citation_accuracy[-2])*100:+.1f}%"
    )

with col3:
    current_latency = retrieval_latency[-1]
    st.metric(
        "Avg Latency",
        f"{current_latency:.2f}s",
        f"{retrieval_latency[-1] - retrieval_latency[-2]:+.2f}s",
        delta_color="inverse"
    )

with col4:
    success_rate = np.random.uniform(0.92, 0.98)
    st.metric(
        "Retrieval Success Rate",
        f"{success_rate:.1%}",
        "+0.3%"
    )

# Charts
st.subheader("📈 Metrics Over Time")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=mrr_scores,
        mode='lines+markers',
        name='MRR',
        line=dict(color='#4CAF50', width=2),
        marker=dict(size=4)
    ))
    fig.add_hline(y=0.75, line_dash="dash", line_color="red", 
                  annotation_text="Target: 0.75")
    fig.update_layout(
        title="Mean Reciprocal Rank (MRR)",
        xaxis_title="Date",
        yaxis_title="MRR Score",
        yaxis=dict(range=[0.6, 1.0]),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=retrieval_latency,
        mode='lines+markers',
        name='Latency',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.2)'
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="red",
                  annotation_text="SLA: 3s")
    fig.update_layout(
        title="Retrieval Latency",
        xaxis_title="Date",
        yaxis_title="Latency (seconds)",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

# Subject breakdown
st.subheader("📚 Performance by Subject")

subjects = ["Biology", "Chemistry", "Physics", "Math", "History"]
subject_data = pd.DataFrame({
    "Subject": subjects,
    "MRR": np.random.uniform(0.70, 0.90, len(subjects)),
    "Citation Accuracy": np.random.uniform(0.85, 0.98, len(subjects)),
    "Avg Latency (s)": np.random.uniform(1.0, 2.5, len(subjects)),
    "Queries": np.random.randint(100, 500, len(subjects))
})

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        subject_data,
        x="Subject",
        y="MRR",
        color="MRR",
        color_continuous_scale="Greens",
        title="MRR by Subject"
    )
    fig.add_hline(y=0.75, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        subject_data,
        x="Subject",
        y="Citation Accuracy",
        color="Citation Accuracy",
        color_continuous_scale="Blues",
        title="Citation Accuracy by Subject"
    )
    fig.add_hline(y=0.90, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# Sample queries table
st.subheader("🔍 Sample Query Analysis")

sample_queries = pd.DataFrame({
    "Query": [
        "How does photosynthesis work?",
        "Explain the quadratic formula",
        "What causes earthquakes?",
        "Define mitochondria function",
        "Explain Newton's third law"
    ],
    "Top-1 Score": [0.92, 0.88, 0.85, 0.91, 0.89],
    "MRR": [0.92, 0.88, 0.85, 0.91, 0.89],
    "Citations": [3, 2, 2, 3, 2],
    "Latency (s)": [1.2, 1.5, 1.8, 1.1, 1.4],
    "Status": ["✅", "✅", "⚠️", "✅", "✅"]
})

st.dataframe(sample_queries, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
