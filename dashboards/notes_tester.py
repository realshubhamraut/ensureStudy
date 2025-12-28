"""
Notes Digitization Tester - Streamlit Dashboard
Test the video processing, page extraction, and PDF generation features
Uses AI Service API instead of direct imports for better isolation.
"""
import streamlit as st
import requests
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Notes Digitization Tester",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")

st.title("📝 Notes Digitization Tester")
st.markdown("Test video processing, page extraction, and PDF generation")

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("### Backend URLs")
backend_url = st.sidebar.text_input("Core Backend", BACKEND_URL)
ai_url = st.sidebar.text_input("AI Service", AI_SERVICE_URL)

# Check service health
st.sidebar.markdown("### Service Health")
try:
    core_health = requests.get(f"{backend_url}/health", timeout=2)
    st.sidebar.success("✅ Core Backend: Online")
except:
    st.sidebar.error("❌ Core Backend: Offline")

try:
    ai_health = requests.get(f"{ai_url}/health", timeout=2)
    st.sidebar.success("✅ AI Service: Online")
except:
    st.sidebar.error("❌ AI Service: Offline")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Video Processor (Local)", 
    "🖼️ Image Enhancer (Local)", 
    "📄 PDF Generator",
    "🔍 Full Pipeline Test (API)"
])

with tab1:
    st.header("🎬 Video Frame Extraction")
    st.markdown("Extract frames from video using OpenCV (runs locally)")
    
    uploaded_video = st.file_uploader(
        "Upload a video of handwritten notes",
        type=["mp4", "mov", "avi", "webm", "mkv"],
        key="video_upload"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Parameters")
        frame_interval = st.slider("Frame Interval (seconds)", 0.5, 5.0, 1.0, 0.5)
        blur_threshold = st.slider("Blur Threshold", 50, 200, 100)
        max_frames = st.slider("Max Frames", 5, 50, 20)
    
    with col1:
        if uploaded_video:
            st.video(uploaded_video)
            
            if st.button("🚀 Extract Frames", key="extract_btn"):
                with st.spinner("Processing video..."):
                    # Save uploaded video to temp file
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(uploaded_video.read())
                        video_path = tmp.name
                    
                    try:
                        # Simple frame extraction using OpenCV
                        cap = cv2.VideoCapture(video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        frames = []
                        frame_skip = int(fps * frame_interval)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        frame_idx = 0
                        while len(frames) < max_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            
                            if not ret:
                                break
                            
                            # Check blur
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                            
                            if blur_score > blur_threshold:
                                frames.append(frame)
                                status_text.text(f"Extracted frame {len(frames)} (blur: {blur_score:.1f})")
                            
                            frame_idx += frame_skip
                            progress = min(1.0, frame_idx / total_frames)
                            progress_bar.progress(progress)
                        
                        cap.release()
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ Extracted {len(frames)} frames!")
                        
                        # Save frames to session state
                        st.session_state['extracted_frames'] = frames
                        
                        # Display frames
                        st.markdown("### Extracted Frames")
                        cols = st.columns(min(4, len(frames)))
                        for i, frame in enumerate(frames):
                            with cols[i % 4]:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.image(frame_rgb, caption=f"Frame {i+1}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        os.unlink(video_path)

with tab2:
    st.header("🖼️ Image Enhancement")
    st.markdown("Enhance images using OpenCV (runs locally)")
    
    uploaded_images = st.file_uploader(
        "Upload images to enhance",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="image_upload"
    )
    
    if uploaded_images:
        st.markdown("### Original Images")
        cols = st.columns(min(4, len(uploaded_images)))
        for i, img in enumerate(uploaded_images):
            with cols[i % 4]:
                st.image(img, caption=f"Original {i+1}")
        
        if st.button("✨ Enhance Images", key="enhance_btn"):
            with st.spinner("Enhancing..."):
                try:
                    enhanced_images = []
                    
                    for i, uploaded in enumerate(uploaded_images):
                        # Load image
                        pil_image = Image.open(uploaded)
                        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        
                        # Simple enhancement pipeline
                        # 1. Convert to LAB and apply CLAHE
                        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l = clahe.apply(l)
                        lab = cv2.merge([l, a, b])
                        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                        
                        # 2. Denoise
                        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 8, 8, 7, 21)
                        
                        # 3. Sharpen
                        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
                        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
                        
                        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        enhanced_images.append(enhanced_rgb)
                    
                    st.markdown("### Enhanced Images")
                    cols = st.columns(min(4, len(enhanced_images)))
                    for i, img in enumerate(enhanced_images):
                        with cols[i % 4]:
                            st.image(img, caption=f"Enhanced {i+1}")
                    
                    st.session_state['enhanced_images'] = enhanced_images
                    st.success("Enhancement complete!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

with tab3:
    st.header("📄 PDF Generation")
    st.markdown("Combine images into a PDF document")
    
    if 'extracted_frames' in st.session_state and st.session_state['extracted_frames']:
        st.info(f"Found {len(st.session_state['extracted_frames'])} extracted frames")
        
        if st.button("📄 Generate PDF from extracted frames", key="pdf_btn"):
            with st.spinner("Generating PDF..."):
                try:
                    import img2pdf
                    
                    # Convert frames to PNG bytes
                    image_bytes = []
                    for frame in st.session_state['extracted_frames']:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        # Save to bytes
                        buf = io.BytesIO()
                        pil_img.save(buf, format='PNG')
                        buf.seek(0)
                        image_bytes.append(buf.getvalue())
                    
                    # Create PDF
                    pdf_bytes = img2pdf.convert(image_bytes)
                    
                    st.success("PDF generated!")
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name="digitized_notes.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("No extracted frames found. Use the Video Processor tab first.")
        
        # Manual upload
        st.markdown("---")
        st.markdown("### Or upload images manually")
        pdf_images = st.file_uploader(
            "Upload images to combine into PDF",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="pdf_images"
        )
        
        if pdf_images and st.button("📄 Generate PDF from uploads", key="pdf_manual_btn"):
            with st.spinner("Generating..."):
                try:
                    import img2pdf
                    
                    image_bytes = []
                    for img in pdf_images:
                        image_bytes.append(img.read())
                    
                    pdf_bytes = img2pdf.convert(image_bytes)
                    
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name="digitized_notes.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab4:
    st.header("🔍 Full Pipeline Test (via API)")
    st.markdown("Test the complete backend API pipeline")
    
    # Test credentials
    st.markdown("### Test Login")
    col1, col2 = st.columns(2)
    with col1:
        test_email = st.text_input("Email", value="student@gmail.com")
    with col2:
        test_password = st.text_input("Password", value="Zxc@1234", type="password")
    
    classroom_id = st.text_input("Classroom ID", placeholder="Enter a classroom ID to test")
    
    if st.button("🔐 Login & Get Token"):
        try:
            response = requests.post(
                f"{backend_url}/api/auth/login",
                json={"email": test_email, "password": test_password},
                timeout=10
            )
            if response.ok:
                data = response.json()
                st.session_state['auth_token'] = data.get('access_token')
                st.session_state['user'] = data.get('user', {})
                st.success(f"Logged in as: {st.session_state['user'].get('name', 'User')}")
            else:
                st.error(f"Login failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if 'auth_token' in st.session_state:
        st.markdown("---")
        st.markdown("### Upload Video for Processing")
        
        test_video = st.file_uploader(
            "Upload video",
            type=["mp4", "mov"],
            key="pipeline_video"
        )
        
        title = st.text_input("Notes Title", value="Test Notes")
        
        if test_video and classroom_id and st.button("📤 Upload & Process"):
            with st.spinner("Uploading and processing..."):
                try:
                    files = {'file': (test_video.name, test_video, 'video/mp4')}
                    data = {
                        'classroom_id': classroom_id,
                        'title': title,
                        'description': 'Test upload from Streamlit'
                    }
                    headers = {'Authorization': f"Bearer {st.session_state['auth_token']}"}
                    
                    response = requests.post(
                        f"{backend_url}/api/notes/upload",
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.ok:
                        result = response.json()
                        job_id = result.get('job', {}).get('id')
                        st.success(f"Uploaded! Job ID: {job_id}")
                        
                        # Poll for status
                        st.markdown("### Processing Status")
                        status_container = st.empty()
                        
                        import time
                        for _ in range(60):
                            status_resp = requests.get(
                                f"{backend_url}/api/notes/jobs/{job_id}",
                                headers=headers,
                                timeout=10
                            )
                            if status_resp.ok:
                                job = status_resp.json().get('job', {})
                                status = job.get('status')
                                progress = job.get('progress_percent', 0)
                                step = job.get('current_step', '')
                                
                                status_container.markdown(f"""
                                **Status:** {status}  
                                **Progress:** {progress}%  
                                **Step:** {step}
                                """)
                                
                                if status == 'completed':
                                    st.success("✅ Processing complete!")
                                    break
                                elif status == 'failed':
                                    st.error(f"Processing failed: {job.get('error_message')}")
                                    break
                            
                            time.sleep(3)
                    else:
                        st.error(f"Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("*ensureStudy - AI-Powered Learning Platform*")
