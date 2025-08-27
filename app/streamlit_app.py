import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration: prefer environment, fallback to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="TDS Virtual TA", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .source-link {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .debug-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 TDS Virtual TA</h1>', unsafe_allow_html=True)
    st.caption("Ask questions about course materials or upload an image. Powered by the existing FastAPI backend.")
    
    # Health check at the top
    with st.expander("🔍 API Status", expanded=False):
        if st.button("Check API Health", key="health_check"):
            check_api_health(API_BASE_URL)
    
    # Main tabs
    tab_q, tab_img = st.tabs(["📝 Text Question", "🖼️ Image Analysis"])
    
    with tab_q:
        render_text_question_tab(API_BASE_URL, 60, False)
    
    with tab_img:
        render_image_analysis_tab(API_BASE_URL, 60, False)

def render_text_question_tab(api_url, timeout_s, show_debug):
    """Render the text question tab"""
    st.subheader("Ask a Question")
    
    question = st.text_area(
        "Your question about the course materials:",
        height=120, 
        placeholder="e.g., What are the requirements for Project 1? How do I use Docker in this course?"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ask_btn = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    
    if ask_btn:
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            process_text_question(question, api_url, timeout_s, show_debug)

def render_image_analysis_tab(api_url, timeout_s, show_debug):
    """Render the image analysis tab"""
    st.subheader("Analyze an Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        image_file = st.file_uploader(
            "Upload an image (PNG/JPG/JPEG):",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to analyze its content"
        )
        
        image_question = st.text_area(
            "Optional question about the image:",
            height=100,
            placeholder="e.g., What does this diagram show? Explain this error message."
        )
    
    with col2:
        if image_file:
            st.image(image_file, caption="Image Preview", use_container_width=True)
            st.info(f"📁 File: {image_file.name}")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ask_img_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    
    if ask_img_btn:
        if not image_file and not image_question.strip():
            st.warning("⚠️ Please upload an image or provide a question.")
        else:
            process_image_question(image_question.strip() or None, image_file, api_url, timeout_s, show_debug)

from typing import Optional

def call_api(question_text: Optional[str], image_file_obj, api_url: str, timeout_s: int) -> dict:
    """Call the FastAPI backend API"""
    url = f"{api_url.rstrip('/')}/api"
    # Debug: print the URL being called
    print(f"DEBUG: Calling API at: {url}")
    files = {}
    data = {}
    
    if question_text:
        data["question"] = question_text

    if image_file_obj:
        files["image"] = (
            image_file_obj.name, 
            image_file_obj.getvalue(), 
            image_file_obj.type
        )

    try:
        t0 = time.time()
        
        if files:
            # Use multipart form for image uploads
            resp = requests.post(url, data=data, files=files, timeout=timeout_s)
        else:
            # Use JSON for text-only questions
            resp = requests.post(url, json={"question": question_text}, timeout=timeout_s)

        elapsed = time.time() - t0
        
        if resp.status_code != 200:
            return {
                "answer": f"❌ Request failed: HTTP {resp.status_code}",
                "links": [],
                "error": f"HTTP {resp.status_code}",
                "elapsed": elapsed
            }

        result = resp.json()
        result["elapsed"] = elapsed
        return result
        
    except requests.exceptions.Timeout:
        return {
            "answer": f"⏰ Request timed out after {timeout_s} seconds",
            "links": [],
            "error": "Timeout",
            "elapsed": timeout_s
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "🔌 Connection error. Please check the API URL and ensure the backend is running.",
            "links": [],
            "error": "Connection Error",
            "elapsed": 0
        }
    except requests.RequestException as e:
        return {
            "answer": f"❌ Network error: {str(e)}",
            "links": [],
            "error": str(e),
            "elapsed": 0
        }

def process_text_question(question: str, api_url: str, timeout_s: int, show_debug: bool):
    """Process a text question"""
    with st.spinner("🔍 Searching course materials and generating answer..."):
        result = call_api(question, None, api_url, timeout_s)
    
    # Display answer
    st.subheader("💡 Answer")
    st.write(result.get("answer", "No answer received."))
    
    # Display sources
    links = result.get("links", [])
    if links:
        st.subheader("📚 Sources")
        for i, link in enumerate(links, 1):
            url = link.get("url", "")
            text = link.get("text", f"Source {i}")
            
            if url:
                st.markdown(f"""
                <div class="source-link">
                    <strong>{i}.</strong> 
                    <a href="{url}" target="_blank">{text}</a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class=\"source-link\"><strong>{i}.</strong> {text}</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No sources provided.")

def process_image_question(question: Optional[str], image_file, api_url: str, timeout_s: int, show_debug: bool):
    """Process an image question"""
    with st.spinner("🔍 Analyzing image and generating answer..."):
        result = call_api(question, image_file, api_url, timeout_s)
    
    # Display answer
    st.subheader("💡 Answer")
    st.write(result.get("answer", "No answer received."))
    
    # Display sources
    links = result.get("links", [])
    if links:
        st.subheader("📚 Sources")
        for i, link in enumerate(links, 1):
            url = link.get("url", "")
            text = link.get("text", f"Source {i}")
            
            if url:
                st.markdown(f"""
                <div class="source-link">
                    <strong>{i}.</strong> 
                    <a href="{url}" target="_blank">{text}</a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class=\"source-link\"><strong>{i}.</strong> {text}</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No sources provided.")

def check_api_health(api_url: str):
    """Check the health of the API"""
    try:
        health_url = f"{api_url.rstrip('/')}/"
        resp = requests.get(health_url, timeout=10)
        
        if resp.ok:
            st.success("✅ API is healthy and responding")
            try:
                data = resp.json()
                if "endpoints" in data:
                    st.info(f"Available endpoints: {', '.join(data['endpoints'])}")
            except:
                pass
        else:
            st.warning(f"⚠️ API responded with HTTP {resp.status_code}")
            
    except requests.exceptions.Timeout:
        st.error("⏰ API health check timed out")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to API. Check the URL and ensure the backend is running.")
    except Exception as e:
        st.error(f"❌ Health check failed: {str(e)}")

if __name__ == "__main__":
    main()
