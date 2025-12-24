"""Career Assistant - Main Streamlit App"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Career Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚀 LLM Career Assistant")
st.markdown("""
An intelligent system that analyzes job descriptions, compares them with your CV, 
identifies missing skills, and generates tailored cover letters.

This portfolio demonstrates:
1. **Ideal Output** - Product vision using template-based generation
2. **Current Model** - Real LLM behavior with honest performance evaluation
3. **Scaling Plan** - Production architecture and engineering decisions

---
""")

# Sidebar for shared inputs
st.sidebar.header("📋 Input Data")
st.sidebar.write("Enter your CV and Job Description to analyze:")

cv_text = st.sidebar.text_area(
    "Your CV",
    height=150,
    placeholder="Paste your resume or CV here...",
    key="cv_input"
)

jd_text = st.sidebar.text_area(
    "Job Description",
    height=150,
    placeholder="Paste the job description here...",
    key="jd_input"
)

# Store in session state for page access
if cv_text and jd_text:
    st.session_state.cv = cv_text
    st.session_state.jd = jd_text
    st.sidebar.success("✅ Data loaded. Navigate to other tabs to analyze.")
else:
    st.sidebar.warning("⚠️ Please enter both CV and Job Description to proceed.")

# Homepage content
st.markdown("""
## How This Works

### Three-Tab Architecture

**Tab 1: Ideal Output**
- Demonstrates the system's target behavior using intelligent templates
- Shows what perfect CV-JD alignment looks like
- This is your product vision

**Tab 2: Current Model**
- Shows real LLM output (FLAN-T5 base on CPU)
- Compares LLM vs template approach side-by-side
- Transparent about constraints: "Quantized model for demo, production would use 7B+ on GPU"
- Demonstrates honest evaluation, not excuses

**Tab 3: Scaling Plan**
- Engineering roadmap to production
- Identifies bottlenecks and solutions
- Shows how to scale with better infrastructure
- Signals production-ready thinking

---

## Key Technical Details

### Current Stack
- **Embedding Model:** intfloat/e5-base-v2 (384-dim vectors)
- **LLM:** google/flan-t5-base (60M params, CPU-friendly)
- **Vector DB:** Qdrant with COSINE similarity
- **API:** FastAPI endpoints for real-time inference
- **Chunking:** 300-char chunks with 70-char overlap for RAG

### Matching Logic (3-Tier Assessment)
- **Poor Match (<60%):** Shows skill gaps, no cover letter generated
- **Moderate Match (60-80%):** Generates letter addressing gaps
- **Good Match (>80%):** Professional letter celebrating strengths

### Why This Approach?
Shows both engineering pragmatism (templates work great) and cutting-edge thinking (LLM/RAG).
Recruiters judge the process, not just the output. This demonstrates:
- Problem definition ✅
- Pipeline design ✅
- Intelligent evaluation ✅
- Honest trade-offs ✅
- Scaling strategy ✅

---

## Getting Started

1. Enter your CV in the sidebar
2. Paste the job description
3. Navigate to "Ideal Output" tab to see template-based results
4. Check "Current Model" for LLM performance and comparison
5. Review "Scaling Plan" for production architecture

**Need help?** Each tab includes disclaimers and context.

---

*Built with FastAPI, Streamlit, Qdrant, and Transformers*
""")
