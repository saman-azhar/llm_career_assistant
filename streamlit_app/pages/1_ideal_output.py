"""Page 1: Ideal Output - Template-Based Generation"""

import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://api:8000")


st.info("""
This demonstrates what the system is **designed to produce**.
Using intelligent templates with actual skill matching and professional wording.

""")


# Check if data is loaded
if "cv" not in st.session_state or "jd" not in st.session_state:
    st.warning("Please enter CV and Job Description in the sidebar first.")
    st.stop()

cv = st.session_state.cv
jd = st.session_state.jd

# Call FastAPI to get template-based output only
try:
    with st.spinner("Analyzing CV and job description..."):
        response = requests.post(
            f"{API_URL}/match",
            json={"cv_text": cv, "jd_text": jd}
        )
    
    if response.status_code == 200:
        result = response.json()
        
        # Display metrics
        st.subheader("Match Assessment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_score = result.get("match_score", 0)
            st.metric(
                "Match Score",
                f"{match_score:.0%}",
                delta="Strong fit" if match_score > 0.8 else ("Moderate" if match_score > 0.6 else "Needs development")
            )
        
        with col2:
            verdict = result.get("verdict", "unknown")
            st.metric("Verdict", verdict.upper())
        
        with col3:
            matched = len(result.get("matched_skills", []))
            missing = len(result.get("missing_skills", []))
            st.metric("Skills", f"{matched} matched / {missing} missing")
        
        # Skills breakdown
        st.subheader("Skill Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Matched Skills:**")
            matched_skills = result.get("matched_skills", [])
            if matched_skills:
                for skill in matched_skills:
                    st.write(f"- {skill}")
            else:
                st.write("*No matched skills found*")
        
        with col2:
            st.write("**Missing Skills:**")
            missing_skills = result.get("missing_skills", [])
            if missing_skills:
                for skill in missing_skills[:10]:  # Show top 10
                    st.write(f"- {skill}")
                if len(missing_skills) > 10:
                    st.write(f"... and {len(missing_skills) - 10} more")
            else:
                st.write("*No missing skills - perfect alignment!*")
        
        
        # Cover letter
        cover_letter = result.get("cover_letter")
        if cover_letter:
            st.subheader("Generated Cover Letter")
            st.text_area(
                "Cover Letter",
                value=cover_letter,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            st.warning("**Note:** Cover letter not generated for poor matches. Focus on developing the missing skills first.")
        
        # Footer
        st.divider()
        st.markdown("""
        Go to the "Current Model" tab to see how real inference performs.
        """)
        
    else:
        st.error(f"API Error: {response.status_code}")
        st.write(response.text)

except requests.exceptions.ConnectionError:
    st.error("""
    **Cannot connect to API**
    
    Make sure FastAPI is running:
    ```bash
    uvicorn career_assistant.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    """)
except Exception as e:
    st.error(f"Error: {str(e)}")
