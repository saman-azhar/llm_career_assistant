"""Page 2: Current Model - LLM vs Template Comparison"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000")

st.header("Current Model Comparison - LLM vs Template")

st.warning("""
 *Evaluation:**

**Current Setup:** FLAN-T5-base (~0.2B params) running on CPU

**Production Setup:** Would use 7B-13B model on GPU with LoRA fine-tuning
""")

# Check if data is loaded
if "cv" not in st.session_state or "jd" not in st.session_state:
    st.warning("Please enter CV and Job Description in the sidebar first.")
    st.stop()

cv = st.session_state.cv
jd = st.session_state.jd

# Call FastAPI compare endpoint
try:
    with st.spinner("Comparing LLM and Template approaches..."):
        response = requests.post(
            f"{API_URL}/compare",
            json={"cv_text": cv, "jd_text": jd},
            timeout=60
        )
    
    if response.status_code == 200:
        data = response.json()
        
        # Side-by-side comparison
        st.subheader("Side-by-Side Comparison")
        col1, col2 = st.columns(2)
        
        # LLM Output
        with col1:
            st.markdown("### LLM-Based (FLAN-T5)")
            llm = data['llm_output']
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Match Score", f"{llm['match_score']:.0%}")
            m2.metric("Inference Time", f"{llm['inference_time_ms']:.0f}ms")
            m3.metric("Match Level", llm['match_level'].upper())
            
            # Skills
            st.write("**Matched Skills:**")
            st.write(", ".join(llm['matched_skills']) if llm['matched_skills'] else "None")
            
            st.write("**Missing Skills:**")
            st.write(", ".join(llm['missing_skills'][:5]) if llm['missing_skills'] else "None")
            if len(llm['missing_skills']) > 5:
                st.caption(f"... and {len(llm['missing_skills']) - 5} more")
            
            # Cover letter preview
            if llm['cover_letter']:
                st.write("**Cover Letter Preview:**")
                st.text_area(
                    "LLM Letter",
                    value=llm['cover_letter'][:500] + ("..." if len(llm['cover_letter']) > 500 else ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.info("No cover letter generated for this match level")
        
        # Template Output
        with col2:
            st.markdown("### Template-Based (Deterministic)")
            template = data['template_output']
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Match Score", f"{template['match_score']:.0%}")
            m2.metric("Inference Time", f"{template['inference_time_ms']:.0f}ms")
            m3.metric("Match Level", template['match_level'].upper())
            
            # Skills
            st.write("**Matched Skills:**")
            st.write(", ".join(template['matched_skills']) if template['matched_skills'] else "None")
            
            st.write("**Missing Skills:**")
            st.write(", ".join(template['missing_skills'][:5]) if template['missing_skills'] else "None")
            if len(template['missing_skills']) > 5:
                st.caption(f"... and {len(template['missing_skills']) - 5} more")
            
            # Cover letter preview
            if template['cover_letter']:
                st.write("**Cover Letter Preview:**")
                st.text_area(
                    "Template Letter",
                    value=template['cover_letter'][:500] + ("..." if len(template['cover_letter']) > 500 else ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.info("No cover letter generated for this match level")
        
        # Comparison Analysis
        st.divider()
        st.subheader("Comparison Analysis")
        
        comparison = data['comparison']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Speed Advantage", comparison['speed_advantage'])
        
        with col2:
            match_agree = "Yes" if comparison['match_level_agreement'] else "❌ No"
            st.metric("Match Level Agreement", match_agree)
        
        with col3:
            skill_overlap = f"{comparison['skill_coverage_agreement']:.0%}"
            st.metric("Skill Overlap", skill_overlap)
        
        # Detailed insights
        st.markdown("### Key Insights")
        
        insights = []
        
        if comparison['llm_inference_ms'] > 1000:
            insights.append(f"**LLM is slow on CPU:** {comparison['llm_inference_ms']:.0f}ms. GPU would reduce this to ~100ms.")
        
        if comparison['match_level_agreement']:
            insights.append("**Models agree on match level:** Both approaches reach same conclusion.")
        else:
            insights.append(f"**Models disagree:** LLM says '{data['llm_output']['match_level']}', Template says '{data['template_output']['match_level']}'")
        
        if comparison['skill_coverage_agreement'] > 0.8:
            insights.append("**Strong skill alignment:** Both extract similar skills.")
        else:
            insights.append("**Different skill extraction:** Models disagree on some skills.")
        
        if comparison['match_score_diff'] > 0.1:
            insights.append(f"**Score difference:** {comparison['match_score_diff']:.1%} gap between approaches.")
        
        for insight in insights:
            st.write(insight)
        
        # Explanation
        st.divider()
        st.markdown("""
        ### What This Tells You
        
        **LLM Approach:**
        - Pro: More flexible, can generate diverse text
        - Con: Slower, hallucination risk, token limits
        - Use case: Show AI/ML skills, advanced features
        
        **Template Approach:**
        - Pro: Fast, deterministic, zero hallucination
        - Con: Less flexible, feels "templated"
        - Use case: Production, reliability, speed
        
        **The Verdict:**
        This is a **real engineering trade-off**, not a limitation.
        Production systems often choose templates for reliability.
        The LLM approach showcases your ability to integrate cutting-edge models.
        Both together = comprehensive engineering thinking.
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
