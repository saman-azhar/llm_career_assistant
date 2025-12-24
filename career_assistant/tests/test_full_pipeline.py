#!/usr/bin/env python
"""Test full RAG pipeline end-to-end."""

import sys
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
from career_assistant.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # Test with RAW CV and JD samples from actual training data
    # Using raw non-preprocessed text to match real-world input format
    
    cv_text = """
    Education Details
    May 2013 to May 2017 B.E Computer Science Data Scientist
    Data Scientist - Matelabs
    
    Skill Details
    Python - Experience - 24 months
    scikit-learn - Experience - 12 months
    AWS - Experience - 18 months
    Keras - Experience - 12 months
    
    Machine Learning: Regression, SVM, Naive Bayes, KNN, Random Forest, Decision Trees, 
    Boosting techniques, Cluster Analysis, Word Embedding, Sentiment Analysis, 
    Natural Language Processing, Dimensionality reduction, Topic Modelling.
    
    Database & Visualization: MySQL, SqlServer, Cassandra, HBase, ElasticSearch, 
    Tableau, Kibana, matplotlib, ggplot.
    
    Others: Regular Expression, HTML, CSS, Flask, Git, Docker, Computer Vision - Open CV, 
    Deep Learning understanding.
    
    Company Details: Matelabs - ML platform for business professionals.
    
    Achievements:
    - Deployed automated classification and regression models for predictive analytics
    - Implemented outlier detection algorithms for data quality improvement
    - Deployed time series forecasting models using ARIMA and Prophet
    - Feature engineering and dimensionality reduction for model optimization
    - Created data preprocessing pipeline handling missing values, encoding, and scaling
    """
    
    jd_text = """
    Data Scientist Position
    Location: Multiple Locations
    
    General Summary:
    We are seeking a Data Scientist to join our Analytics team. The successful candidate 
    will have 3+ years of experience with Machine Learning, Predictive Modeling, 
    Statistical Analysis, and Algorithm Development.
    
    Principal Responsibilities:
    - Develops predictive and prescriptive analytic models in support of organizational initiatives
    - Deploys solutions that provide actionable insights and are embedded with application systems
    - Works in a team to drive disruptive innovation and process improvements
    - Builds and extends analytics portfolio supported by robust documentation
    - Works with autonomy to find solutions to complex problems using open source tools
    - Creates and manages project plans and provides updates to leadership
    - Develops relationships with business, IT and clinical leaders across the enterprise
    
    Education and Experience Required:
    - Bachelor's or higher degree in Computer Science, Statistics, Engineering, or related field
    - 3+ years of experience with Machine Learning, Predictive Analytics, Algorithm Development
    - Experience with tools such as Python, R, or other open source statistical tools strongly desired
    - Strong development skills in Python, Java, C++, or related languages
    - Excellent communication and presentation skills
    - Ability to work effectively in team environments
    - Strong understanding of statistical modeling and mathematical optimization
    
    Skills Required:
    - Python, SQL, Statistical Analysis, Machine Learning
    - Data visualization and reporting tools
    - Data preprocessing and feature engineering
    - Predictive modeling and algorithm development
    """
    
    print("\n" + "="*80)
    print("RUNNING FULL RAG PIPELINE")
    print("="*80)
    
    logger.info("\nCV Snippet:")
    logger.info(cv_text[:200] + "...")
    
    logger.info("\nJD Snippet:")
    logger.info(jd_text[:200] + "...")
    
    logger.info("\nRunning RAG pipeline...")
    result = run_rag_pipeline(cv_text, jd_text, top_k=3)
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    logger.info("\nMATCH ASSESSMENT:")
    logger.info("-" * 80)
    logger.info(result.get('assessment_message', 'No assessment'))
    
    logger.info("\nSKILL ANALYSIS:")
    logger.info(f"  Matched Skills ({len(result.get('matched_skills', []))}): {', '.join(result.get('matched_skills', []))}")
    logger.info(f"  Missing Skills ({len(result.get('missing_skills', []))}): {', '.join(result.get('missing_skills', [])[:10])}")
    logger.info(f"  Match Score: {result.get('match_score', 'N/A')}")
    logger.info(f"  Match Level: {result.get('match_level', 'N/A')}")
    
    logger.info(f"\nRetrieved {len(result.get('retrieved_jobs', []))} similar jobs")
    for i, job in enumerate(result.get('retrieved_jobs', [])[:2], 1):
        logger.info(f"  Job {i}: {job.get('content', '')[:120]}...")
    
    logger.info(f"\nRetrieved {len(result.get('retrieved_cvs', []))} similar CVs")
    for i, cv in enumerate(result.get('retrieved_cvs', [])[:2], 1):
        logger.info(f"  CV {i}: {cv.get('content', '')[:120]}...")
    
    logger.info("\nGENERATED COVER LETTER:")
    logger.info("-" * 80)
    cover_letter = result.get('cover_letter', '')
    if cover_letter:
        logger.info(cover_letter)
        logger.info(f"\nCover Letter Stats: {len(cover_letter)} chars, {len(cover_letter.split())} words")
    else:
        logger.warning("No cover letter generated (poor match)!")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline execution complete!")
    logger.info("="*80 + "\n")
