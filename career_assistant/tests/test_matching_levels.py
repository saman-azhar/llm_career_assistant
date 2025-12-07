#!/usr/bin/env python
"""Test all three matching levels: poor, moderate, good."""

import sys
from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
from career_assistant.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    
    # Test Case 1: GOOD MATCH (>80%) - Data Scientist CV with many matching skills
    print("\n" + "="*80)
    print("TEST CASE 1: GOOD MATCH (>80%)")
    print("="*80)
    
    cv_good = """
    Skills: python, machine learning, scikit-learn, tensorflow, pytorch, keras, 
    sql, aws, gcp, nlp, deep learning, classification, statistical analysis, 
    regression, optimization, data visualization, feature engineering
    
    Experience: 6 years as Senior Data Scientist
    - Expertise in tensorflow and pytorch for deep learning
    - Advanced knowledge of python and scikit-learn
    - Proficient in sql and aws
    - Specialized in nlp and classification models
    - Strong in statistical analysis and regression
    """
    
    jd_good = """
    Senior Data Scientist Position
    
    Required skills:
    - python, machine learning, scikit-learn
    - tensorflow, pytorch
    - sql, aws, gcp
    - nlp, deep learning, classification
    """
    
    result_good = run_rag_pipeline(cv_good, jd_good, top_k=1, max_chunks=1)
    
    logger.info("\n" + "="*80)
    logger.info("GOOD MATCH RESULTS")
    logger.info("="*80)
    logger.info(result_good.get('assessment_message', ''))
    logger.info(f"Match Score: {result_good.get('match_score', 'N/A')}")
    logger.info(f"Match Level: {result_good.get('match_level', 'N/A')}")
    if result_good.get('cover_letter'):
        logger.info("\nCover Letter Generated: YES")
    else:
        logger.info("\nCover Letter Generated: NO")
    
    # Test Case 2: MODERATE MATCH (60-80%) - Junior Developer CV + Senior Data Scientist JD
    print("\n" + "="*80)
    print("TEST CASE 2: MODERATE MATCH (60-80%)")
    print("="*80)
    
    cv_moderate = """
    Skills: python, machine learning, deep learning, tensorflow, keras,
    sql, aws, feature engineering, classification, regression, statistical analysis
    
    Experience: 3 years as Data Scientist
    - Working with python and machine learning libraries
    - Experience with tensorflow and keras for deep learning
    - SQL and feature engineering for data preprocessing
    - Classification and regression models in production
    - Statistical analysis and regression modeling
    - AWS cloud deployment experience
    - Missing: pytorch, scikit-learn, xgboost, azure, gcp, nlp, computer vision, reinforcement learning
    """
    
    jd_moderate = """
    Senior Data Scientist Position
    
    Required skills:
    - python, machine learning, deep learning, tensorflow, pytorch, keras
    - sql, aws, azure, gcp
    - scikit-learn, xgboost, reinforcement learning
    - nlp, computer vision
    - Statistical analysis, optimization
    """
    
    result_moderate = run_rag_pipeline(cv_moderate, jd_moderate, top_k=1, max_chunks=1)
    
    logger.info("\n" + "="*80)
    logger.info("MODERATE MATCH RESULTS")
    logger.info("="*80)
    logger.info(result_moderate.get('assessment_message', ''))
    logger.info(f"Match Score: {result_moderate.get('match_score', 'N/A')}")
    logger.info(f"Match Level: {result_moderate.get('match_level', 'N/A')}")
    if result_moderate.get('cover_letter'):
        logger.info("\nCover Letter Generated: YES")
        logger.info("First 300 chars of letter:")
        logger.info(result_moderate.get('cover_letter', '')[:300] + "...")
    else:
        logger.info("\nCover Letter Generated: NO")
    
    # Test Case 3: POOR MATCH (<60%) - Frontend Developer CV + Data Scientist JD
    print("\n" + "="*80)
    print("TEST CASE 3: POOR MATCH (<60%)")
    print("="*80)
    
    cv_poor = """
    Education: B.S Computer Science, 2021
    
    Experience: 3 years as Frontend Developer
    
    Skills: javascript, react, vue.js, html, css, bootstrap, webpack, git,
    responsive design, ui ux, web performance, rest api
    
    Achievements:
    - Built responsive web applications
    - Improved page load time by 40%
    - Worked on e-commerce platform
    - Created reusable UI components
    """
    
    jd_poor = """
    Senior Data Scientist - Machine Learning Focus
    
    Required:
    - 5+ years Data Science and Machine Learning
    - python, machine learning, deep learning, scikit-learn, tensorflow, pytorch, keras
    - sql, aws, azure, gcp, spark, hadoop
    - nlp, computer vision, reinforcement learning, optimization
    - Statistical analysis, hypothesis testing, a b testing, time series
    - MLOps, model deployment, docker, kubernetes, ci cd
    - Team leadership and research publication experience
    """
    
    result_poor = run_rag_pipeline(cv_poor, jd_poor, top_k=1, max_chunks=1)
    
    logger.info("\n" + "="*80)
    logger.info("POOR MATCH RESULTS")
    logger.info("="*80)
    logger.info(result_poor.get('assessment_message', ''))
    logger.info(f"Match Score: {result_poor.get('match_score', 'N/A')}")
    logger.info(f"Match Level: {result_poor.get('match_level', 'N/A')}")
    if result_poor.get('cover_letter'):
        logger.info("\nCover Letter Generated: YES (UNEXPECTED!)")
    else:
        logger.info("\nCover Letter Generated: NO (as expected)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL THREE TEST CASES")
    print("="*80)
    logger.info(f"\nGOOD match:     {result_good.get('match_score', 'N/A')} - Level: {result_good.get('match_level', 'N/A')} - Letter: {'YES' if result_good.get('cover_letter') else 'NO'}")
    logger.info(f"MODERATE match: {result_moderate.get('match_score', 'N/A')} - Level: {result_moderate.get('match_level', 'N/A')} - Letter: {'YES' if result_moderate.get('cover_letter') else 'NO'}")
    logger.info(f"POOR match:     {result_poor.get('match_score', 'N/A')} - Level: {result_poor.get('match_level', 'N/A')} - Letter: {'YES' if result_poor.get('cover_letter') else 'NO'}")
    logger.info("\nExpected:")
    logger.info("  GOOD:     >0.80, level='good', cover_letter='YES'")
    logger.info("  MODERATE: 0.60-0.80, level='moderate', cover_letter='YES'")
    logger.info("  POOR:     <0.60, level='poor', cover_letter='NO'")
    print("="*80 + "\n")
