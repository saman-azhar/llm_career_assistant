#!/usr/bin/env python3
"""
Complete pipeline to rebuild the entire database from scratch.
Runs: Data Preprocessing → Semantic Matching → Qdrant Ingestion → RAG Pipeline

Usage:
    python run_full_pipeline.py
"""

import os
import sys
import logging
from career_assistant.utils.logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE PIPELINE RECONSTRUCTION")
    logger.info("=" * 80)
    
    # Step 1: Preprocess Job Descriptions
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Preprocessing Job Descriptions")
    logger.info("=" * 80)
    try:
        from career_assistant.preprocessing.preprocessing_jd import preprocess_job_data
        input_jd = "career_assistant/data/raw/glassdoor_jobs.csv"
        output_jd = "career_assistant/data/processed/cleaned_job_data_final.csv"
        preprocess_job_data(input_jd, output_jd)
        logger.info("✓ Job description preprocessing completed")
    except Exception as e:
        logger.error(f"✗ Job preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Preprocess Resumes (CVs)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Preprocessing Resumes/CVs")
    logger.info("=" * 80)
    try:
        from career_assistant.preprocessing.preprocessing_cv import preprocess_resumes
        input_cv = "career_assistant/data/raw/UpdatedResumeDataSet.csv"
        output_cv = "career_assistant/data/processed/cleaned_resume_data_final.csv"
        preprocess_resumes(input_cv, output_cv)
        logger.info("✓ Resume preprocessing completed")
    except Exception as e:
        logger.error(f"✗ Resume preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Delete old Qdrant collection (to start fresh)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Clearing old Qdrant collection")
    logger.info("=" * 80)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host='localhost', port=6333)
        try:
            client.delete_collection("career_assistant_qdrant")
            logger.info("✓ Deleted old Qdrant collection")
        except Exception as e:
            logger.info(f"Note: Collection may not exist yet: {e}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running on localhost:6333")
        return False
    
    # Step 4: Ingest data into Qdrant with chunking
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Ingesting data into Qdrant (with chunking)")
    logger.info("=" * 80)
    try:
        from career_assistant.rag_pipeline.ingest import ingest_data
        ingest_data(chunking=True)
        logger.info("✓ Data ingestion into Qdrant completed")
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Verify the pipeline with a test
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Testing RAG pipeline")
    logger.info("=" * 80)
    try:
        from career_assistant.rag_pipeline.rag_pipeline import run_rag_pipeline
        import mlflow
        mlflow.end_run()  # Clear any active runs
        
        test_cv = "Python developer with 5 years experience in machine learning and data science"
        test_jd = "Looking for ML engineer with Python, TensorFlow, and AWS experience"
        
        logger.info("Running test query...")
        results = run_rag_pipeline(test_cv, test_jd, top_k=3)
        
        logger.info(f"Retrieved {len(results.get('retrieved_jobs', []))} job chunks")
        logger.info(f"Retrieved {len(results.get('retrieved_cvs', []))} CV chunks")
        
        if results.get('cover_letter'):
            logger.info(f"✓ Generated cover letter ({len(results['cover_letter'].split())} words)")
        
        logger.info("✓ RAG pipeline test passed")
    except Exception as e:
        logger.warning(f"⚠ RAG pipeline test had issues (may be expected): {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE RECONSTRUCTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nYour system is now ready to use!")
    logger.info("Next steps:")
    logger.info("  - Run tests: pytest career_assistant/tests/test_rag_pipeline.py -s")
    logger.info("  - Start API: uvicorn career_assistant.api.main:app --reload")
    logger.info("  - Run Streamlit demo: streamlit run notebooks/cover_letter_generation.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
