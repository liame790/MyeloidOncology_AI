# Myeloid Oncology AI & LLM Integration Framework

This repository contains the implementation of the framework described in the IEEE paper: **"A Clinical Informatics Framework for Myeloid Oncology: Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance"**.

## Features
- **Synthetic Clinical Dataset**: ~300 records modeled after MyeloMATCH, Beat AML, and TCGA-LAML cohorts.
- **Eligibility Stratification**: Gradient Boosted Decision Trees (GBDT) for real-time patient-trial matching.
- **Response Prediction**: LSTM (Long Short-Term Memory) networks for Day 28 CR/CRi forecasting using longitudinal hematologic data.
- **Adverse Event Forecasting**: Regularized Logistic Regression for Grade 3-4 Neutropenia risk assessment.
- **Agentic AI Layer**: LLM-driven recommendation engine coordinating ML results with FDA compliance guidelines (21 CFR Part 11, PCCP).

## Repository Structure
- `MyeloidOncology_AI_Framework.ipynb`: Main Jupyter Notebook with implementation and evaluation.
- `clinical_dataset.csv`: Synthetic clinical data generated for this project.
- `fda_compliance_docs.json`: Mocked FDA compliance knowledge base for the RAG/Agentic layer.
- `generate_data.py`: Script used to generate the clinical dataset.
- `create_notebook.py`: Script used to programmatically generate the implementation notebook.
- `A Clinical Informatics Framework for Myeloid Oncology Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance.pdf`: Reference IEEE paper.

## Getting Started
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost torch shap matplotlib seaborn nbformat
   ```
2. Run the notebook:
   ```bash
   jupyter notebook MyeloidOncology_AI_Framework.ipynb
   ```

## Performance Benchmarks
The implementation aims to match or approach the AUROC results reported in the paper:
- **Eligibility (GBDT)**: Target 0.927
- **Response (LSTM)**: Target 0.892
- **AE Risk (LogReg)**: Target 0.846
