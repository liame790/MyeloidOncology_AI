<div align="center">
  
# A Clinical Informatics Framework for Myeloid Oncology
### Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance

[![IEEE Paper](https://img.shields.io/badge/IEEE-11378528-blue.svg)](https://ieeexplore.ieee.org/document/11378528)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-f7931e)](https://scikit-learn.org/)
[![FDA Compliance](https://img.shields.io/badge/FDA-21%20CFR%20Part%2011-005288)](https://www.fda.gov/)

</div>

## 📖 Project Overview

This repository provides the official implementation of the framework proposed in the IEEE paper: **"A Clinical Informatics Framework for Myeloid Oncology: Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance"**.

The advancement of treatment strategies in myeloid malignancies (such as Acute Myeloid Leukemia - AML) relies heavily on precision oncology, dynamic biomarker profiling, and adaptive trial designs. This project introduces an **Artificial Intelligence-enabled clinical trial framework** designed to navigate the complexities of molecularly defined eligibility criteria, predictive safety monitoring, and the rigorous demands of accelerated FDA regulatory submissions.

### 🔑 Key SEO Tags & Keywords
`Precision Oncology`, `Myeloid Malignancies`, `Acute Myeloid Leukemia (AML)`, `Adaptive Clinical Trials`, `Machine Learning in Healthcare`, `Electronic Data Capture (EDC)`, `Regulatory Automation`, `Large Language Models (LLMs)`, `FDA 21 CFR Part 11`, `MyeloMATCH`.

---

## ✨ Technical Features & Architecture

This repository translates the conceptual clinical informatics architecture into a deployable, code-driven framework:

- **Data Acquisition & Harmonization (Synthetic):** Generation of high-dimensional clinical and biomarker datasets (FLT3, IDH1/2, NPM1) mimicking the Beat AML and TCGA-LAML open-source cohorts.
- **Biomarker Interpretation & Protocol Rule Engine:** Automated mapping of patient morphology and cytogenetics against protocol-specific thresholds.
- **Patient Stratification (GBDT):** Gradient Boosted Decision Trees deployed for real-time, explainable (SHAP-integrated) trial matching and eligibility assignment.
- **Temporal Response Prediction (LSTM):** Long Short-Term Memory networks designed to forecast Day 28 early treatment response (CR/CRi) using sequential hematological clearance kinetics.
- **Adverse Event Modeling (Regularized Logistic Regression):** Robust, clinically interpretable risk forecasting for Grade 3-4 hematologic toxicities (e.g., Neutropenia).
- **Agentic AI & Regulatory Submission Engine:** An LLM-driven automation layer that maps ML outputs to FDA Predetermined Change Control Plans (PCCP) and 21 CFR Part 11 compliance standards, outputting structured clinical recommendations.

---

## 🔬 Implementation Details (Jupyter Notebook)

The core implementation is housed in `MyeloidOncology_AI_Framework.ipynb`. The notebook is meticulously documented, mapping directly to the paper's headings:
1. **Clinical Data Analysis and Methods** (Data Harmonization)
2. **Machine Learning Model Design**
   - *Eligibility Stratification* (GBDT)
   - *Temporal Response Prediction* (LSTM)
   - *Adverse Event Forecasting* (LogReg)
3. **Agentic Automation Layer** (LLM & FDA Compliance)
4. **ML Model Performance and Benchmarking**

---

## 📝 Citation

If you utilize this framework or code in your research, please cite the original IEEE paper:

> **IEEE Format:**
> S. Vijayakumar, S. P. Kane, S. Senthilkumar, P. Vaiayapuri, and F. Louis, "A Clinical Informatics Framework for Myeloid Oncology: Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance," *IEEE*, 2026. Available: [https://ieeexplore.ieee.org/document/11378528](https://ieeexplore.ieee.org/document/11378528)

**Authors:**
- Senthilkumar Vijayakumar (IEEE Senior Member)
- Shaunak Pai Kane (IEEE Member)
- Selvavaani Senthilkumar
- Dr. Parameshwari Vaiayapuri, MBBS
- Filious Louis (IEEE Senior Member)

---

## 🚀 Getting Started

### Prerequisites
Install the required Python dependencies:
```bash
pip install pandas numpy scikit-learn xgboost torch shap matplotlib seaborn jupyter
```

### Running the Framework
1. Clone the repository.
2. Run the synthetic data generator (if you wish to regenerate the cohort): `python3 generate_data.py`
3. Launch the Jupyter Notebook to explore the models, SHAP explanations, and Agentic AI outputs:
   ```bash
   jupyter notebook MyeloidOncology_AI_Framework.ipynb
   ```
