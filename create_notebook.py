import nbformat as nbf

def create_nb():
    nb = nbf.v4.new_notebook()

    # --- CELL 1: Markdown Introduction ---
    intro_md = """<div align="center">
    
# A Clinical Informatics Framework for Myeloid Oncology
### Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance

[![IEEE Paper](https://img.shields.io/badge/IEEE-11378528-blue.svg)](https://ieeexplore.ieee.org/document/11378528)

</div>

**Authors:** Senthilkumar Vijayakumar, Shaunak Pai Kane, Selvavaani Senthilkumar, Dr. Parameshwari Vaiayapuri MBBS, Filious Louis  
**Citation:** S. Vijayakumar, S. P. Kane, S. Senthilkumar, P. Vaiayapuri, and F. Louis, "A Clinical Informatics Framework for Myeloid Oncology: Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance," IEEE, 2026. Available: https://ieeexplore.ieee.org/document/11378528

---

## Executive Summary
This notebook implements the end-to-end clinical informatics framework described in the aforementioned IEEE paper. It demonstrates the seamless integration of predictive machine learning models and large language models (LLMs) to enhance precision-guided therapy in myeloid malignancies (e.g., AML, MDS). 

The implementation covers:
*   **Clinical Data Analysis:** Foundation of data capture and continuity via synthetic cohorts.
*   **Machine Learning Design:** GBDT for stratification, LSTM for temporal forecasting, and Logistic Regression for adverse event modeling.
*   **Agentic AI Layer:** Automated regulatory reconciliation mapped to FDA 21 CFR Part 11.
*   **Performance Benchmarking:** Validating synthetic results against the paper's targets.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

    # --- CELL 2: Explanation & Code for Imports ---
    md_2 = """## III. CLINICAL DATA ANALYSIS AND METHODS
### A. Foundation of Data Capture and Continuity

**Methodology:**
*   **Data Emulation:** We synthesize a dataset of 300 patient records mirroring the multivariate distributions found in the MyeloMATCH, Beat AML, and TCGA-LAML cohorts.
*   **Clinical Parameters:** Encompasses dynamic variables such as Absolute Neutrophil Count (ANC), Platelet Count, Hemoglobin (Hb), and Bone Marrow Blast percentage.
*   **Biomarker Mapping:** Incorporates critical molecular stratifiers: FLT3-ITD, IDH1, IDH2, and NPM1 mutations alongside overarching cytogenetic risk categories.

**Expected Output:**
*   Verification of the data frame dimensions (300 records, 27 features).
*   A localized preview of the synthetic electronic Case Report Forms (eCRFs)."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_2))

    imports_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import json
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load Data
df = pd.read_csv('clinical_dataset.csv')
print(f"Synthetic Clinical Dataset Loaded: {df.shape[0]} patient records, {df.shape[1]} clinical/biomarker features")
display(df.head())"""
    nb.cells.append(nbf.v4.new_code_cell(imports_code))

    # --- CELL 3: Explanation & Code for GBDT ---
    md_3 = """## IV. MACHINE LEARNING MODEL DESIGN
### A. Gradient Boosted Decision Trees (GBDT) for Eligibility Stratification

**Methodology:**
*   **Objective:** Assign patients to protocol-eligible treatment arms (or the Tier Advancement Pathway) based on real-time molecular and clinical presentation.
*   **Algorithm:** Gradient Boosted Decision Trees (GBDT) are selected for their robustness in capturing non-linear interactions among heterogeneous predictors (e.g., age, cytogenetic risk, and mutational burden).
*   **Interpretability:** Post-hoc SHAP (SHapley Additive exPlanations) analysis is deployed. As mandated by FDA guidance for AI/ML-enabled Software as a Medical Device (SaMD), SHAP ensures clinical plausibility by rendering the variable attribution mathematically transparent.

**Expected Output:**
*   **AUROC & Brier Score:** Quantification of discrimination capacity and calibration alignment.
*   **SHAP Summary Plot:** Visual confirmation that features like `Blast_Pct_Base`, `ANC_Base`, and `FLT3_ITD` govern eligibility predictions."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_3))

    gbdt_code = """# Features for Eligibility Stratification
elig_features = ['Age', 'ECOG', 'Blast_Pct_Base', 'ANC_Base', 'Plt_Base', 'Hb_Base', 
                 'FLT3_ITD', 'IDH1', 'IDH2', 'NPM1', 'Prior_Therapies', 'Comorbidities']
X_elig = df[elig_features]
y_elig = df['Eligible']

# Encode categorical variables (e.g., Cytogenetic Risk)
le = LabelEncoder()
X_elig['Cytogenetic_Risk'] = le.fit_transform(df['Cytogenetic_Risk'])

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_elig, y_elig, test_size=0.2, random_state=seed)

# Train GradientBoostingClassifier
model_gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
model_gbdt.fit(X_train_e, y_train_e)

# Evaluate Discrimination and Calibration
y_pred_e = model_gbdt.predict_proba(X_test_e)[:, 1]
auc_e = roc_auc_score(y_test_e, y_pred_e)
brier_e = brier_score_loss(y_test_e, y_pred_e)

print(f"GBDT Eligibility AUROC: {auc_e:.3f}")
print(f"GBDT Calibration Error (Brier Score): {brier_e:.4f}")

# SHAP Interpretability
explainer = shap.Explainer(model_gbdt)
shap_values = explainer(X_test_e)
shap.summary_plot(shap_values, X_test_e, plot_type="bar", show=False)
plt.title("SHAP Feature Importance: Eligibility Stratification")
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(gbdt_code))

    # --- CELL 4: Explanation & Code for LSTM ---
    md_4 = """### B. Long Short-Term Memory Networks (LSTM) for Temporal Response Prediction

**Methodology:**
*   **Objective:** Forecast early treatment outcomes, specifically the likelihood of achieving Complete Remission (CR/CRi) by Day 28.
*   **Data Engineering:** Longitudinal vectors derived from serial Complete Blood Count (CBC) panels and bone marrow assessments captured at Day 1, Day 8, and Day 15.
*   **Architecture:** LSTMs utilize memory cells and gating mechanisms to retain long-range temporal dependencies. This captures non-linear clearance kinetics (e.g., blast reduction rates and hematopoietic rebound).

**Expected Output:**
*   Convergence of the BCE Loss over training epochs.
*   The mean AUROC score on the test set, evidencing the network's capacity to model temporal recovery trajectories."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_4))

    lstm_code = """# Extract Temporal Sequence: Day 1, Day 8, Day 15 (ANC, Plt, Blast)
def prepare_lstm_data(df):
    X_seq = []
    for _, row in df.iterrows():
        d1 = [row['ANC_D1'], row['Plt_D1'], row['Blast_D1']]
        d8 = [row['ANC_D8'], row['Plt_D8'], row['Blast_D1']] # Blast repeated D8
        d15 = [row['ANC_D15'], row['Plt_D15'], row['Blast_D15']]
        X_seq.append([d1, d8, d15])
    return np.array(X_seq), df['Response_D28'].values

X_lstm, y_lstm = prepare_lstm_data(df)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=seed)

X_train_l = torch.FloatTensor(X_train_l)
X_test_l = torch.FloatTensor(X_test_l)
y_train_l = torch.FloatTensor(y_train_l).view(-1, 1)
y_test_l = torch.FloatTensor(y_test_l).view(-1, 1)

class MyeloidLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out)

model_lstm = MyeloidLSTM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.01)

# Training Loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model_lstm(X_train_l)
    loss = criterion(outputs, y_train_l)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_l = model_lstm(X_test_l).numpy()
    auc_l = roc_auc_score(y_test_l.numpy(), y_pred_l)
    print(f"LSTM Temporal Response Forecasting AUROC: {auc_l:.3f}")"""
    nb.cells.append(nbf.v4.new_code_cell(lstm_code))

    # --- CELL 5: Explanation & Code for LogReg ---
    md_5 = """### C. Regularized Logistic Regression for Adverse Event Forecasting

**Methodology:**
*   **Objective:** Anticipate life-threatening hematologic toxicities, particularly early-onset Grade 3-4 Neutropenia.
*   **Rationale:** The framework deliberately selects L2-Regularized Logistic Regression over black-box methods due to its extreme transparency, algorithmic stability under small sample sizes, and established acceptance in regulatory review processes (e.g., Data Safety Monitoring Boards).
*   **Class Imbalance:** Evaluates baseline features (Age, Baseline ANC, Comorbidity Flags) against sparse outcome data.

**Expected Output:**
*   The AUROC score validating the model's predictive precision in capturing high-risk patients before the onset of severe clinical cytopenia."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_5))

    ae_code = """ae_features = ['Age', 'ANC_Base', 'Comorbidities']
X_ae = df[ae_features]
y_ae = df['AE_Grade_3_4']

scaler = StandardScaler()
X_ae_scaled = scaler.fit_transform(X_ae)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_ae_scaled, y_ae, test_size=0.2, random_state=seed)

# L2-Regularized Logistic Regression
model_lr = LogisticRegression(penalty='l2', C=1.0)
model_lr.fit(X_train_a, y_train_a)

y_pred_a = model_lr.predict_proba(X_test_a)[:, 1]
auc_a = roc_auc_score(y_test_a, y_pred_a)
print(f"Regularized LogReg Adverse Event Risk AUROC: {auc_a:.3f}")"""
    nb.cells.append(nbf.v4.new_code_cell(ae_code))

    # --- CELL 6: Explanation & Code for Agentic AI ---
    md_6 = """## II. E. Regulatory Documentation and Submission Engine
### Agentic Automation Layer (LangChain & HuggingFace Integration)

**Methodology:**
*   **Objective:** Accelerate regulatory preparedness and reduce administrative workload by linking predictive outputs directly to compliance standards.
*   **Architecture:** Implements an LLM-driven autonomous agent using **LangChain**, **ChromaDB**, and **HuggingFace**. A Retrieval-Augmented Generation (RAG) architecture ingests internal clinical assignments and cross-references them with loaded FDA policy documents (e.g., Predetermined Change Control Plans).
*   **Models:** Utilizes `all-MiniLM-L6-v2` for semantic embeddings and a local open-source LLM pipeline (`TinyLlama-1.1B-Chat`) to autonomously generate structured recommendations without relying on external API keys.

**Expected Output:**
*   Initialization of the LangChain VectorStore.
*   Loading of the HuggingFace LLM Pipeline."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_6))

    agent_code = """from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import json
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class LangChainFDAAdvisor:
    def __init__(self, docs_path):
        # 1. Load FDA Knowledge Base
        with open(docs_path, 'r') as f:
            docs = json.load(f)
            
        lc_docs = [Document(page_content=d['content'], metadata={"title": d['title']}) for d in docs]
        
        # 2. Setup Vector Store (Embeddings)
        print("Loading HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(lc_docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 3. Setup Local HuggingFace LLM Pipeline
        print("Loading Local LLM (TinyLlama-1.1B) for Agentic Inference...")
        pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=250,
            temperature=0.1,
            repetition_penalty=1.1,
            device="cpu"  # Running on CPU for broad compatibility
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # 4. Agentic Prompt Template
        template = \"\"\"<|system|>
You are an expert FDA Clinical Informatics Agent. You review patient clinical data, ML predictions, and FDA compliance guidelines to make clear, actionable clinical trial recommendations.
Use the following FDA guidance as your regulatory framework:
{context}
</s>
<|user|>
Patient Data: {patient_data}
ML Predictions: {ml_results}

Based on the above, provide a short, structured clinical recommendation. Ensure you state how it complies with the provided FDA guidance.
</s>
<|assistant|>
\"\"\"
        self.prompt = PromptTemplate(template=template, input_variables=["context", "patient_data", "ml_results"])
        
    def recommend(self, patient_data, ml_results):
        # Retrieve context via RAG
        query = "FDA Predetermined Change Control Plan and AI/ML eligibility rules"
        relevant_docs = self.retriever.invoke(query)
        # Using chr(10) instead of literal slash-n to avoid code generation parsing bugs
        context = chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])
        
        # Format ML Results
        ml_str = f"Eligibility Prob: {ml_results['elig_prob']:.2f}, Response Forecast: {ml_results['resp_prob']:.2f}, AE Risk: {ml_results['ae_prob']:.2f}"
        
        # Execute LLM Chain
        chain = self.prompt | self.llm
        result = chain.invoke({"context": context, "patient_data": str(patient_data), "ml_results": ml_str})
        
        # Parse output
        if "<|assistant|>" in result:
            output = result.split("<|assistant|>")[-1].strip()
        else:
            output = result.strip()
            
        return output

# Initialize the LangChain Agent
advisor = LangChainFDAAdvisor('fda_compliance_docs.json')
print("✅ LangChain Agentic Advisor Online.")"""
    nb.cells.append(nbf.v4.new_code_cell(agent_code))

    # --- CELL 7: Explanation & Code for Benchmarking ---
    md_7 = """## V. ML MODEL PERFORMANCE AND BENCHMARKING

**Methodology:**
*   Evaluating machine learning models in a precision-targeted clinical trial requires stringent benchmarking against published baselines.
*   We map our synthetic model outputs against the target metrics published in Table I of the IEEE manuscript.

**Expected Output:**
*   A comparative DataFrame displaying our simulated AUROC versus the paper's empirical results.
*   A localized graphical visualization emphasizing systemic clinical robustness."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_7))

    bench_code = """benchmarks = pd.DataFrame({
    'Clinical Task': ['Eligibility (GBDT)', 'Response (LSTM)', 'AE Risk (LogReg)'],
    'Synthetic Implementation AUROC': [auc_e, auc_l, auc_a],
    'Target Paper AUROC': [0.927, 0.892, 0.846]
})

display(benchmarks)

# Bar chart visualization
ax = benchmarks.plot(x='Clinical Task', kind='bar', figsize=(10, 6), color=['#2c3e50', '#18bc9c'])
plt.title("Framework Performance: Implementation vs Baseline IEEE Targets", fontsize=14, fontweight='bold')
plt.ylabel("Area Under the ROC Curve (AUROC)", fontsize=12)
plt.xlabel("")
plt.xticks(rotation=0, fontsize=11)
plt.ylim(0, 1.1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(bench_code))

    # --- CELL 8: Explanation & Code for Inference ---
    md_8 = """## VI. RESULTS AND DISCUSSION: END-TO-END INFERENCE

**Methodology:**
*   **Data Pipeline Execution:** A singular subject's raw diagnostic array is sequentially passed through the trained GBDT, LSTM, and Logistic Regression engines.
*   **Decision Support Finalization:** The probability matrices are forwarded to the Agentic Automation Layer. 
*   **Traceability:** This outputs a human-readable, regulatory-defensible clinical action plan, demonstrating how the system operates at the intersection of oncology and informatics.

**Expected Output:**
*   A comprehensive, clinically formatted Markdown summary indicating the subject's status, predicted trajectories, and safety protocols."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_8))

    inf_code = """from IPython.display import Markdown

# Select Subject: Target a patient with high-risk markers for demonstration (Subject 10)
subject_idx = 10 
p_data = df.iloc[subject_idx].to_dict()

# 1. Pipeline: Stratification (GBDT)
p_elig_prob = model_gbdt.predict_proba(X_elig.iloc[[subject_idx]])[0, 1]

# 2. Pipeline: Response Forecast (LSTM)
p_seq = torch.FloatTensor(X_lstm[subject_idx]).unsqueeze(0)
p_resp_prob = model_lstm(p_seq).detach().numpy()[0, 0]

# 3. Pipeline: Safety Oversight (LogReg)
p_ae_feat = scaler.transform(X_ae.iloc[[subject_idx]])
p_ae_prob = model_lr.predict_proba(p_ae_feat)[0, 1]

# Consolidate Intelligence
ml_intelligence = {'elig_prob': p_elig_prob, 'resp_prob': p_resp_prob, 'ae_prob': p_ae_prob}

# Agentic Evaluation
final_report = advisor.recommend(p_data, ml_intelligence)
Markdown(final_report)"""
    nb.cells.append(nbf.v4.new_code_cell(inf_code))

    # Write notebook
    with open('MyeloidOncology_AI_Framework.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_nb()
    print("Notebook 'MyeloidOncology_AI_Framework.ipynb' successfully generated with IEEE structuring.")
