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
*   **Agentic AI Layer:** Automated regulatory reconciliation using LangChain, ChromaDB, and HuggingFace.
*   **Performance Benchmarking:** Validating synthetic results against the paper's targets.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

    # --- CELL 2: Explanation & Code for Imports ---
    md_2 = """## III. CLINICAL DATA ANALYSIS AND METHODS
### A. Foundation of Data Capture and Continuity

**Methodology:** We synthesize a dataset of 300 patient records mirroring the multivariate distributions found in the MyeloMATCH, Beat AML, and TCGA-LAML cohorts, including clinical parameters (ANC, Platelets, Blast %) and biomarkers (FLT3-ITD, IDH1/2, NPM1).

**Input:** Synthetic data generation parameters.  
**Output:** A localized preview of the synthetic electronic Case Report Forms (eCRFs) dataframe, showing the shape (300 records, 27 features)."""
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

**Methodology:** Gradient Boosted Decision Trees (GBDT) are selected for their robustness in capturing non-linear interactions among heterogeneous predictors. Post-hoc SHAP analysis is deployed for FDA compliance (transparency).

**Input:** Clinical and biomarker features (Age, ECOG, Blast_Pct_Base, FLT3_ITD, etc.).  
**Output:** AUROC and Brier Score metrics quantifying model performance, followed by a SHAP Summary Plot visualizing feature importance (e.g., Blast_Pct_Base driving eligibility)."""
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

**Methodology:** LSTMs forecast early treatment outcomes (Complete Remission by Day 28) by utilizing memory cells to retain long-range temporal dependencies from Day 1, Day 8, and Day 15 CBC panels.

**Input:** Longitudinal vectors of ANC, Platelets, and Blast percentages reshaped into a 3D time-series tensor.  
**Output:** The AUROC score on the test set, evidencing the network's capacity to model temporal recovery trajectories."""
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

**Methodology:** L2-Regularized Logistic Regression is used to anticipate life-threatening hematologic toxicities (Grade 3-4 Neutropenia). Selected for extreme transparency and algorithmic stability under class imbalance.

**Input:** Scaled baseline features (Age, Baseline ANC, Comorbidity Flags).  
**Output:** The AUROC score validating the model's predictive precision for adverse event risk."""
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

**Methodology:** Implements an LLM-driven autonomous agent using LangChain, ChromaDB, and HuggingFace. A RAG architecture ingests FDA policy documents (Predetermined Change Control Plans). We utilize `all-MiniLM-L6-v2` for embeddings and a fast, lightweight open-source LLM (`HuggingFaceTB/SmolLM-135M-Instruct`) to execute locally on CPU without remote API dependencies.

**Input:** Path to `fda_compliance_docs.json` knowledge base.  
**Output:** Initialization logs of the LangChain VectorStore and HuggingFace LLM Pipeline."""
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
        # Using a lightweight fast model to ensure fast CPU inference within the notebook
        print("Loading Local LLM (SmolLM-135M-Instruct) for Agentic Inference...")
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM-135M-Instruct",
            max_new_tokens=150,
            temperature=0.1,
            repetition_penalty=1.1,
            device="cpu"
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # 4. Agentic Prompt Template
        template = \"\"\"System: You are an expert FDA Clinical Informatics Agent. Review the patient data and ML predictions to provide a short clinical recommendation.
Use this FDA guidance context: {context}

User: 
Patient Data: {patient_data}
ML Predictions: {ml_results}
Provide a short recommendation.

Assistant: Based on the clinical data and ML predictions, here is the recommendation:
\"\"\"
        self.prompt = PromptTemplate(template=template, input_variables=["context", "patient_data", "ml_results"])
        
    def recommend(self, patient_data, ml_results):
        # Retrieve context via RAG
        query = "FDA Predetermined Change Control Plan and AI/ML eligibility rules"
        relevant_docs = self.retriever.invoke(query)
        context = chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])
        
        # Format ML Results
        ml_str = f"Eligibility Prob: {ml_results['elig_prob']:.2f}, Response Forecast: {ml_results['resp_prob']:.2f}, AE Risk: {ml_results['ae_prob']:.2f}"
        
        # Execute LLM Chain
        chain = self.prompt | self.llm
        result = chain.invoke({"context": context, "patient_data": str(patient_data), "ml_results": ml_str})
        
        return result

# Initialize the LangChain Agent
advisor = LangChainFDAAdvisor('fda_compliance_docs.json')
print("✅ LangChain Agentic Advisor Online.")"""
    nb.cells.append(nbf.v4.new_code_cell(agent_code))

    # --- CELL 7: Explanation & Code for Benchmarking ---
    md_7 = """## V. ML MODEL PERFORMANCE AND BENCHMARKING

**Methodology:** We map our synthetic model outputs against the target metrics published in Table I of the IEEE manuscript to validate systemic clinical robustness.

**Input:** Computed AUROC scores from GBDT, LSTM, and LogReg.  
**Output:** A comparative DataFrame and a Bar Chart visualization displaying our simulated AUROC versus the empirical results from the paper."""
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

**Methodology:** A singular subject's raw diagnostic array is sequentially passed through the trained GBDT, LSTM, and Logistic Regression engines. The probability matrices are forwarded to the LangChain Agentic Automation Layer, which retrieves FDA compliance rules and generates an action plan.

**Input:** Clinical features of Patient 10 and their calculated predictive probabilities.  
**Output:** A human-readable, regulatory-defensible clinical action plan generated dynamically by the local HuggingFace LLM, demonstrating AI-driven decision support."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_8))

    inf_code = """# Select Subject: Target a patient with high-risk markers for demonstration (Subject 10)
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

print(f"--- Processing Patient {p_data['PatientID']} ---")
print(f"GBDT Eligibility Probability: {p_elig_prob:.2f}")
print(f"LSTM Response Forecast Probability: {p_resp_prob:.2f}")
print(f"LogReg AE Risk Probability: {p_ae_prob:.2f}")
print("\\n--- Agentic FDA Compliance Recommendation ---")

# Agentic Evaluation via LLM
final_report = advisor.recommend(p_data, ml_intelligence)

# Extract only the assistant's response to clean up output
if "Assistant:" in final_report:
    clean_report = final_report.split("Assistant:")[-1].strip()
else:
    clean_report = final_report.strip()

print(clean_report)"""
    nb.cells.append(nbf.v4.new_code_cell(inf_code))

    # Write notebook
    with open('MyeloidOncology_AI_Framework.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_nb()
    print("Notebook 'MyeloidOncology_AI_Framework.ipynb' successfully generated.")
