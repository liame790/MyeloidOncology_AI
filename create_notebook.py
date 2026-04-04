import nbformat as nbf

def create_nb():
    nb = nbf.v4.new_notebook()

    # --- CELL 1: Markdown Introduction ---
    intro_md = """# Clinical Informatics Framework for Myeloid Oncology
## Scalable AI and LLM Integration for Adaptive Trial Management and Automated FDA Compliance

This notebook implements the framework described in the IEEE paper: **A Clinical Informatics Framework for Myeloid Oncology**. 
It demonstrates the end-to-end integration of machine learning models for clinical trial stratification, temporal response forecasting, adverse event prediction, and an LLM-driven Agentic AI layer for FDA-compliant recommendations.

Below, each step of the framework is explained, executed, and its output recorded.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

    # --- CELL 2: Explanation & Code for Imports ---
    md_2 = """### 1. Environment Setup & Data Loading
**Method:** 
We load standard data science libraries (Pandas, Numpy, Scikit-learn, PyTorch) and interpretability tools (SHAP). 
We load a synthetic dataset (`clinical_dataset.csv`) that simulates 300 patient records based on the clinical parameters and molecular profiles (FLT3-ITD, IDH1/2, NPM1) discussed in the MyeloMATCH, Beat AML, and TCGA-LAML cohorts.
**Expected Output:** 
Confirmation of the loaded dataset size and a preview of the first few records showing the structured clinical features."""
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
print(f"Dataset Loaded: {df.shape[0]} records, {df.shape[1]} features")
display(df.head())"""
    nb.cells.append(nbf.v4.new_code_cell(imports_code))

    # --- CELL 3: Explanation & Code for GBDT ---
    md_3 = """### 2. Eligibility Stratification (Gradient Boosted Decision Trees)
**Method:** 
As described in the paper, Gradient Boosted Decision Trees (GBDT) are used for eligibility stratification because of their ability to capture non-linear interactions among heterogeneous clinical parameters (e.g., Blast percentage, FLT3-ITD, ECOG status). We frame eligibility as a classification problem. 
We also apply SHAP (SHapley Additive exPlanations) to provide regulatory transparency, tracing predictions back to specific input features.
**Expected Output:** 
1. AUROC score representing the model's discriminatory capacity.
2. Brier score representing model calibration.
3. A SHAP summary plot showing feature importance, validating that features like `Blast_Pct_Base` and `ANC_Base` drive the clinical decision."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_3))

    gbdt_code = """# Features for Eligibility
elig_features = ['Age', 'ECOG', 'Blast_Pct_Base', 'ANC_Base', 'Plt_Base', 'Hb_Base', 
                 'FLT3_ITD', 'IDH1', 'IDH2', 'NPM1', 'Prior_Therapies', 'Comorbidities']
X_elig = df[elig_features]
y_elig = df['Eligible']

# Encode categorical feature
le = LabelEncoder()
X_elig['Cytogenetic_Risk'] = le.fit_transform(df['Cytogenetic_Risk'])

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_elig, y_elig, test_size=0.2, random_state=seed)

# Train GradientBoostingClassifier
model_gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
model_gbdt.fit(X_train_e, y_train_e)

# Evaluate
y_pred_e = model_gbdt.predict_proba(X_test_e)[:, 1]
auc_e = roc_auc_score(y_test_e, y_pred_e)
brier_e = brier_score_loss(y_test_e, y_pred_e)

print(f"GBDT Eligibility AUROC: {auc_e:.3f}")
print(f"GBDT Calibration (Brier Score): {brier_e:.4f}")

# SHAP Interpretability
explainer = shap.Explainer(model_gbdt)
shap_values = explainer(X_test_e)
shap.summary_plot(shap_values, X_test_e, plot_type="bar", show=False)
plt.title("SHAP Feature Importance for Eligibility Stratification")
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(gbdt_code))

    # --- CELL 4: Explanation & Code for LSTM ---
    md_4 = """### 3. Temporal Response Forecasting (Long Short-Term Memory Networks)
**Method:** 
Long Short-Term Memory (LSTM) networks are employed to predict early treatment outcomes (Day 28 Complete Remission - CR/CRi). LSTMs model time-series data from serial hematological assessments (Days 1, 8, 15), capturing critical clearance kinetics like blast reduction and neutrophil recovery.
**Expected Output:** 
The AUROC score for predicting Day 28 response, demonstrating the model's ability to handle longitudinal clinical vectors."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_4))

    lstm_code = """# Time-series features: ANC (D1, D8, D15), Plt (D1, D8, D15), Blast (D1, D15)
def prepare_lstm_data(df):
    X_seq = []
    for _, row in df.iterrows():
        # Day 1
        d1 = [row['ANC_D1'], row['Plt_D1'], row['Blast_D1']]
        # Day 8 (Blast repeated as missing in simulation)
        d8 = [row['ANC_D8'], row['Plt_D8'], row['Blast_D1']]
        # Day 15
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
    print(f"LSTM Response Prediction AUROC: {auc_l:.3f}")"""
    nb.cells.append(nbf.v4.new_code_cell(lstm_code))

    # --- CELL 5: Explanation & Code for LogReg ---
    md_5 = """### 4. Adverse Event Forecasting (Regularized Logistic Regression)
**Method:** 
To predict high-grade adverse events (like Grade 3-4 Neutropenia), the framework uses L2-Regularized Logistic Regression. This method is selected for its robustness against class imbalances and small-sample instability, as well as its high interpretability required for regulatory safety oversight.
**Expected Output:** 
The AUROC score indicating how well the model predicts severe adverse events based on baseline clinical features."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_5))

    ae_code = """ae_features = ['Age', 'ANC_Base', 'Comorbidities']
X_ae = df[ae_features]
y_ae = df['AE_Grade_3_4']

scaler = StandardScaler()
X_ae_scaled = scaler.fit_transform(X_ae)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_ae_scaled, y_ae, test_size=0.2, random_state=seed)

model_lr = LogisticRegression(penalty='l2', C=1.0)
model_lr.fit(X_train_a, y_train_a)

y_pred_a = model_lr.predict_proba(X_test_a)[:, 1]
auc_a = roc_auc_score(y_test_a, y_pred_a)
print(f"LogReg Adverse Event Risk AUROC: {auc_a:.3f}")"""
    nb.cells.append(nbf.v4.new_code_cell(ae_code))

    # --- CELL 6: Explanation & Code for Agentic AI ---
    md_6 = """### 5. Agentic Automation Layer (LLM & FDA Compliance Mockup)
**Method:** 
The final layer of the framework coordinates the predictive ML outputs with an automated reasoning engine. This class simulates a RAG (Retrieval-Augmented Generation) based LLM Agent. It ingests FDA compliance documents (e.g., 21 CFR Part 11, AI/ML PCCP) and patient-specific ML predictions to output a structured clinical recommendation.
**Expected Output:** 
Initialization of the Agent, confirming it has successfully loaded the regulatory knowledge base."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_6))

    agent_code = """class FDAAgenticAdvisor:
    def __init__(self, compliance_docs):
        self.docs = compliance_docs
        
    def recommend(self, patient_data, ml_results):
        elig = "Eligible" if ml_results['elig_prob'] > 0.5 else "Ineligible"
        resp = "High" if ml_results['resp_prob'] > 0.6 else "Moderate/Low"
        ae = "High Risk" if ml_results['ae_prob'] > 0.4 else "Standard Risk"
        
        # Simulating RAG check for Predetermined Change Control Plans
        pccp_guidance = [d for d in self.docs if d['doc_id'] == 'FDA-AI-ML-GUIDANCE'][0]['content']
        
        recommendation = f"### Agentic Recommendation for {patient_data['PatientID']}\\n"
        recommendation += f"**Clinical Status**: {elig} | **Predicted Response**: {resp} | **Safety Risk**: {ae}\\n\\n"
        
        if elig == "Eligible":
            recommendation += "- **Action**: Proceed with Tier 1 Induction matching.\\n"
            recommendation += f"- **Regulatory Compliance**: SDTM dataset structured. Output aligned with FDA Guidance: *'{pccp_guidance}'*\\n"
        else:
            recommendation += "- **Action**: Refer to Tier Advancement Pathway (TAP).\\n"
            
        if ae == "High Risk":
            recommendation += "- **Alert**: Enhanced monitoring for Grade 3-4 Neutropenia required per safety protocol.\\n"
            
        return recommendation

# Load FDA Compliance Knowledge Base
with open('fda_compliance_docs.json', 'r') as f:
    compliance_data = json.load(f)

advisor = FDAAgenticAdvisor(compliance_data)
print(f"Agentic Advisor Initialized with {len(compliance_data)} regulatory documents.")"""
    nb.cells.append(nbf.v4.new_code_cell(agent_code))

    # --- CELL 7: Explanation & Code for Benchmarking ---
    md_7 = """### 6. System Performance Benchmarking
**Method:** 
Evaluates the synthetic performance against the baseline metrics published in the paper. This validates whether the implemented architecture meets the required discriminatory thresholds for clinical deployment.
**Expected Output:** 
A Pandas DataFrame and a Bar Plot visualizing the implemented model AUROC scores versus the target scores from the study."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_7))

    bench_code = """benchmarks = pd.DataFrame({
    'Model': ['GBDT (Eligibility)', 'LSTM (Response)', 'LogReg (AE Risk)'],
    'Implemented AUROC': [auc_e, auc_l, auc_a],
    'Target Paper AUROC': [0.927, 0.892, 0.846]
})

print("Benchmark Comparison:")
display(benchmarks)

# Visual comparison
benchmarks.plot(x='Model', kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
plt.title("Model Performance Benchmarking vs Paper Baselines")
plt.ylabel("AUROC")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(bench_code))

    # --- CELL 8: Explanation & Code for Inference ---
    md_8 = """### 7. End-to-End Inference Demonstration
**Method:** 
We select a single high-risk patient from the synthetic dataset, run their data through all three ML models (GBDT, LSTM, LogReg), and feed the probabilities into the `FDAAgenticAdvisor`. 
**Expected Output:** 
A formatted Markdown block representing the final system output: an AI-generated clinical assignment recommendation enriched with regulatory compliance audit trails."""
    nb.cells.append(nbf.v4.new_markdown_cell(md_8))

    inf_code = """from IPython.display import Markdown

# Select a patient (e.g., patient index 10)
patient_idx = 10 
p_data = df.iloc[patient_idx].to_dict()

# 1. Eligibility Pred (GBDT)
p_elig_prob = model_gbdt.predict_proba(X_elig.iloc[[patient_idx]])[0, 1]

# 2. Response Pred (LSTM)
p_seq = torch.FloatTensor(X_lstm[patient_idx]).unsqueeze(0)
p_resp_prob = model_lstm(p_seq).detach().numpy()[0, 0]

# 3. AE Risk Pred (LogReg)
p_ae_feat = scaler.transform(X_ae.iloc[[patient_idx]])
p_ae_prob = model_lr.predict_proba(p_ae_feat)[0, 1]

ml_res = {'elig_prob': p_elig_prob, 'resp_prob': p_resp_prob, 'ae_prob': p_ae_prob}

# 4. Generate Agentic Recommendation
rec = advisor.recommend(p_data, ml_res)
Markdown(rec)"""
    nb.cells.append(nbf.v4.new_code_cell(inf_code))

    # Write notebook
    with open('MyeloidOncology_AI_Framework.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_nb()
    print("Notebook 'MyeloidOncology_AI_Framework.ipynb' created with explanations.")
