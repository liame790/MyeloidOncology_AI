import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(n_records=300):
    data = []
    for i in range(n_records):
        patient_id = f"PAT-{1000+i}"
        age = np.random.randint(18, 85)
        gender = random.choice(['M', 'F'])
        
        # Clinical parameters (Eligibility focused)
        ecog = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
        blast_pct = np.random.uniform(5, 95)
        anc_base = np.random.uniform(0.1, 5.0)
        plt_base = np.random.uniform(10, 300)
        hb_base = np.random.uniform(7, 15)
        
        # Biomarkers
        flt3_itd = np.random.choice([0, 1], p=[0.7, 0.3])
        idh1 = np.random.choice([0, 1], p=[0.85, 0.15])
        idh2 = np.random.choice([0, 1], p=[0.8, 0.2])
        npm1 = np.random.choice([0, 1], p=[0.6, 0.4])
        
        cytogenetic_risk = np.random.choice(['Low', 'Intermediate', 'High'], p=[0.2, 0.5, 0.3])
        prior_therapies = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        comorbidities = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Rule-based Eligibility (from paper)
        # ECOG <= 2, Blast >= 20, ANC >= 0.5, Plt >= 50
        is_eligible = 1 if (ecog <= 2 and blast_pct >= 20 and anc_base >= 0.5 and plt_base >= 50 and comorbidities == 0) else 0
        
        # Tiers (Simplified logic)
        if prior_therapies == 0:
            tier = 1 # Induction
        elif prior_therapies > 0:
            tier = 2 # Refractory
        else:
            tier = 3 # Post-Remission (if we had state info)
            
        # Time Series for LSTM (Days 1, 8, 15)
        # Simulate decline in blast and recovery/fluctuation in ANC/Plt
        anc_d1 = anc_base
        anc_d8 = anc_base * np.random.uniform(0.5, 1.2)
        anc_d15 = anc_d8 * np.random.uniform(0.8, 2.0)
        
        plt_d1 = plt_base
        plt_d8 = plt_base * np.random.uniform(0.4, 1.1)
        plt_d15 = plt_d8 * np.random.uniform(0.7, 1.8)
        
        blast_d1 = blast_pct
        blast_d15 = blast_pct * np.random.uniform(0.1, 0.8) # Treatment effect
        
        # Target: Response D28 (CR/CRi)
        # Probability depends on blast reduction, cytogenetics, and mutations
        # High risk or FLT3+ usually have lower CR rates in some induction protocols
        response_prob = 0.7 if is_eligible else 0.1
        if cytogenetic_risk == 'High': response_prob -= 0.2
        if flt3_itd == 1: response_prob -= 0.1
        if blast_d15 < 5: response_prob += 0.2
        
        response_d28 = 1 if np.random.random() < np.clip(response_prob, 0, 1) else 0
        
        # Target: Adverse Event (Grade 3-4 Neutropenia)
        # Probability depends on baseline ANC, Age, and Comorbidities
        ae_prob = 0.1
        if anc_base < 1.0: ae_prob += 0.3
        if age > 65: ae_prob += 0.2
        if comorbidities == 1: ae_prob += 0.2
        
        ae_grade_3_4 = 1 if np.random.random() < np.clip(ae_prob, 0, 1) else 0
        
        data.append({
            'PatientID': patient_id, 'Age': age, 'Gender': gender, 'ECOG': ecog,
            'Blast_Pct_Base': blast_pct, 'ANC_Base': anc_base, 'Plt_Base': plt_base, 'Hb_Base': hb_base,
            'FLT3_ITD': flt3_itd, 'IDH1': idh1, 'IDH2': idh2, 'NPM1': npm1,
            'Cytogenetic_Risk': cytogenetic_risk, 'Prior_Therapies': prior_therapies,
            'Comorbidities': comorbidities, 'Eligible': is_eligible, 'Tier': tier,
            'ANC_D1': anc_d1, 'ANC_D8': anc_d8, 'ANC_D15': anc_d15,
            'Plt_D1': plt_d1, 'Plt_D8': plt_d8, 'Plt_D15': plt_d15,
            'Blast_D1': blast_d1, 'Blast_D15': blast_d15,
            'Response_D28': response_d28, 'AE_Grade_3_4': ae_grade_3_4
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_data(300)
    df.to_csv('clinical_dataset.csv', index=False)
    print(f"Generated {len(df)} records in clinical_dataset.csv")
