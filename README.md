### Advanced Customer Churn Prediction System

* * *
### 📌 Executive Summary

This project delivers a data-driven customer retention system for telecom analytics. By replacing "blind" mass-marketing strategies with targeted machine learning interventions, the system captures at-risk customers early. The final model achieves **80% Recall**, effectively identifying 4 out of 5 at-risk customers and generating **$47,960 in projected net savings** per test cycle.

* * *

### 🛠️ Tech Stack & Key Skills

*   **Programming Language:** Python
*   **Machine Learning & Frameworks:** Scikit-Learn (Advanced Algorithms, GridSearch Tuning)
*   **Data Imbalance Resolution:** SMOTE (Synthetic Minority Over-sampling Technique)
*   **Deployment & Interface:** Streamlit Dashboard
*   **Core Competencies:** Data Leakage Prevention, Feature Engineering, Business Impact & ROI Calculator Analysis

* * *

### 📊 Data Pipeline & Feature Engineering

1.  **Data Cleaning:** Resolved crucial data integrity discrepancies by casting structural variables (e.g., parsing `TotalCharges` from string values to floats).
2.  **Class Imbalance:** Solved a severe 73% Loyal / 27% Churn dataset imbalance using training-stage SMOTE.
3.  **Leakage-Proof Pipeline:** Deployed an end-to-end `ColumnTransformer` to safely isolate data splits, normalize continuous numerical features (such as customer tenure), and apply One-Hot Encoding to categorical telecom services.

* * *

### 📉 Machine Learning Model Benchmarking

Models were explicitly evaluated on **Recall** to minimize missed churners (False Negatives), which carry the highest business penalty (-$400 per customer).

Model Variant

Test Accuracy

Test Recall

F1-Score

Status

**Logistic Regression (Baseline)**

**73.7%**

**80.0%**

**0.617**

**🏆 Selected Winner**

Support Vector Machine (Baseline)

75.4%

75.4%

0.619

Benchmarked

Random Forest (Tuned via Grid)

77.3%

64.7%

0.602

Overfitted

Random Forest (Baseline)

77.7%

57.5%

0.578

Overfitted

### Model Performance Visualization

*Note: While tree-based models scored higher in baseline accuracy, Logistic Regression provided superior class detection boundaries for active churn flight risks.*

* * *

### 💡 Key Drivers & Business Insights

Analysis of the winning Logistic Regression weights revealed the following core risk indicators:

*   **Month-to-Month Contracts:** The single largest catalyst for customer flight risk.
*   **Fiber Optic Internet:** Users churn faster than DSL clients, indicating underlying price-performance or service friction.
*   **Early-Stage Tenure:** Customer churn drops dramatically after the 12-month milestone, highlighting early lifecycle vulnerability.

### Feature Importance Visualization

* * *

### 💰 Financial ROI Calculator

*   **Do Nothing Baseline:** Results in a raw operational deficit of **$149,600** in unmitigated churn losses.
*   **Algorithmic Strategy:** Lowers financial impact losses to **$101,640** by offering optimized retention incentives ($40 cost per alert).
*   **Net Value Generated:** Yields **$47,960** in pure cycle savings.

<img width="966" height="516" alt="image" src="https://github.com/user-attachments/assets/4d098937-504c-4d9a-a123-afbc17968c68" />


### Financial Savings Breakdown

* * *

### 🚀 Strategic Actions & Recommendations

1.  **Targeted Migration:** Immediately deploy $40 financial incentives to transition high-risk Month-to-Month Fiber Optic users into stable 1-Year contracts.
2.  **Dashboard Deployment:** Launch the active **Streamlit Dashboard** to customer service teams for real-time risk evaluation during inbound support calls.
3.  **Infrastructure Review:** Initiate an operational investigation into Fiber Optic infrastructure and service quality issues to address the core driver of churn.

* * *

### 💻 Technical Usage Instructions

### 1\. Installation

Clone the repository and install the project dependencies inside a clean virtual environment:

bash

    # Clone the repository
    git clone https://github.com
    cd YOUR_REPOSITORY_NAME
    
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    
    # Install requirements
    pip install -r requirements.txt
    

Use code with caution.

### 2\. Run the Streamlit Application

Launch the local web dashboard to interactively test customer churn risk scoring profiles:

bash

    streamlit run app.py
    

Use code with caution.

* * *

### 👥 Connect & Links

*   [Project Presentation](https://docs.google.com/presentation/d/1eNzVFkmCByDzb5TGQOjK-hxhn-hbbx_Bw3QHVOavu60/edit?slide=id.p#slide=id.p)
*   [LinkedIn Profile](https://www.linkedin.com/in/victor-oduronbi-62b22132b/)
