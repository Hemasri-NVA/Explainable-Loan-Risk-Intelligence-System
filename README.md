Explainable Loan Risk Intelligence System

An end-to-end Machine Learning project that predicts loan approval status based on an applicant's financial and personal details. The project also focuses on model interpretability by identifying the most influential features affecting the prediction.


Project Overview

Loan approval is an important decision-making process for financial institutions. Traditional evaluation methods can be time-consuming and may vary depending on the evaluator. This project aims to build a machine learning model that can assist in predicting whether a loan application is likely to be approved based on applicant information.

To improve transparency, the project also includes feature importance analysis using SHAP, helping understand the factors that contribute most to the model's predictions.

The trained model is deployed as an interactive web application using Streamlit.

Objectives

- Build a machine learning model for loan approval prediction.
- Compare multiple machine learning algorithms.
- Select the best-performing model based on evaluation metrics.
- Improve prediction transparency using Explainable AI techniques.
- Deploy the trained model through a user-friendly web interface.

Technologies Used

Programming Language : Python 
Machine Learning     : Scikit-learn 
Data Processing      : Pandas, NumPy 
Model Explainability : SHAP 
Visualization        : Matplotlib 
Deployment           : Streamlit 
Model Serialization  : Joblib 


Dataset

The dataset contains information about loan applicants and their financial background.

Input Features

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Assets Value

Target Variable

- Loan Status
  - Approved
  - Rejected

Project Workflow

Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
(Logistic Regression & Random Forest)
      │
      ▼
Model Evaluation
      │
      ▼
Feature Importance & SHAP Analysis
      │
      ▼
Model Deployment using Streamlit
 



Machine Learning Models

The following models were implemented and evaluated:

- Logistic Regression
- Random Forest Classifier

After comparing the results, the Random Forest model was selected for deployment due to its superior performance.



Model Performance

 Model                    Training Accuracy                Testing Accuracy  
 Logistic Regression           80.53%                           82.31% 
 Random Forest                 99.38%                         **97.07%** 



Explainable AI

To understand the model's decision-making process, SHAP (SHapley Additive exPlanations) was used.

The analysis showed that the most influential features include:

- CIBIL Score
- Loan Amount
- Loan Term
- Annual Income

These insights help improve the interpretability of the machine learning model.



Streamlit Application

The project includes an interactive Streamlit application where users can:

- Enter applicant details
- Predict loan approval status
- View prediction confidence
- Explore model information


Project Structure


Explainable-Loan-Risk-Intelligence-System/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── loan_model.pkl
│
├── notebooks/
│   └── loan_pred.ipynb
│
├── data/
│   └── loan_approval_dataset.csv
│
└── images/
    ├── app_home.png
    ├── feature_importance.png
    └── shap_summary.png



How to Run the Project

 Clone the repository

```bash
git clone https://github.com/Hemasri-NVA/Explainable-Loan-Risk-Intelligence-System.git
```

 Navigate to the project directory

```bash
cd Explainable-Loan-Risk-Intelligence-System
```

 Install dependencies

```bash
pip install -r requirements.txt
```

 Launch the Streamlit application

```bash
streamlit run app.py
```
**Note**
This project has been published for educational, learning, and portfolio purposes only.
The source code is shared to demonstrate my implementation and technical skills.
Reproduction, redistribution, or commercial use of this project without prior written permission from the author is not permitted.


Project Screenshots

Added screenshots of the following:

- Home Page
- About the Model
- Loan Approved Prediction
- Loan Rejected Prediction
- Feature Importance Plot
- SHAP Summary Plot


Future Improvements

Some enhancements that can be added in future versions include:

- XGBoost and LightGBM implementation
- Hyperparameter optimization using GridSearchCV
- PDF report generation
- REST API integration
- Real-time database connectivity



Key Learnings

Through this project, I gained practical experience in:

- Data preprocessing
- Feature engineering
- Machine learning model development
- Model evaluation and comparison
- Explainable AI using SHAP
- Building interactive web applications with Streamlit
- Deploying machine learning models


About This Project

This project was developed independently as part of my machine learning portfolio to strengthen my understanding of:

- Data preprocessing
- Classification algorithms
- Model evaluation3
- Explainable AI (SHAP)
- Streamlit deployment
- End-to-end machine learning workflows

It reflects my practical implementation skills in applying machine learning techniques to a real-world loan approval prediction problem.

Author

Annadevara Hemasri

B.Tech – Computer Science and Engineering (AI & ML)

This project was developed as part of my machine learning portfolio to strengthen my understanding of predictive modeling, explainable AI, and deployment of ML applications.



If you found this project interesting, feel free to explore the code and share your feedback.