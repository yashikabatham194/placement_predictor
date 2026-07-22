# Student Placement Prediction System

An end-to-end machine learning project that predicts whether a student is likely to be **Placed** or **Not Placed**, based on academic performance, skill scores, and experience indicators.

---

## Problem Statement

Colleges currently assess placement readiness informally, making it hard to identify at-risk students early or allocate placement-training resources efficiently. This project builds a data-driven system to predict placement outcomes objectively.

## Objective

- Classify students as Placed / Not Placed using their academic and skill profile.
- Identify the factors that most strongly influence placement (e.g. CGPA, technical score, internships, backlogs).
- Provide a foundation for an interactive Streamlit app for instant predictions.

## Dataset

- **File:** `student_placement_prediction_dataset.csv`
- **Size:** 5,000 student records × 23 columns
- **Target:** `Placement_Status` (Placed / Not Placed)
- **Features:** SSC/HSC/Degree percentages, CGPA, Aptitude/Communication/Technical scores, Internships, Projects, Certifications, Backlogs, Work Experience, and more.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.769 | 0.786 | 0.852 | **0.818** |
| Random Forest | 0.758 | 0.787 | 0.826 | 0.806 |
| Decision Tree | 0.698 | 0.745 | 0.765 | 0.755 |

**Final model:** Logistic Regression (best F1 score); hyperparameter tuning and SMOTE rebalancing did not improve on the baseline.
**Top predictive features:** Academic_Avg, Skill_Index, Aptitude_Test_Score, Degree_Percentage, Technical_Score, CGPA

## Technologies used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Developed By:
Yashika Batham

MBA — Business Analytics
