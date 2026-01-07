# 🎓 Student Exam Performance Prediction – Machine Learning Project

This machine learning project predicts **students' exam performance** based on given academic or demographic factors.  
The model analyzes data, preprocesses categorical features, trains machine learning algorithms, and selects the best-performing model for prediction.

---

## 📌 Project Overview

The goal of this project is to develop a predictive model that estimates a student's **exam score** using historical dataset features.  
Two machine learning regression models were used:

- **Linear Regression** (Baseline Model)
- **Random Forest Regressor** (Advanced Ensemble Model)

Both models were evaluated, and **Random Forest was selected** because it delivered higher prediction accuracy.

---

## 📁 Dataset

The dataset used is `student.csv`.  
The target column is: `exam_score`

### Data Preprocessing Performed:
- Removal of missing values
- Categorical variable encoding using `LabelEncoder`
- Feature/Target split
- Train-test dataset division

---

## 🧠 Machine Learning Models Used

| Model | Description |
|-------|------------|
| Linear Regression | Simple baseline regression |
| Random Forest Regressor | Ensemble-based, better accuracy |

### Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

The Random Forest model outperformed Linear Regression and was saved as the final model.

---

## 💾 Saved Output Files

The following files are generated for deployment or later prediction use:

| File Name | Description |
|----------|-------------|
| `reg_model.pkl` | Trained Random Forest model |
| `feature_names.pkl` | Feature column names used in training |
| `label_encoders.pkl` | Encoders for categorical variables |

These files are essential for running prediction with the same dataset structure.

---

## 🧾 Project Structure

Student-Exam-Prediction/
│
├── main.py                 # Main Python script (prediction logic)
├── student.csv             # Dataset used for prediction
├── reg_model.pkl           # Trained regression model
├── feature_names.pkl       # Feature names used during training
├── label_encoders.pkl      # Label encoders for categorical data
└── README.md               # Project documentation





