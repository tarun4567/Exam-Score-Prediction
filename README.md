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
├── main.py (example script) or this code block
├── student.csv
├── reg_model.pkl
├── feature_names.pkl
├── label_encoders.pkl
├── README.md


---

## ▶ How to Run This Project

### 1️⃣ Install Requirements
```bash
pip install pandas numpy scikit-learn
2️⃣ Place Dataset

Ensure student.csv exists in the same folder as the script.

3️⃣ Run the Model Training Script
python main.py


After running, your trained .pkl model files will be generated.

🔧 Future Enhancements

🔹 Add Flask / Django web app UI for real-time prediction
🔹 Hyperparameter tuning (GridSearchCV)
🔹 Feature importance visualization
🔹 Support for additional ML algorithms



This project is created for academic and learning purposes.
Feel free to modify and extend the project for production use.
