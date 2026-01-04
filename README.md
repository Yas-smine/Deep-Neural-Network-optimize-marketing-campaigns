# 📊 Bank Marketing Campaign – Deep Learning Prediction

## 📌 Project Overview
This project aims to build a **Deep Learning model** to predict the outcome of a bank marketing campaign, specifically whether a client will **subscribe to a term deposit** (`deposit: yes/no`).

The dataset contains demographic, financial, and campaign-related information about clients contacted during marketing campaigns.

---

## 📂 Dataset Description
Each row represents a client. The target variable is:

- **`deposit`**:  
  - `yes` → client subscribed to the deposit  
  - `no` → client did not subscribe  

### Main feature groups:
- **Client information**: age, job, marital status, education  
- **Financial status**: balance, loan, housing  
- **Campaign details**: contact type, duration, campaign count, previous contacts  

---

## 🛠️ Steps Performed

### 1️⃣ Data Import & Initial Analysis
- Loaded the dataset using **Pandas**
- Displayed dataset structure with `df.info()`
- Checked missing values and data types
- Generated descriptive statistics
- Visualized:
  - Numeric feature distributions (histograms)
  - Correlation heatmap
  - Categorical distributions (count plots)

---

### 2️⃣ Data Cleaning & Preprocessing
- Replaced `"unknown"` values with missing values
- Filled missing categorical values using **mode**
- Removed duplicate rows
- Converted the target variable:
  - `deposit → 1 (yes), 0 (no)`
- Encoded categorical variables using **One-Hot Encoding**
- Scaled numeric features using **StandardScaler**
- Split data into:
  - **Training set**
  - **Validation set**
  - **Test set**

---

### 3️⃣ Deep Learning Model
- Built a **fully connected neural network (Dense NN)** using **TensorFlow / Keras**
- Architecture:
  - Input layer
  - Multiple hidden layers with ReLU activation
  - Batch Normalization & Dropout (to reduce overfitting)
  - Output layer with **Sigmoid activation** (binary classification)
- Loss function: `binary_crossentropy`
- Optimizer: `Adam`
- Used **Early Stopping** and **Learning Rate Reduction**
- Applied **class weights** to handle class imbalance

---

### 4️⃣ Model Evaluation
Evaluated the model on the test set using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC score
- Training vs Validation loss and accuracy plots

---

## 📈 Results
The trained deep learning model successfully predicts whether a client will subscribe to a term deposit, achieving strong classification performance and generalization on unseen data.

---

## 📦 Saved Outputs
- Trained model: `bank_deposit_model.h5`
- Preprocessing pipeline: `preprocessor.joblib`

---

## 🚀 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## 🔮 Future Improvements
- Hyperparameter tuning
- Feature importance analysis
- Comparison with classical ML models (Logistic Regression, Random Forest)
- Deployment using **Streamlit**

---


