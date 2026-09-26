# 🫀 CardioGuard AI - Heart Risk Prediction

CardioGuard AI is an educational machine learning project and web application that provides a quick cardiovascular risk assessment based on a patient's vital signs, symptoms, lifestyle, and medical history.

The project consists of a Python-based machine learning pipeline (Logistic Regression) and an interactive web interface built with Streamlit.

## 🚀 Features
- **Risk Assessment:** Calculates cardiovascular risk probability using engineered clinical features.
- **Interactive Dashboard:** Easy-to-use Streamlit web interface for inputting patient data.
- **Health Summary:** Instant feedback on BMI, Blood Pressure stages, and Cholesterol levels.
- **Preventative Advice:** Provides actionable next steps and lifestyle recommendations.

## 📊 Model Evaluation
The heart risk prediction model was trained using Logistic Regression with comprehensive feature engineering. It performs exceptionally well on the test dataset:

- **ROC-AUC Score:** 99.9%
- **Recall (Sensitivity):** 99.4%
- **Precision:** 98.9%
- **F1-Score:** 99.2%

*Note: The dataset used is highly structured/synthetic, which contributes to the near-perfect accuracy.*

## 📁 The `.pkl` Model File
The trained machine learning model is saved as a serialized Python object file named **`heart_disease_model.pkl`**. 
- `train_model.py` generates this file after processing the dataset.
- `app.py` loads this file to make real-time predictions in the browser.

## 🛠️ How to Run Locally & Deploy

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Install Dependencies
Navigate to the project directory and install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Before running the web app, you must train the machine learning model. This will process the dataset and generate the `heart_disease_model.pkl` file.
```bash
python train_model.py
```

### 4. Start the Application
Launch the Streamlit web interface by running:
```bash
streamlit run app.py
```
The app will open automatically in your web browser, typically at `http://localhost:8501`.

### 5. Deployment
To deploy this application to the public internet for free:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app, connect your GitHub repository, select `app.py` as the main file, and click **Deploy**.

---
**Disclaimer:** This is an AI-based screening tool intended for educational and portfolio purposes only. It is not a substitute for a professional medical diagnosis.
