# 🧠 AI Personal Health Tracker
### *Multi-Modal Heart Disease Risk Prediction System*

---

## 📺 Project Demo
Check out the full walkthrough of the application on my YouTube channel, **Mubeen Tech**:

[![AI Personal Health Tracker Demo](https://img.youtube.com/vi/DGfnnwHmDkg/maxresdefault.jpg)](https://youtu.be/DGfnnwHmDkg)

*Click the image above to watch the demo on YouTube.*

---

## 📌 Project Overview
The **AI Personal Health Tracker** is an end-to-end artificial intelligence application designed to assess heart disease risk by combining multiple types of patient data into one unified prediction system.

Instead of relying on a single data source, this project integrates:
* **Clinical & Demographic Data** (Tabular)
* **ECG Time-Series Signals**
* **Chest X-ray Images**

Each data type is processed using a specialized deep learning model, and their outputs are fused to produce a final, patient-level risk prediction. This simulates a real-world clinical decision support system.

## 🎯 Project Goals
* **Multi-Modal Integration:** Apply ANN, CNN, and LSTM in a single unified pipeline.
* **End-to-End Development:** Practice full-stack ML system design.
* **Interpretable Results:** Deliver clear risk levels with actionable health recommendations.
* **Professional Portfolio:** Create a job-ready, high-impact AI project.

## 🚀 Key Features
* **Deep Feature Fusion:** Combines tabular, image, and signal data at the feature level.
* **Real-Time Inference:** Built with a Flask web application for immediate results.
* **Modular Architecture:** Scalable project structure for future medical modalities.

---

## 🧠 Model Architecture Overview

### 1️⃣ ANN – Clinical & Demographic Data
* **Purpose:** Analyze structured patient data (Age, BMI, BP, Cholesterol).
* **Architecture:** Dense (128) → ReLU → Dropout → Dense (64) → ReLU → Dropout → Embedding Layer (32-D).
* **Performance:** Accuracy ~90%.

### 2️⃣ CNN – Chest X-ray Analysis
* **Purpose:** Extract visual patterns from chest X-ray images.
* **Model:** Pre-trained **MobileNetV2** (Transfer Learning).
* **Input Size:** 224 × 224 RGB.
* **Performance:** Test Accuracy ~83%.

### 3️⃣ LSTM – ECG Signal Processing
* **Purpose:** Analyze ECG time-series signals to detect abnormal heart rhythms.
* **Architecture:** Bidirectional LSTM (64 units) → LSTM (32 units) → Embedding Layer (64-D).

### 4️⃣ Fusion Model – Multi-Modal Integration
* **Mechanism:** Concatenation of embeddings (32-D ANN + 1280-D CNN + 64-D LSTM).
* **Architecture:** Dense (256) → ReLU → Dropout → Dense (128) → Sigmoid Output.

---

## 📊 Risk Interpretation

| Prediction Probability | Risk Level | Recommendation |
| :--- | :--- | :--- |
| **≥ 0.75** | 🔴 **High Risk** | Immediate cardiologist consultation recommended |
| **0.45 – 0.74** | 🟡 **Moderate Risk** | Lifestyle improvement and regular monitoring |
| **< 0.45** | 🟢 **Low Risk** | Maintain a healthy routine |

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Science:** Scikit-learn, NumPy, Pandas, OpenCV
* **Web Framework:** Flask
* **Frontend:** HTML & CSS

## 📦 Deployment
The system is deployment-ready and can be hosted via:
* **Render / Hugging Face Spaces**
* **Docker-based cloud services**

## ⚠️ Disclaimer
This project is for **educational and research purposes only**. It is not a certified medical diagnostic system and should not be used as a replacement for professional medical advice.

---

## 👩‍💻 Author
**Musfira Mubeen** *Aspiring AI Engineer & Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/YOUR_LINKEDIN_HERE)
[![YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=flat&logo=youtube)](https://youtu.be/DGfnnwHmDkg)

⭐ *If you find this project interesting, feel free to star the repository!*
