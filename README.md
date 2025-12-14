🧠 AI Personal Health Tracker
Multi-Modal Heart Disease Risk Prediction System

📌 Project Overview
The AI Personal Health Tracker is an end-to-end artificial intelligence application designed to assess heart disease risk by combining multiple types of patient data into one unified prediction system.

Instead of relying on a single data source, this project integrates:

Clinical & demographic data (tabular)

ECG time-series signals

Chest X-ray images

Each data type is processed using a specialized deep learning model, and their outputs are fused to produce a final, patient-level risk prediction.
The goal of this project is to simulate a real-world clinical decision support system and demonstrate how multi-modal AI can be applied in healthcare.

This project is built for portfolio presentation, interviews, and real-world learning, focusing on applied system design rather than isolated models.

🎯 Project Goals
Build a realistic, multi-modal AI system for healthcare
Apply ANN, CNN, and LSTM in a single unified pipeline
Practice end-to-end ML system development
Deliver interpretable risk levels with actionable recommendations
Create a job-ready, professional AI portfolio project

🚀 Key Features
Multi-modal AI system (tabular + image + time-series)
Independent training of ANN, CNN, and LSTM models
Deep feature-level fusion for final prediction
Real-time inference via a Flask web application
Clear risk categorization and health recommendations
Modular and scalable project structure
Suitable for portfolio, interviews, and demos

🧠 Model Architecture Overview
1️⃣ ANN – Clinical & Demographic Data
Purpose:
Analyze structured patient data and generate meaningful feature embeddings.
Sample Input Features:
Age
BMI
Blood pressure
Cholesterol level
Gender
Architecture Summary:
Dense (128) → ReLU → Dropout
Dense (64) → ReLU → Dropout
Embedding Layer (32-D)
Sigmoid Output
Performance:
Accuracy: ~90%

2️⃣ CNN – Chest X-ray Analysis
Purpose:
Extract visual patterns from chest X-ray images that may indicate heart-related abnormalities.
Model Details:
Pre-trained MobileNetV2 (transfer learning)
Global Average Pooling
Dropout for regularization
Sigmoid output layer
Input Size:
224 × 224 RGB images
Performance:
Test Accuracy: ~83%

3️⃣ LSTM – ECG Signal Processing
Purpose:
Analyze ECG time-series signals to detect abnormal heart rhythms and patterns.
Architecture Summary:
Bidirectional LSTM (64 units)
LSTM (32 units)
Embedding Layer (64-D)
Sigmoid Output

4️⃣ Fusion Model – Multi-Modal Integration
Purpose:
Combine insights from all three models into a single, more accurate prediction.
Fusion Inputs:
ANN Embedding: 32-D
CNN Embedding: 1280-D
LSTM Embedding: 64-D

Fusion Architecture:
Concatenation layer
Dense (256) → ReLU → Dropout
Dense (128) → ReLU
Sigmoid Output

📊 Risk Interpretation
Prediction Probability	Risk Level	Recommendation
≥ 0.75	High Risk	Immediate cardiologist consultation recommended
0.45 – 0.74	Moderate Risk	Lifestyle improvement and regular monitoring
< 0.45	Low Risk	Maintain a healthy routine

🌐 Web Application
Frontend
HTML & CSS
Clean, healthcare-focused design
Responsive layout for different devices
Backend
Flask (Python)
Secure file uploads
Modular prediction pipeline
User Flow
User enters clinical details
Uploads ECG signal data
Uploads optional chest X-ray image
System returns risk level and recommendation

🛠️ Tech Stack
Python
TensorFlow / Keras
Scikit-learn
NumPy, Pandas
OpenCV
Flask
HTML & CSS

📦 Deployment Status
Due to model size constraints, the project is currently not deployed publicly.
The system is fully deployment-ready and can be hosted using platforms such as:
Render
Hugging Face Spaces
Docker-based cloud services

💼 Potential Use Cases
Health-tech startups
Clinical AI research prototypes
Decision support systems
Advanced AI portfolio projects

⚠️ Disclaimer
This project is intended for educational and research purposes only.
It is not a certified medical diagnostic system and should not be used as a replacement for professional medical advice.

👩‍💻 Author
Musfira Mubeen
Aspiring AI Engineer & Data Scientist

⭐ If you find this project interesting, feel free to star the repository!
