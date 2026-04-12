#  AI-Based Gait Analysis System

##  Overview

This project is an end-to-end AI system that analyzes human gait using video and sensor data to detect abnormalities and classify gait patterns.

It combines:

* Computer Vision (pose estimation)
* Deep Learning (LSTM, CNN)
* Machine Learning (Random Forest, XGBoost)
* Web Development (FastAPI + React)

---

##  Features

*  Video-based gait analysis using pose estimation
*  Sensor-based gait classification
*  Hybrid ML + DL models
*  Risk assessment and report generation
*  Full-stack web application (frontend + backend)

---

##  Project Structure

```
backend/        → API, model inference, database
training/       → model training & preprocessing
frontend-app/   → user interface (React)
```

---

##  Models Used

* LSTM (Video sequence modeling)
* CNN + ExtraTrees (Hybrid model)
* Random Forest
* XGBoost

---

##  Tech Stack

* Python, FastAPI
* PyTorch, Scikit-learn
* MediaPipe (pose detection)
* React (frontend)

---

##  Dataset & Models

Due to size limitations, datasets and trained models are not included.

 Dataset & Models: [https://drive.google.com/drive/folders/1HeVQF2_33u4MBPdATKQm8XSn2F-Dv8Vz]

---

##  How to Run

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend-app
npm install
npm run dev
```

---

##  Applications

* Healthcare monitoring
* Injury detection
* Rehabilitation analysis
* Sports performance tracking

---

##  Author

**Pradhuman Singh Shekhawat**
B.Tech AI & ML

---

##  Future Improvements

* Real-time gait detection
* Mobile app integration
* More disease classification
