# 🤟 Sign Language Recognition System

A hand gesture recognition system for Indian Sign Language (ISL) alphabets using MediaPipe and Machine Learning.

## 📋 Overview
This project detects hand gestures using a webcam and predicts the corresponding alphabet.
It uses:
- **MediaPipe** for hand landmark detection (21 points per hand)
- **scikit-learn** ML models for classification
- **Streamlit** web interface for real-time recognition

**Best Model**: KNN with **76.77% test accuracy**

## 🛠️ Tech Stack

- **Hand Detection**: MediaPipe 0.10.7
- **ML**: scikit-learn 1.3.2
- **Computer Vision**: OpenCV 4.8.1.78
- **Web App**: Streamlit 1.28.2
- **Data Processing**: Pandas, NumPy, Matplotlib

## ⚡ Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run Web Interface
```bash
streamlit run app.py
```

### Real-time Recognition
```bash
python src/recognise.py
```

### Train Models
```bash
python src/train_model.py
```

## 📁 Project Structure

```
sign/
├── app.py                    # Streamlit web interface
├── hand_landmarker.task      # MediaPipe model
├── requirements.txt
├── data/
│   ├── raw/                  # Raw gesture images (person_1, 2, 3)
│   └── processed/            # Extracted features (train/test/val.pkl)
├── src/
│   ├── collect_data.py       # Capture gesture images
│   ├── create_dataset.py     # Extract hand landmarks
│   ├── train_model.py        # Train & compare models
│   └── recognise.py          # Real-time recognition
├── models/
│   └── sign_language_model.pkl  # Best trained model
└── plots/                    # Generated visualizations
```

## 📊 Model Performance

| Model         | Test Accuracy |
|---------------|---------------|
| **KNN**       | **76.77%**    |
| SVM           | 72.31%        |
| Random Forest | 71.54%        |
| MLP           | 65.38%        |

- **Training Data**: person_1 (26 classes × ~50 samples)
- **Test Data**: person_2 (different person)
- **Validation Data**: person_3 (different person)
- **Features**: 84 (21 landmarks × 2 hands × 2 coordinates, normalized + scaled)
- **Evaluation**: 5-fold cross-validation

## 💡 Key Features

✅ Real-time gesture recognition from webcam  
✅ Supports A–Z alphabets  
✅ Shows hand landmarks  
✅ Multiple ML model comparison  
✅ Web interface (Streamlit) for easy testing  
✅ Desktop app for continuous recognition 

## ⚠️ Limitations

- Limited dataset (only 3 people)
- Accuracy can vary depending on lighting and hand position
- Only works for static gestures

## 🚀 Future Improvements

- Collect more data from different people
- Improve accuracy using better features or deep learning
- Add support for dynamic gestures (words/sentences)

## 📋 Dataset Info

- **Classes**: 26 (A-Z alphabets)
- **Split**: Person-based (person_1=train, person_2=test, person_3=val)
- **Total Samples**: ~4,000 images
- **Features**: 84-dimensional (hand landmarks)

---

**Status**: Working Prototype ✅  
**Accuracy**: 76.77% (KNN)  
**Last Updated**: April 2026 