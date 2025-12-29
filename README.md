---
title: Face To Voice
emoji: 🗣️
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: AI Face-to-Voice Generator
---

# 🗣️ Face-to-Voice AI

An AI system that generates synthetic voice from facial images by predicting voice characteristics based on facial features and demographics.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yyuukkii/face-to-voice)

---

## 📋 Overview

This project maps facial features to voice characteristics using deep learning:

1. **Face Analysis**: Extracts face embeddings (ArcFace) and demographics (age, gender)
2. **Voice Prediction**: Maps face features to voice style vectors via trained neural networks
3. **Speech Synthesis**: Generates natural speech using StyleTTS2 with predicted voice

---

## 🏗️ Project Structure

```
face-to-voice/
├── data/
│   ├── face_vectors.pkl           # Preprocessed face embeddings
│   ├── audio_vectors.pkl          # Preprocessed audio embeddings
│   └── face_to_voice_dataset.pkl  # Combined training dataset
│
├── models/
│   ├── model_m.keras              # Male voice prediction model
│   └── model_f.keras              # Female voice prediction model
│
├── notebooks/
│   ├── 01_create_dataset.ipynb    # Dataset creation pipeline
│   ├── 02_train_model.ipynb       # Model training
│   └── 03_inference.ipynb         # Inference testing
│
├── src/
│   ├── app.py                     # Streamlit web application
│   ├── config.py                  # Configuration settings
│   ├── inference.py               # Inference engine
│   └── utils.py                   # Utility functions
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- FFmpeg (required for audio processing)
- eSpeak-ng (required for phonemization in TTS)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Yuki-20000/face-to-voice.git
cd face-to-voice
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the web application**
```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

---

## 🎯 Usage

1. Upload a face image (JPG/PNG)
2. Enter text to be spoken (English only, max 200 characters)
3. Click "Generate Voice" to synthesize speech
4. Listen to the generated audio or download as WAV file

---

## 📊 Performance

**Evaluation Metric**: Cosine Similarity between predicted and ground-truth voice embeddings

| Model | Mean Cosine Similarity |
|-------|------------------------|
| Male | 0.8000 |
| Female | 0.7997 |

---

## 🔬 Technical Approach

### 1. Gender-Specific Modeling

Training a single unified model struggled to capture the fine-grained nuances within each gender.  
This often resulted in regression to the mean, producing generic, "averaged" voices that failed to reflect the diverse characteristics found within specific gender groups.

**Solution**:  
Implemented a routing mechanism that creates specialized latent spaces for each gender.  
By dedicating a model to each group, the system can better capture the complex intra-gender distributions, resulting in more convincing and distinct voice generation.

### 2. Variance Regularization Loss

Standard training methods often encourage the model to output a "safe" average voice to minimize prediction error.  
This results in every generated voice sounding identical (low variance).

**Solution**:  
Designed a custom penalty term in the loss function that monitors the diversity of voices within each training batch.  
If the model starts producing similar voices, it applies a penalty, forcing the network to maintain natural human-like variety in its predictions while keeping high similarity to ground truth.

### 3. Architecture Stack

- **Input**: Face embedding (ArcFace, 512-dim) + Age (one-hot encoded, 6 bins)
- **Model**: Multi-Layer Perceptron with Dropout and L2 normalization
- **Output**: Voice style vector (256-dim, StyleTTS2 latent space)
- **Training**: Cosine similarity loss + variance regularization

### 4. Training Pipeline

1. **Dataset Creation**: Extract face/voice embeddings from VGGFace2/VoxCeleb2, align by speaker ID
2. **Model Training**: Train separate models for male/female voices with custom loss function
3. **Inference**: Predict voice style vector from face, synthesize speech with StyleTTS2

### Datasets Used

- **VGGFace2**: Face images dataset
  - Citation: Cao, Q., Shen, L., Xie, W., Parkhi, O. M., & Zisserman, A. (2018). VGGFace2: A dataset for recognising faces across pose and age. *FG 2018*.
  - Source: [Academic Torrents](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b)

- **VoxCeleb2**: Audio dataset
  - Citation: Chung, J. S., Nagrani, A., & Zisserman, A. (2018). VoxCeleb2: Deep Speaker Recognition. *INTERSPEECH 2018*.
  - Source: [Hugging Face](https://huggingface.co/datasets/Reverb/voxceleb2)

---

## ⚠️ Limitations

- **Language**: English text generation only (StyleTTS2 base model limitation)
- **Input Quality**: Requires clear, frontal face images for accurate embedding extraction
- **Voice Accuracy**: Predicted voice characteristics may not perfectly match actual voice

---

## 📄 License

This project is for educational and research purposes only.

**Dataset Licenses:**
- VGGFace2: Research use only
- VoxCeleb2: Research use only

Please cite the original datasets if you use this work.

---

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for face analysis
- [StyleTTS2](https://github.com/yl4579/StyleTTS2) for speech synthesis
- VGGFace2 and VoxCeleb2 dataset creators

---

**Note**: This is a research project and should not be used for malicious purposes such as deepfakes or identity fraud.
