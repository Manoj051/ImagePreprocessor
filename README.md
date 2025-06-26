# 🧠 Face Preprocessing Pipeline – Built for Accuracy, Forged in Grit

This project is the foundational preprocessing pipeline for a facial recognition system.
It performs high-accuracy face detection, alignment, enhancement, and vector serialization
— optimized specifically for ML models like FaceNet.

But more than that, it's the **first step in a personal journey**.
Built from scratch, self-taught, debugged through setbacks, and shipped with resolve.

---

## 🚀 Features

* 🧽 **CLAHE Enhancement** – Improves contrast on grayscale face images
* 🎯 **MTCNN Detection & Alignment** – Robust multi-stage face landmarking
* 📀 **Vector Storage** – Saves `.npy`-formatted embeddings ready for downstream ML
* ⚙️ **One-click Local Execution** – Minimal setup using `requirements-working.txt`
* ↻ **Built for Integration** – Designed to plug directly into FaceNet-based systems

---

## 💪 Tech Stack

| Component  | Version        |
| ---------- | -------------- |
| Python     | 3.10           |
| OpenCV     | 4.9.0.80       |
| MTCNN      | 0.1.1          |
| TensorFlow | 2.10.0         |
| NumPy      | < 2.0 (1.24.x) |

---

## 🧠 Setup Instructions

```bash
# Create Virtual Environment
python -m venv facenet-env

# Activate (Windows)
facenet-env\Scripts\activate

# Install Dependencies
pip install -r requirements-working.txt

# Run Preprocessor
python preprocessor.py
```

---

## 📈 Why This Project Matters

This isn't just a script.
It's a key component in a modular facial recognition system I'm building from the ground up:
from **image capture** to **vector generation**, from **embedding matching** to **user verification.**

It was also the moment where I realized:

> **I could build serious tech.**
> Not with a team. Not with handholding. But with relentless effort and curiosity.

---

## 🔮 Next Steps

* 🧠 Integrate with FaceNet for real-time embedding inference
* 🌐 Wrap in FastAPI for microservice deployment
* 𞳁 Add JWT-based login and admin-only fallback support
* 📦 Containerize for local & cloud deployment
* 📊 Build a dashboard for logging attendance via face match

---

## ✍️ Author

**Manu**

* Self-driven learner, future ML Engineer
* Code built with intention, not tutorials
* Building phase-by-phase toward product-grade software

---

**“This was Phase 1. And it runs.
What comes next... scales.”**
