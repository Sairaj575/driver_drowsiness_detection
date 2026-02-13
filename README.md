# 🚗 Driver Drowsiness Detection System

Real-time Driver Drowsiness Detection System using:

-   👁️ Eye Aspect Ratio (EAR) for eye closure detection
-   😮 Yawning detection using a CNN (PyTorch)
-   🎯 MediaPipe Face Mesh for facial landmarks
-   🔊 Alarm alert system
-   🌐 Streamlit Web App Interface

------------------------------------------------------------------------

## 📌 Features

-   Real-time webcam monitoring
-   Eye blink detection using EAR threshold
-   Yawn detection using Deep Learning (CNN)
-   Alarm sound when drowsiness detected
-   Streamlit-based UI
-   GPU support (if CUDA available)

------------------------------------------------------------------------

## 🏗️ Project Structure

    .
    ├── app.py
    ├── drowsiness_detect.py
    ├── eye_utils.py
    ├── mouth_utils.py
    ├── model.py
    ├── train_mouth_pytorch.py
    ├── alert.py
    ├── config.py
    ├── requirements.txt
    ├── mouth_cnn.pth (after training)
    ├── dataset/
    │   └── train/
    │       ├── yawn/
    │       └── no_yawn/
    └── sounds/
        └── alarm.wav

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/Sairaj575/driver_drowsiness_detection.git
cd driver-drowsiness-detection
```

### 2️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🧠 Yawn Detection Model (CNN)

-   3 Convolution layers
-   MaxPooling
-   Dropout (0.5)
-   Fully Connected layers
-   Output: 2 classes (Yawn / No Yawn)

Input size: **64×64 grayscale image**

------------------------------------------------------------------------

## 🏋️ Training the Mouth Model

Dataset structure:

    dataset/train/
        ├── yawn/
        └── no_yawn/

Run:

``` bash
python train_mouth_pytorch.py
```

This generates:

    mouth_cnn.pth

------------------------------------------------------------------------

## 👁️ Eye Detection (EAR Method)

Configuration:

-   EAR_THRESHOLD = 0.25
-   EAR_CONSEC_FRAMES = 20
-   YAWN_CONSEC_FRAMES = 15

------------------------------------------------------------------------

## 🔔 Alert System

Make sure:

    sounds/alarm.wav

exists in your project directory.

------------------------------------------------------------------------

## 🌐 Running the Web App

``` bash
streamlit run app.py
```

Click **Start Camera** to begin monitoring.

------------------------------------------------------------------------

## 🔄 How It Works

1.  Webcam captures frame
2.  MediaPipe detects face landmarks
3.  Eye landmarks → EAR calculated
4.  Mouth region extracted → CNN classification
5.  If eyes closed for long duration OR yawning detected continuously →
    Alarm triggered

------------------------------------------------------------------------

## 🖥️ Requirements

-   Python 3.9 recommended
-   Webcam
-   Optional: GPU (CUDA supported)

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Performance optimization
-   Larger dataset for better accuracy

------------------------------------------------------------------------

## 👨‍💻 Author

Sairaj Umbarkar\
AI/ML Enthusiast

⭐ If you find this project useful, feel free to star the repository!
