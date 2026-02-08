# Emotion Recognition System (PyTorch Edition)

A high-performance, real-time facial emotion recognition system built with PyTorch.

This application analyzes video from your webcam to detect faces and classify emotions into 8 categories (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral, Contempt). It features a modern "Sci-Fi" HUD overlay and automatic hardware acceleration.

---

## ✨ Key Features

* **Robust Face Detection:** Uses **MTCNN** (Multi-task Cascaded Convolutional Networks) for accurate detection even in difficult lighting.
* **Real-Time Analysis:** Optimized pipeline using **HSEmotion** for fast inference.
* **Hardware Acceleration:** Automatically detects and uses **CUDA** (NVIDIA GPUs) or falls back to optimized CPU execution.
* **Graphical Interface:**
    * **Loading Screen:** Visual progress bar during model initialization/download.
    * **HUD Overlay:** Custom graphical overlay (`overlay.png`) with dynamic text shadows for readability.
* **Standalone Executables:** Available as a single file (`.exe` or `.app`) for easy distribution.

---

## 🛠️ Developer Setup (Run from Source)

If you are a developer and want to run or modify the code, follow these steps.

### 1. Prerequisites
* Python **3.10** or **3.11** (Recommended).
* A working webcam.

### 2. Create a Virtual Environment (Recommended)
Using Conda is highly recommended to manage dependencies cleanly.

```bash
# Create the environment
conda create -n emotion_env python=3.10

# Activate the environment
conda activate emotion_env

# Install dependencies
pip install -r requirements.txt
```
### 3. Run the Application
```bash
python demo.py
```

> First Run Notice: The first time you run the app, a loading bar will appear. The system is downloading the pre-trained neural network weights (approx. 100MB). This happens only once; subsequent runs will be instant.

## 📦 Standalone Executables (No Python Required)

If you just want to use the application without installing Python, you can download the pre-compiled versions.

### How to Download
1.  Go to the **[Releases](../../releases)** section of this repository (on the right sidebar).
2.  Download the file for your operating system:
    * 🪟 **Windows:** `EmotionDemo_Win.exe`
    * 🍎 **macOS:** `EmotionDemo_Mac.zip`

### How to Run

#### On Windows
Double-click `EmotionDemo_Win.exe`.
* **Security Warning:** If Windows SmartScreen says *"Windows protected your PC"*, click **More Info** -> **Run Anyway**.
* *Why?* This happens because the application is not code-signed with a paid Microsoft certificate. It is safe to proceed.

#### On macOS
Unzip the file and double-click the application.
* **Security Warning:** If you see *"App cannot be opened because the developer cannot be verified"*:
    1.  Go to **System Settings** -> **Privacy & Security**.
    2.  Scroll down to the Security section.
    3.  Click **Open Anyway** next to the notification about the blocked app.

> **⚠️ Important: Startup Delay**
> The standalone executable is a compressed archive. When you double-click it for the first time, it may take **10 to 30 seconds** to extract itself to a temporary folder before the Loading Screen appears.
> **Please be patient and do not double-click multiple times.**

---

## 📂 Project Structure

* `demo.py`: Main entry point. Contains the GUI logic (Tkinter loading screen), AI threading, and video processing loop.
* `requirements.txt`: List of Python dependencies (pinned versions to ensure stability).
* `overlay.png`: The graphical frame overlaid on the video feed.
* `.github/workflows/build.yaml`: CI/CD configuration to automatically build Windows and Mac executables via GitHub Actions.

---

## 📄 Credits & Licenses

This project utilizes the following open-source libraries:
* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for face detection.
* [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) for emotion recognition.
* [timm](https://github.com/huggingface/pytorch-image-models) for computer vision models.
* [OpenCV](https://opencv.org/) for real-time video processing.