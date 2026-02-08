import cv2
import torch
import numpy as np
import threading
import time
import sys
import os
import traceback
import tkinter as tk
from tkinter import ttk

# --- CONFIGURATION ---
DEBUG_MODE = False
PROCESS_WIDTH = 640  # Resolution for AI processing (lower = faster)

# --- VISUAL SETTINGS ---
COLORS = {
    'positive': (0, 255, 255),   # Yellow
    'negative': (0, 0, 255),     # Red
    'neutral':  (255, 255, 255), # White
}

# Translations (English Key -> Italian Display)
TRANSLATIONS = {
    'Anger': 'ARRABBIATO', 
    'Disgust': 'DISGUSTATO', 
    'Fear': 'SPAVENTATO',
    'Happiness': 'FELICE', 
    'Sadness': 'TRISTE', 
    'Surprise': 'SORPRESO', 
    'Neutral': 'NEUTRO', 
    'Contempt': 'DISPREZZO' 
}

# --- CRITICAL FIX FOR PYTORCH 2.6+ ---
# hsemotion is not yet updated for 'weights_only=True'.
# This monkey-patch allows loading older weights safely.
try:
    _original_load = torch.load
    def safe_load_wrapper(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = safe_load_wrapper
except:
    pass

from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

# --- UTILITIES ---
def log(message):
    """Prints debug messages to console if enabled."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def overlay_transparent(background, overlay):
    """Overlays a PNG with transparency on top of the background."""
    if overlay is None: return background
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    
    if (bg_w, bg_h) != (ov_w, ov_h):
        try:
            overlay = cv2.resize(overlay, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
        except:
            return background
            
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        alpha = a / 255.0
        alpha = np.dstack((alpha, alpha, alpha))
        background = (1.0 - alpha) * background + alpha * overlay_rgb
        
    return background.astype(np.uint8)

def get_device():
    """Detects best available hardware acceleration."""
    if torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device('cuda') # NVIDIA
    else:
        return torch.device('cpu')

# --- LOADING SCREEN GUI ---
class LoadingScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion AI - Starting")
        
        # Center the window
        w, h = 400, 150
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        self.root.configure(bg='#2c3e50')
        self.root.overrideredirect(True) # Remove title bar for splash effect

        # Label
        self.lbl = tk.Label(self.root, text="Initializing AI Models...", fg='white', bg='#2c3e50', font=("Helvetica", 14))
        self.lbl.pack(pady=20)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10)

        self.analyzer = None

    def update_status(self, text, value):
        """Updates text and progress bar."""
        self.lbl.config(text=text)
        self.progress['value'] = value
        self.root.update_idletasks()

    def start_loading(self):
        """Runs the loading process in a separate thread."""
        t = threading.Thread(target=self._load_models)
        t.start()
        self.root.mainloop()

    def _load_models(self):
        try:
            self.update_status("Detecting Hardware...", 10)
            device = get_device()
            time.sleep(0.5) # Slight pause for UX

            self.update_status(f"Loading MTCNN (Face Detection) on {device}...", 30)
            
            # Initialize Analyzer (MTCNN loads here)
            mtcnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            mtcnn = MTCNN(
                keep_all=True, 
                device=mtcnn_device, 
                select_largest=False, 
                post_process=False,
                thresholds=[0.6, 0.7, 0.7]
            )
            
            self.update_status("Downloading/Loading HSEmotion Weights...", 60)
            # HSEmotion loads here
            emo_model = HSEmotionRecognizer(device=device)
            
            self.update_status("Warming up inference engine...", 90)
            # Dummy inference to compile graphs
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            mtcnn.detect(dummy)

            self.update_status("Ready!", 100)
            time.sleep(0.5)
            
            # Pass loaded models to the Analyzer class
            self.analyzer = PyTorchAnalyzer(device, mtcnn, emo_model)
            
            # Close GUI and start main app
            self.root.destroy()
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 0)
            print(traceback.format_exc())
            time.sleep(5)
            self.root.destroy()
            sys.exit(1)

# --- AI ANALYZER ---
class PyTorchAnalyzer:
    def __init__(self, device, mtcnn, emo_model):
        self.device = device
        self.mtcnn = mtcnn
        self.emo_model = emo_model
        
        self.frame_to_process = None
        self.results = []
        self.running = True
        self.lock = threading.Lock()
        self.new_data_available = False

    def start(self):
        """Starts the background analysis loop."""
        threading.Thread(target=self._analyze_loop, daemon=True).start()

    def update_frame(self, frame):
        """Push a new frame for analysis."""
        with self.lock:
            self.frame_to_process = frame.copy()
            self.new_data_available = True

    def get_results(self):
        """Get latest results."""
        return self.results

    def _analyze_loop(self):
        log("Analysis loop started.")
        while self.running:
            process_now = False
            frame = None
            with self.lock:
                if self.new_data_available:
                    frame = self.frame_to_process
                    self.new_data_available = False
                    process_now = True
            
            if process_now and frame is not None:
                try:
                    # 1. Resize (Optimization)
                    h_orig, w_orig = frame.shape[:2]
                    scale = PROCESS_WIDTH / float(w_orig)
                    new_h = int(h_orig * scale)
                    small_frame = cv2.resize(frame, (PROCESS_WIDTH, new_h))
                    
                    # MTCNN needs RGB
                    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # 2. Face Detection
                    boxes, probs = self.mtcnn.detect(rgb_small)
                    
                    if boxes is None:
                        self.results = [] 
                        continue
                    
                    processed = []
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        
                        # Boundary checks
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(PROCESS_WIDTH, x2); y2 = min(new_h, y2)
                        
                        # Size filter
                        if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                        
                        # 3. Face Extraction
                        face_img = rgb_small[y1:y2, x1:x2]
                        if face_img.size == 0: continue

                        # 4. Emotion Recognition
                        emotion, scores = self.emo_model.predict_emotions(face_img, logits=False)
                        log(f"Detected: {emotion}")
                        
                        # 5. Upscale Coordinates
                        real_x1 = int(x1 / scale)
                        real_y1 = int(y1 / scale)
                        real_w = int((x2 - x1) / scale)
                        real_h = int((y2 - y1) / scale)
                        
                        # Color Logic
                        color = COLORS['neutral']
                        if emotion in ['Happiness', 'Surprise']: 
                            color = COLORS['positive']
                        elif emotion in ['Anger', 'Sadness', 'Fear', 'Disgust', 'Contempt']: 
                            color = COLORS['negative']

                        processed.append({
                            'box': (real_x1, real_y1, real_w, real_h),
                            'label': TRANSLATIONS.get(emotion, emotion),
                            'col': color
                        })
                            
                    self.results = processed
                    
                except Exception as e:
                    print(f"[THREAD ERROR]: {e}")
            else:
                time.sleep(0.01)

# --- HUD DRAWING ---
def draw_hud(img, data):
    x, y, w, h = data['box']
    color = data['col']
    label_text = data['label']
    
    # Tech Corners
    corner_len = int(w * 0.25)
    thick = 3
    
    # Draw corners
    for fx, fy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
        dx = corner_len if fx == x else -corner_len
        dy = corner_len if fy == y else -corner_len
        cv2.line(img, (fx, fy), (fx + dx, fy), color, thick)
        cv2.line(img, (fx, fy), (fx, fy + dy), color, thick)

    # Label styling
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale = 2.0
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)
    
    # Position logic
    bg_x = x + (w - tw) // 2
    bg_y = y - 20 if y > 100 else y + h + 20 + th
        
    bg_x = max(10, min(bg_x, img.shape[1] - tw - 10))
    bg_y = max(th + 10, min(bg_y, img.shape[0] - 10))
    
    # Background Box
    pad = 10
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x - pad, bg_y - th - pad), (bg_x + tw + pad, bg_y + pad), color, -1)
    img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    
    # Text
    cv2.rectangle(img, (bg_x - pad, bg_y - th - pad), (bg_x + tw + pad, bg_y + pad), (50, 50, 50), 2)
    # Drop Shadow
    cv2.putText(img, label_text, (bg_x + 2, bg_y + 2), font, scale, (0,0,0), thickness + 2, cv2.LINE_AA)
    # Main Text
    text_col = (0,0,0) if color != COLORS['negative'] else (255,255,255)
    cv2.putText(img, label_text, (bg_x, bg_y), font, scale, text_col, thickness, cv2.LINE_AA)

# --- MAIN ENTRY POINT ---
def main():
    # 1. Show Loading Screen (Blocking until ready)
    loader = LoadingScreen()
    loader.start_loading()
    
    # 2. Retrieve Initialized Analyzer
    analyzer = loader.analyzer
    if analyzer is None:
        print("Initialization failed. Exiting.")
        return

    # 3. Start Application
    print("--- EMOTION RECOGNITION SYSTEM STARTED ---")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    overlay = None
    try: overlay = cv2.imread(resource_path('overlay.png'), -1)
    except: pass

    analyzer.start()

    win_name = 'Emotion Recognition System'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        analyzer.update_frame(frame)
        
        results = analyzer.get_results()
        for res in results:
            draw_hud(frame, res)

        frame = overlay_transparent(frame, overlay)

        cv2.imshow(win_name, frame)
        
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    analyzer.running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()