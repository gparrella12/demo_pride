import os
import sys
import cv2
import numpy as np
import threading
import time
import platform
import traceback
import datetime

# --- CONFIGURAZIONE AMBIENTE OFFLINE ---
if getattr(sys, 'frozen', False):
    # Se siamo in un exe (PyInstaller)
    # Impostiamo la HOME di DeepFace alla cartella temporanea dell'exe
    bundle_dir = sys._MEIPASS
    os.environ['DEEPFACE_HOME'] = bundle_dir
    
    # Debug (scrive su file se qualcosa non va)
    print(f"[INFO] Running Frozen. DeepFace Home: {bundle_dir}")
else:
    # Se siamo in sviluppo locale
    print("[INFO] Running Standard Script.")
    # Usa la default ~/.deepface

# --- IMPORT DOPO IL SETUP ---
# Silenzia TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deepface import DeepFace

# --- CONFIGURAZIONE ---
BACKEND = 'ssd' 

COLORS = {
    'positive': (0, 255, 255),
    'negative': (0, 0, 255),
    'neutral':  (255, 255, 255),
}
TRADUZIONI = {
    'angry': 'Arrabbiato', 'disgust': 'Disgustato', 'fear': 'Spaventato',
    'happy': 'Felice', 'sad': 'Triste', 'surprise': 'Sorpreso', 'neutral': 'Neutro',
    'Man': 'Uomo', 'Woman': 'Donna'
}

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def overlay_transparent(background, overlay):
    if overlay is None: return background
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    if (bg_w, bg_h) != (ov_w, ov_h):
        try: overlay = cv2.resize(overlay, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
        except: return background
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        alpha = a / 255.0
        alpha = np.dstack((alpha, alpha, alpha))
        background = (1.0 - alpha) * background + alpha * overlay_rgb
    return background.astype(np.uint8)

def warmup_system():
    print("  CARICAMENTO SISTEMA (Attendere, potrebbe richiedere secondi)...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    # Primo test a vuoto: se i file ci sono, li carica istantaneamente
    DeepFace.analyze(dummy, actions=['emotion', 'age', 'gender'], detector_backend=BACKEND, enforce_detection=False, silent=True)
    print("✅ SISTEMA PRONTO.")

class EmotionAnalyzer:
    def __init__(self):
        self.frame_to_process = None
        self.results = []
        self.running = True
        self.lock = threading.Lock()
        self.new_data_available = False
        self.thread = threading.Thread(target=self._analyze_loop, daemon=True)

    def start(self): self.thread.start()
    def stop(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join(timeout=1.0)

    def update_frame(self, frame):
        with self.lock:
            self.frame_to_process = frame.copy()
            self.new_data_available = True
    def get_results(self): return self.results

    def _analyze_loop(self):
        while self.running:
            process_now = False
            with self.lock:
                if self.new_data_available:
                    frame = self.frame_to_process
                    self.new_data_available = False
                    process_now = True
            if process_now and frame is not None:
                try:
                    raw = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False, detector_backend=BACKEND, silent=True)
                    if isinstance(raw, dict): raw = [raw]
                    processed = []
                    for face in raw:
                        if face['region']['w'] < 40: continue
                        eng_emo = face['dominant_emotion']
                        color = COLORS['neutral']
                        txt_col = (0,0,0)
                        if eng_emo in ['happy', 'surprise']: color = COLORS['positive']
                        elif eng_emo in ['angry', 'sad', 'fear', 'disgust']: 
                             color = COLORS['negative']
                             txt_col = (255,255,255)
                        processed.append({
                            'box': (face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']),
                            'label': f"{TRADUZIONI.get(eng_emo, eng_emo).upper()}",
                            'sub': f"{TRADUZIONI.get(face['dominant_gender'])}, {face['age']}",
                            'col': color, 'tcol': txt_col
                        })
                    self.results = processed
                except: pass
            else: time.sleep(0.01)

def draw_hud(img, data):
    x, y, w, h = data['box']
    color = data['col']
    tcol = data['tcol']
    # Mirino semplificato
    l = int(w * 0.2)
    cv2.line(img, (x, y), (x + l, y), color, 4)
    cv2.line(img, (x, y), (x, y + l), color, 4)
    cv2.line(img, (x+w, y), (x+w-l, y), color, 4)
    cv2.line(img, (x+w, y), (x+w, y+l), color, 4)
    cv2.line(img, (x, y+h), (x+l, y+h), color, 4)
    cv2.line(img, (x, y+h), (x, y+h-l), color, 4)
    cv2.line(img, (x+w, y+h), (x+w-l, y+h), color, 4)
    cv2.line(img, (x+w, y+h), (x+w, y+h-l), color, 4)
    
    # Testo
    cv2.putText(img, data['label'], (x, y-25), cv2.FONT_HERSHEY_TRIPLEX, 1.2, tcol, 2)
    cv2.putText(img, data['sub'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcol, 1)

def main():
    import multiprocessing
    multiprocessing.freeze_support() # Fix per Windows frozen

    warmup_system()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920); cap.set(4, 1080)
    
    overlay = None
    try: overlay = cv2.imread(resource_path('overlay.png'), -1)
    except: pass

    analyzer = EmotionAnalyzer()
    analyzer.start()

    win_name = 'Emotion AI Demo'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    print("Premi 'q' per uscire.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            analyzer.update_frame(frame)
            for data in analyzer.get_results(): draw_hud(frame, data)
            frame = overlay_transparent(frame, overlay)
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt: pass
    finally:
        analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # BLOCCO ANTI-CRASH PER MAC
    # Se l'app crasha, scrive il motivo sul Desktop
    try:
        main()
    except Exception as e:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        err_file = os.path.join(desktop, "CRASH_LOG_EMOTION.txt")
        with open(err_file, "w") as f:
            f.write(f"Crash Time: {datetime.datetime.now()}\n")
            f.write(traceback.format_exc())