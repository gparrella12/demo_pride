import os
import sys
import cv2
import numpy as np
import threading
import time
import platform

if getattr(sys, 'frozen', False):
    # Siamo dentro l'eseguibile
    # Puntiamo alla cartella interna dove abbiamo mappato i pesi (vedi build.yml)
    os.environ['DEEPFACE_HOME'] = os.path.join(sys._MEIPASS, 'deepface_home')
    print(f"Running in frozen mode. DeepFace home set to internal: {os.environ['DEEPFACE_HOME']}")
    
    # Verifica debug (opzionale, per vedere se i file ci sono)
    weights_path = os.path.join(os.environ['DEEPFACE_HOME'], 'weights')
    if os.path.exists(weights_path):
         print(f"Internal weights found: {os.listdir(weights_path)}")
    else:
         print("ERROR: Internal weights folder not found!")
else:
    # Siamo in sviluppo locale (file .py standard)
    print("Running in standard script mode.")
    # Non serve fare nulla, usa il default ~/.deepface


from deepface import DeepFace

# --- CONFIGURAZIONE DEMO ---
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
    """ Gestisce i percorsi per file dati (overlay.png) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
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
    print("  WARMUP IN CORSO (Caricamento pesi dai file interni)...")
    # Questo passaggio ora caricherà i file direttamente dall'interno dell'exe
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(dummy, actions=['emotion', 'age', 'gender'], detector_backend=BACKEND, enforce_detection=False, silent=True)
        print("✅ WARMUP COMPLETATO.")
    except Exception as e:
        print(f"⚠️ Errore Warmup: {e}")

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
    # (Codice disegno semplificato per brevità, usa quello tuo completo se preferisci)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, data['label'], (x, y-25), cv2.FONT_HERSHEY_TRIPLEX, 1.2, tcol, 2)
    cv2.putText(img, data['sub'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcol, 1)

def main():
    # Su Windows, serve per evitare che il multiprocessing (usato da alcune librerie)
    # crei loop infiniti quando l'app è congelata.
    import multiprocessing
    multiprocessing.freeze_support()

    warmup_system()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920); cap.set(4, 1080)
    
    overlay = None
    try: overlay = cv2.imread(resource_path('overlay.png'), -1)
    except: pass

    analyzer = EmotionAnalyzer()
    analyzer.start()

    win_name = 'Emotion AI Demo All-In-One'
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
    main()