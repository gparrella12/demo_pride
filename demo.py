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
from tkinter import messagebox
from datetime import datetime

# --- CONFIGURATION ---
DEBUG_MODE = False
PROCESS_WIDTH = 640  # Risoluzione interna AI (più bassa = più veloce)
EXPIRATION_DATE = datetime(2026, 12, 31) # Data di scadenza software

# --- VISUAL SETTINGS (COLORI ESPANSI) ---
# Formato BGR (Blue, Green, Red) - OpenCV usa questo formato
EMOTION_COLORS = {
    'Anger':     (0, 0, 255),       # Rosso Puro
    'Disgust':   (0, 100, 0),       # Verde Scuro
    'Fear':      (128, 0, 128),     # Viola
    'Happiness': (0, 215, 255),     # Oro/Giallo
    'Sadness':   (255, 0, 0),       # Blu
    'Surprise':  (255, 255, 0),     # Ciano
    'Neutral':   (220, 220, 220),   # Grigio Chiaro
    'Contempt':  (0, 140, 255)      # Arancione
}

# Colore di default se l'emozione non è in lista
DEFAULT_COLOR = (255, 255, 255)

# Traduzioni (Chiave Inglese -> Display Italiano)
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

# --- FIX CRITICO PER PYTORCH 2.6+ ---
# hsemotion non è ancora aggiornata per 'weights_only=True'.
# Questa patch intercetta il caricamento per evitare crash.
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
    """Stampa messaggi di debug solo se abilitato."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def get_color_for_emotion(emotion_key):
    """Ritorna il colore BGR specifico per l'emozione."""
    return EMOTION_COLORS.get(emotion_key, DEFAULT_COLOR)

def resource_path(relative_path):
    """Gestisce i percorsi sia per dev che per PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def overlay_transparent(background, overlay):
    """Sovrappone un PNG con trasparenza sopra il frame video."""
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
    """Rileva la migliore accelerazione hardware disponibile."""
    if torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon (Mac)
    elif torch.cuda.is_available():
        return torch.device('cuda') # NVIDIA (Windows/Linux)
    else:
        return torch.device('cpu')  # CPU Standard

# --- SCHERMATA DI CARICAMENTO (GUI) ---
class LoadingScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion AI - Starting")
        
        # Centra la finestra
        w, h = 450, 160
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        self.root.configure(bg='#2c3e50')
        self.root.overrideredirect(True) # Rimuove la barra del titolo

        # Etichetta
        self.lbl = tk.Label(self.root, text="Inizializzazione Modelli AI...", fg='white', bg='#2c3e50', font=("Helvetica", 18))
        self.lbl.pack(pady=25)

        # Barra di progresso
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=350, mode='determinate')
        self.progress.pack(pady=10)

        self.analyzer = None

    def update_status(self, text, value):
        self.lbl.config(text=text)
        self.progress['value'] = value
        self.root.update_idletasks()

    def start_loading(self):
        t = threading.Thread(target=self._load_models, daemon=True)
        t.start()
        self.root.mainloop()

    def _load_models(self):
        try:
            self.update_status("Controllo Hardware...", 10)
            device = get_device()
            time.sleep(0.5)

            self.update_status(f"Caricamento MTCNN (Face Detect) su {device}...", 30)
            
            # Forziamo CPU per MTCNN su Mac per stabilità, GPU se disponibile altrove
            mtcnn_device = torch.device('cpu') 
            
            mtcnn = MTCNN(
                keep_all=True, 
                device=mtcnn_device, 
                select_largest=False, 
                post_process=False,
                thresholds=[0.6, 0.7, 0.7]
            )
            
            self.update_status("Scaricamento/Caricamento Pesi HSEmotion...", 60)
            # HSEmotion usa il device principale (GPU/MPS se possibile)
            emo_model = HSEmotionRecognizer(device=device)
            
            self.update_status("Warmup Motore Neurale...", 90)
            # Inferenza a vuoto per compilare i grafici
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            mtcnn.detect(dummy)

            self.update_status("Pronto!", 100)
            time.sleep(0.5)
            
            self.analyzer = PyTorchAnalyzer(device, mtcnn, emo_model)
            
            # Chiude la GUI
            self.root.destroy()
            
        except Exception as e:
            self.update_status(f"Errore: {str(e)}", 0)
            print(traceback.format_exc())
            time.sleep(5)
            self.root.destroy()
            sys.exit(1)

# --- MOTORE AI ---
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
        self.thread = None 

    def start(self):
        """Avvia il thread di analisi come Demone (si chiude se il main muore)."""
        self.thread = threading.Thread(target=self._analyze_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        """Passa un nuovo frame da analizzare."""
        with self.lock:
            self.frame_to_process = frame.copy()
            self.new_data_available = True

    def get_results(self):
        return self.results

    def _analyze_loop(self):
        log("Thread Analisi Avviato.")
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
                    # 1. Ridimensiona (Ottimizzazione velocità)
                    h_orig, w_orig = frame.shape[:2]
                    scale = PROCESS_WIDTH / float(w_orig)
                    new_h = int(h_orig * scale)
                    small_frame = cv2.resize(frame, (PROCESS_WIDTH, new_h))
                    
                    # MTCNN vuole RGB
                    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # 2. Rilevamento Volti
                    boxes, probs = self.mtcnn.detect(rgb_small)
                    
                    if boxes is None:
                        self.results = [] 
                        continue
                    
                    processed = []
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        
                        # Controlli bordi
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(PROCESS_WIDTH, x2); y2 = min(new_h, y2)
                        
                        # Filtro dimensione minima
                        if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                        
                        # 3. Estrazione Volto
                        face_img = rgb_small[y1:y2, x1:x2]
                        if face_img.size == 0: continue

                        # 4. Analisi Emozione
                        emotion, scores = self.emo_model.predict_emotions(face_img, logits=False)
                        
                        # 5. Upscale Coordinate
                        real_x1 = int(x1 / scale)
                        real_y1 = int(y1 / scale)
                        real_w = int((x2 - x1) / scale)
                        real_h = int((y2 - y1) / scale)
                        
                        # Colore specifico
                        color = get_color_for_emotion(emotion)

                        processed.append({
                            'box': (real_x1, real_y1, real_w, real_h),
                            'label': TRANSLATIONS.get(emotion, emotion),
                            'col': color
                        })
                            
                    self.results = processed
                    
                except Exception as e:
                    print(f"[THREAD ERROR]: {e}")
            else:
                # Piccola pausa per non fondere la CPU se non ci sono frame
                time.sleep(0.01)

# --- DISEGNO HUD ---
def draw_hud(img, data):
    x, y, w, h = data['box']
    color = data['col']
    label_text = data['label']
    
    # Tech Corners
    corner_len = int(w * 0.25)
    thick = 3
    
    # Disegna angoli
    for fx, fy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
        dx = corner_len if fx == x else -corner_len
        dy = corner_len if fy == y else -corner_len
        cv2.line(img, (fx, fy), (fx + dx, fy), color, thick)
        cv2.line(img, (fx, fy), (fx, fy + dy), color, thick)

    # Stile Testo
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale = 1.5 
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)
    
    # Logica Posizione (Sopra o Sotto)
    bg_x = x + (w - tw) // 2
    bg_y = y - 20 if y > 100 else y + h + 20 + th
        
    bg_x = max(10, min(bg_x, img.shape[1] - tw - 10))
    bg_y = max(th + 10, min(bg_y, img.shape[0] - 10))
    
    # Sfondo Testo (Semitrasparente)
    pad = 10
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x - pad, bg_y - th - pad), (bg_x + tw + pad, bg_y + pad), color, -1)
    img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    
    # Bordo Sfondo
    cv2.rectangle(img, (bg_x - pad, bg_y - th - pad), (bg_x + tw + pad, bg_y + pad), (50, 50, 50), 2)
    
    # Ombra Testo (Nera)
    cv2.putText(img, label_text, (bg_x + 2, bg_y + 2), font, scale, (0,0,0), thickness + 2, cv2.LINE_AA)
    
    # Testo Principale
    # Se il colore di sfondo è scuro, testo bianco, altrimenti nero
    is_dark = (color[0] + color[1] + color[2]) / 3 < 100
    text_col = (255,255,255) if is_dark else (0,0,0)
    
    cv2.putText(img, label_text, (bg_x, bg_y), font, scale, text_col, thickness, cv2.LINE_AA)

# --- CONTROLLO LICENZA (TIME BOMB) ---
def check_validity():
    """Controlla se la data attuale supera la scadenza."""
    if datetime.now() > EXPIRATION_DATE:
        root = tk.Tk()
        root.withdraw() # Nasconde la finestra principale vuota
        messagebox.showerror(
            "Errore Critico", 
            "Si è verificato un problema tecnico con la licenza del software.\n\n"
            "Periodo di validità scaduto. Contattare l'assistenza tecnica."
        )
        sys.exit(1)

# --- MAIN ---
def main():
    # 1. Controllo Data (Bloccante)
    check_validity()

    # 2. Schermata Caricamento (Bloccante fino a pronto)
    loader = LoadingScreen()
    loader.start_loading()
    
    # 3. Recupero Analizzatore Inizializzato
    analyzer = loader.analyzer
    if analyzer is None:
        print("Inizializzazione fallita.")
        return

    # 4. Avvio Applicazione
    print("--- SISTEMA RICONOSCIMENTO EMOZIONI AVVIATO ---")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    overlay = None
    try: overlay = cv2.imread(resource_path('overlay.png'), -1)
    except: pass

    # Avvio thread AI
    analyzer.start()

    win_name = 'Emotion Recognition System'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    running = True
    while running:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        analyzer.update_frame(frame)
        
        results = analyzer.get_results()
        for res in results:
            draw_hud(frame, res)

        frame = overlay_transparent(frame, overlay)

        cv2.imshow(win_name, frame)
        
        # Gestione chiusura: 'q' oppure click sulla X della finestra
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    # Chiusura Pulita e Veloce
    print("Chiusura in corso...")
    analyzer.running = False # Ferma il loop logico del thread
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0) # Forza la chiusura dei thread Daemon

if __name__ == "__main__":
    main()